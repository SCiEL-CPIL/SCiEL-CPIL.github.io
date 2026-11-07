"""Optimizer for BBOB f15 (Rastrigin rotated) using lattice-based candidate generation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Lattice-based optimizer exploiting the 2π-periodicity of Rastrigin in the rotated frame.
    
    Strategy:
    1. Initial exploration: evaluate at a diverse set of points to get initial best estimates.
    2. Lattice search: The rotated Rastrigin f15 has minima at integer multiples of 2π in the
       rotated coordinate system. We generate candidate minima by exploring shifts of ±2π, ±4π
       along various directions from the current best point.
    3. Refine: Do a small local search around the best candidate to find the exact minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

    def _counted_func(self, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return self.func(x)

    def _make_key(self, x):
        """Create a hashable key for a point."""
        return tuple(np.round(x, 6))

    def _evaluate(self, x):
        """Evaluate a point if not already evaluated and within budget."""
        if self.calls >= self.budget:
            return self.best_f
        x = np.clip(x, -5, 5)
        key = self._make_key(x)
        if key in self.evaluated:
            return self.best_f
        self.evaluated.add(key)
        fx = self._counted_func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def _generate_lattice_candidates(self, best_x, n_candidates=80):
        """
        Generate candidate minima by shifting best_x by ±2π, ±4π along various directions.
        This exploits the periodicity of Rastrigin in the rotated frame.
        """
        dim = self.dim
        period = 2 * np.pi
        
        candidates = []
        
        # Axis directions
        for d in range(dim):
            e = np.zeros(dim)
            e[d] = 1.0
            for t in [-4 * np.pi, -2 * np.pi, 2 * np.pi, 4 * np.pi]:
                x_cand = best_x + t * e
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
        
        # Diagonal directions (pairs of axes)
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                e = np.zeros(dim)
                e[d1] = 1.0 / np.sqrt(2)
                e[d2] = 1.0 / np.sqrt(2)
                for t in [-2 * np.pi, 2 * np.pi]:
                    x_cand = best_x + t * e
                    x_cand = np.clip(x_cand, -5, 5)
                    candidates.append(x_cand)
        
        # Combinations of two axis shifts
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for t1 in [-2 * np.pi, 2 * np.pi]:
                    for t2 in [-2 * np.pi, 2 * np.pi]:
                        x_cand = best_x.copy()
                        x_cand[d1] += t1
                        x_cand[d2] += t2
                        x_cand = np.clip(x_cand, -5, 5)
                        candidates.append(x_cand)
        
        # Sort by distance to best_x and return top n
        candidates.sort(key=lambda x: np.linalg.norm(x - best_x))
        return candidates[:n_candidates]

    def _local_refine(self, x_center, radius, n_evals):
        """
        Do a small coordinate-descent local search around x_center.
        """
        dim = self.dim
        best_x = x_center.copy()
        best_f = self._evaluate(best_x)
        step = radius
        n_used = 1
        
        while n_used < n_evals and step > 0.05:
            improved = False
            for d in range(dim):
                if n_used >= n_evals:
                    break
                # Try positive step
                x_try = best_x.copy()
                x_try[d] += step
                x_try = np.clip(x_try, -5, 5)
                f_try = self._evaluate(x_try)
                n_used += 1
                if f_try < best_f:
                    best_x = x_try
                    best_f = f_try
                    improved = True
                    continue
                
                # Try negative step
                if n_used >= n_evals:
                    break
                x_try = best_x.copy()
                x_try[d] -= step
                x_try = np.clip(x_try, -5, 5)
                f_try = self._evaluate(x_try)
                n_used += 1
                if f_try < best_f:
                    best_x = x_try
                    best_f = f_try
                    improved = True
            
            if not improved:
                step *= 0.5
        
        return best_x, best_f, n_used

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration - diverse set of points
        n_initial = min(25, budget // 5)
        initial_points = []
        
        # Origin
        initial_points.append(np.zeros(dim))
        
        # Axis points at ±2
        for d in range(dim):
            e = np.zeros(dim)
            e[d] = 2.0
            initial_points.append(e)
            initial_points.append(-e)
        
        # Random points to fill up to n_initial
        while len(initial_points) < n_initial:
            initial_points.append(rng.uniform(-5, 5, dim))
        
        for x0 in initial_points:
            if self.calls >= budget - 10:
                break
            self._evaluate(x0)

        # Phase 2: Lattice-based candidate generation and evaluation
        if self.best_x is not None:
            candidates = self._generate_lattice_candidates(self.best_x, n_candidates=100)
            
            # Evaluate candidates in order of proximity to best
            for x_cand in candidates:
                if self.calls >= budget - 10:
                    break
                self._evaluate(x_cand)
        
        # Phase 3: Local refinement around the best point found
        if self.best_x is not None and self.calls < budget - 5:
            remaining = budget - self.calls - 2
            if remaining > 0:
                # First refinement pass with larger radius
                n_refine1 = min(remaining, 25)
                self.best_x, self.best_f, _ = self._local_refine(
                    self.best_x, radius=1.0, n_evals=n_refine1
                )
                
                # Second refinement pass with smaller radius if budget allows
                if self.calls < budget - 5:
                    remaining = budget - self.calls - 2
                    if remaining > 0:
                        n_refine2 = min(remaining, 15)
                        self.best_x, self.best_f, _ = self._local_refine(
                            self.best_x, radius=0.3, n_evals=n_refine2
                        )

        # Ensure we've made at least one call
        if self.calls == 0:
            x0 = np.zeros(dim)
            self._evaluate(x0)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
