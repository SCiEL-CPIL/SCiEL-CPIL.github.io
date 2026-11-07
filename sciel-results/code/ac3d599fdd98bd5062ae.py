"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = float('inf')

    def _evaluate(self, x, func):
        """Evaluate function with budget tracking."""
        if self._call_count >= self.budget:
            return float('inf')
        x = np.clip(x, -5, 5)
        f = func(x)
        self._call_count += 1
        if f < self._best_f:
            self._best_f = f
            self._best_x = x.copy()
        return f

    def _nelder_mead(self, x0, func, max_calls):
        """
        Custom Nelder-Mead implementation with strict call budget.
        """
        dim = self.dim
        # Initialize simplex
        # Start with x0 and perturb each dimension
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            # Perturbation size: 0.05 * range (10) = 0.5, or smaller for Weierstrass?
            # Weierstrass has features at multiple scales. 0.1 is a safe start.
            simplex[i + 1, i] += 0.1 if simplex[i + 1, i] < 4.9 else -0.1
            simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
        
        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        for i in range(dim + 1):
            if self._call_count >= self.budget or self._call_count >= max_calls:
                return
            fvals[i] = self._evaluate(simplex[i], func)

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        calls_in_this_run = 0
        max_iterations = max_calls # Rough upper bound on iterations
        
        while calls_in_this_run < max_iterations and self._call_count < self.budget:
            # Sort simplex by fitness
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            if fvals[0] - fvals[-1] < 1e-8:
                break # Converged

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = self._evaluate(xr, func)
            calls_in_this_run += 1
            if self._call_count >= self.budget:
                break

            if fr < fvals[0]:
                # 2. Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = self._evaluate(xe, func)
                calls_in_this_run += 1
                if self._call_count >= self.budget:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # 3. Contraction
                if fr < fvals[-1]:
                    # Outer contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inner contraction
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                fc = self._evaluate(xc, func)
                calls_in_this_run += 1
                if self._call_count >= self.budget:
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # 4. Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if self._call_count >= self.budget:
                            break
                        fvals[i] = self._evaluate(simplex[i], func)
                        calls_in_this_run += 1
                    if self._call_count >= self.budget:
                        break

        # Update global best
        if self._best_x is None:
             self._best_x = simplex[0]
             self._best_f = fvals[0]

    def __call__(self, func):
        self._call_count = 0
        self._best_f = float('inf')
        self._best_x = None
        
        # Phase 1: Global Search / Initialization
        # Use 20% of budget for random sampling to find good basins
        n_init = max(10, self.budget // 5)
        # Ensure we don't exceed budget
        n_init = min(n_init, self.budget)
        
        init_points = []
        init_fvals = []
        
        # Include center
        center = np.zeros(self.dim)
        f_center = self._evaluate(center, func)
        init_points.append(center)
        init_fvals.append(f_center)
        
        # Random points
        for _ in range(n_init - 1):
            if self._call_count >= self.budget:
                break
            x = np.random.uniform(-5, 5, self.dim)
            f = self._evaluate(x, func)
            init_points.append(x)
            init_fvals.append(f)
        
        if not init_points:
            return None, float('inf')

        # Sort by fitness and pick top K
        sorted_idx = np.argsort(init_fvals)
        top_k = min(5, len(sorted_idx))
        best_starts = [init_points[i] for i in sorted_idx[:top_k]]
        
        # Phase 2: Local Refinement
        # Distribute remaining budget among top starts
        remaining_budget = self.budget - self._call_count
        if remaining_budget <= 0:
            best_idx = np.argmin(init_fvals)
            return init_points[best_idx], init_fvals[best_idx]
        
        budget_per_start = remaining_budget // top_k
        
        for start in best_starts:
            if self._call_count >= self.budget:
                break
            # Allocate a portion of remaining budget to each
            # We pass a max_calls limit to Nelder-Mead
            # Note: Nelder-Mead might use fewer calls if it converges
            self._nelder_mead(start, func, budget_per_start)
            
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
