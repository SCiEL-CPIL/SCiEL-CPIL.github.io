import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.archive_F = []
        self.evals_used = 0

    def _compute_hypervolume(self, F, ref):
        """Compute hypervolume for 3-objective case"""
        if len(F) == 0:
            return 0.0
        # Filter points strictly better than reference
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Sort by first objective
        idx = np.argsort(F[:, 0])
        F = F[idx]
        
        hv = 0.0
        prev = np.array([ref[1], ref[2]])
        
        for i in range(len(F)):
            curr = F[i, 1:]
            # Volume of box from curr to prev
            if np.all(prev > curr):
                vol = (prev[0] - curr[0]) * (prev[1] - curr[1])
                hv += vol
            prev = curr
        
        return hv

    def _select_diverse_point(self, F, ref):
        """Select point that contributes most to hypervolume"""
        if len(F) == 0:
            return None
        if len(F) == 1:
            return 0
        
        # Compute marginal hypervolume contribution
        base_hv = self._compute_hypervolume(F, ref)
        best_idx = 0
        best_hv = 0
        
        for i in range(len(F)):
            F_without_i = np.delete(F, i, axis=0)
            hv_without = self._compute_hypervolume(F_without_i, ref)
            contribution = base_hv - hv_without
            if contribution > best_hv:
                best_hv = contribution
                best_idx = i
        
        return best_idx

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        # 1. Initial Sampling - use more points for better spread
        init_size = min(60, self.budget)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(init_size)
        X_init = lo + points * (hi - lo)
        
        F_init = np.array([problem(x) for x in X_init])
        self.evals_used = init_size
        
        # Filter to only keep points better than reference
        valid_mask = np.all(F_init < ref, axis=1)
        X_init = X_init[valid_mask]
        F_init = F_init[valid_mask]
        
        if len(F_init) > 0:
            self.archive = X_init
            self.archive_F = F_init
        else:
            self.archive = X_init[:1] if len(X_init) > 0 else np.array([np.zeros(self.dim)])
            self.archive_F = F_init[:1] if len(F_init) > 0 else np.array([[ref[0]+1, ref[1]+1, ref[2]+1]])
        
        # 2. Adaptive Refinement with hypervolume-based selection
        rng = np.random.default_rng(123)
        
        while self.evals_used < self.budget:
            if len(self.archive_F) == 0:
                # No valid points, do random sampling
                x_new = lo + rng.random(self.dim) * (hi - lo)
                f_new = problem(x_new)
                self.evals_used += 1
                if np.all(f_new < ref):
                    self.archive = np.array([x_new])
                    self.archive_F = np.array([f_new])
                continue
            
            # Select point with highest marginal hypervolume contribution
            idx = self._select_diverse_point(self.archive_F, ref)
            x_current = self.archive[idx]
            f_current = self.archive_F[idx]
            
            # Adaptive step size based on current position
            step_size = 0.05 * (hi - lo)
            
            # Try multiple directions
            best_x = None
            best_f = None
            best_hv_gain = -1
            
            n_trials = min(5, self.budget - self.evals_used)
            for _ in range(n_trials):
                # Random direction in unit sphere
                direction = rng.normal(size=self.dim)
                direction /= np.linalg.norm(direction) + 1e-10
                
                # Also try axis-aligned directions
                if rng.random() < 0.5:
                    dim_idx = rng.integers(0, self.dim)
                    sign = rng.choice([-1, 1])
                    direction = np.zeros(self.dim)
                    direction[dim_idx] = sign
                
                x_new = x_current + step_size * direction
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                self.evals_used += 1
                
                if self.evals_used >= self.budget:
                    break
                
                # Check if this point improves hypervolume
                if np.all(f_new < ref):
                    # Compute hypervolume with new point
                    F_test = np.vstack([self.archive_F, [f_new]])
                    hv_with = self._compute_hypervolume(F_test, ref)
                    hv_without = self._compute_hypervolume(self.archive_F, ref)
                    gain = hv_with - hv_without
                    
                    if gain > best_hv_gain:
                        best_hv_gain = gain
                        best_x = x_new
                        best_f = f_new
            
            # Add best point if it improves
            if best_x is not None and best_hv_gain > 0:
                self.archive = np.vstack([self.archive, best_x])
                self.archive_F = np.vstack([self.archive_F, best_f])
                
                # Remove dominated points
                n = len(self.archive_F)
                is_dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        if np.all(self.archive_F[j] <= self.archive_F[i]) and np.any(self.archive_F[j] < self.archive_F[i]):
                            is_dominated[i] = True
                            break
                mask = ~is_dominated
                self.archive = self.archive[mask]
                self.archive_F = self.archive_F[mask]
            
            # Adapt step size
            step_size *= 0.95
        
        return self.archive
