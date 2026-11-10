"""Multi-objective optimizer on WFG4 using a deterministic Sobol sequence with adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_sequence(n, dim, seed=0):
    """Generate n points in [0,1]^dim using a simple Halton-like quasi-random sequence."""
    # Use a Halton sequence for simplicity and determinism
    points = np.zeros((n, dim))
    # Prime bases for each dimension
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    for d in range(dim):
        base = primes[d % len(primes)]
        for i in range(n):
            f = 1.0
            r = 0.0
            j = i + 1
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            points[i, d] = r
    return points


class Optimizer:
    """Deterministic grid-based sampling with adaptive local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Phase 1: Uniform coverage using Halton sequence
        # Use about 60% of budget for initial exploration
        n_explore = int(self.budget * 0.6)
        
        # Generate Halton points
        halton_points = _sobol_sequence(n_explore, self.dim)
        
        # Map to bounds
        X_explore = lo + (hi - lo) * halton_points
        
        # Evaluate
        F_explore = np.array([problem(x) for x in X_explore])
        used = n_explore
        
        # Check deadline
        if time.time() > deadline:
            return X_explore
        
        # Phase 2: Adaptive local refinement around best points
        # Find non-dominated points
        n_explore_pts = len(F_explore)
        non_dominated_mask = np.zeros(n_explore_pts, dtype=bool)
        for i in range(n_explore_pts):
            dominated = False
            for j in range(n_explore_pts):
                if i == j:
                    continue
                if np.all(F_explore[j] <= F_explore[i]) and np.any(F_explore[j] < F_explore[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated_mask[i] = True
        
        # Get non-dominated points
        nd_indices = np.where(non_dominated_mask)[0]
        
        # If no non-dominated points found (shouldn't happen), use all
        if len(nd_indices) == 0:
            nd_indices = np.arange(n_explore_pts)
        
        # Phase 2: Local refinement around top non-dominated points
        # Use remaining budget for local search
        n_refine = self.budget - used
        if n_refine <= 0:
            return X_explore
        
        # Select top K non-dominated points for refinement
        K = min(10, len(nd_indices))
        # Sort by sum of objectives (simple proxy)
        sums = F_explore[nd_indices].sum(axis=1)
        top_k_local = nd_indices[np.argsort(sums)[:K]]
        
        # Generate local perturbations
        # For each top point, generate several nearby points
        points_per_center = max(1, n_refine // K)
        
        X_refine = []
        F_refine = []
        
        rng = np.random.default_rng(42)
        
        for center_idx in top_k_local:
            center = X_explore[center_idx]
            center_f = F_explore[center_idx]
            
            for _ in range(points_per_center):
                if time.time() > deadline:
                    break
                if len(X_refine) >= n_refine:
                    break
                    
                # Adaptive step size: start small, based on spread of non-dominated points
                # Use a fraction of the range
                spread = (hi - lo)
                step_size = 0.05 * spread  # 5% of range
                
                # Generate perturbation
                perturbation = rng.standard_normal(self.dim) * step_size
                new_point = center + perturbation
                new_point = np.clip(new_point, lo, hi)
                
                # Evaluate
                new_f = problem(new_point)
                X_refine.append(new_point)
                F_refine.append(new_f)
        
        if time.time() > deadline:
            # Combine what we have
            if X_refine:
                X_all = np.vstack([X_explore, np.array(X_refine)])
                F_all = np.vstack([F_explore, np.array(F_refine)])
            else:
                X_all = X_explore
                F_all = F_explore
            return X_all
        
        # Phase 3: Final polish - evaluate a few more points in promising regions
        # Combine all points
        if X_refine:
            X_all = np.vstack([X_explore, np.array(X_refine)])
            F_all = np.vstack([F_explore, np.array(F_refine)])
        else:
            X_all = X_explore
            F_all = F_explore
        
        # Find current non-dominated set
        n_all = len(F_all)
        non_dominated_mask_all = np.zeros(n_all, dtype=bool)
        for i in range(n_all):
            dominated = False
            for j in range(n_all):
                if i == j:
                    continue
                if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated_mask_all[i] = True
        
        nd_indices_all = np.where(non_dominated_mask_all)[0]
        
        if len(nd_indices_all) == 0:
            nd_indices_all = np.arange(n_all)
        
        # Use remaining budget for very fine local search
        remaining = self.budget - (used + len(X_refine))
        if remaining > 0 and time.time() < deadline:
            # Select top 5 for very fine refinement
            K2 = min(5, len(nd_indices_all))
            sums_all = F_all[nd_indices_all].sum(axis=1)
            top_k2_local = nd_indices_all[np.argsort(sums_all)[:K2]]
            
            points_per_center2 = max(1, remaining // K2)
            
            X_polish = []
            F_polish = []
            
            for center_idx in top_k2_local:
                center = X_all[center_idx]
                
                for _ in range(points_per_center2):
                    if time.time() > deadline:
                        break
                    if len(X_polish) >= remaining:
                        break
                    
                    # Very small step
                    step_size = 0.01 * (hi - lo)
                    perturbation = rng.standard_normal(self.dim) * step_size
                    new_point = center + perturbation
                    new_point = np.clip(new_point, lo, hi)
                    
                    new_f = problem(new_point)
                    X_polish.append(new_point)
                    F_polish.append(new_f)
            
            if X_polish:
                X_all = np.vstack([X_all, np.array(X_polish)])
                F_all = np.vstack([F_all, np.array(F_polish)])
        
        return X_all
# EVOLVE-BLOCK-END
