"""Optimized Katsuura (f23) minimizer with lattice-based basin identification and tiered local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23).
    
    Strategy:
    1. Lattice-based coarse sampling to identify distinct basins efficiently.
    2. Tiered local search:
       - Phase 1: Quick Nelder-Mead on top distinct basins.
       - Phase 2: Deeper search on the best basin found.
       - Phase 3: Final high-precision polish on the global best.
    3. Time and budget guards to ensure valid results within constraints.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Use a fixed seed for reproducibility within a single instance
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # 1. Lattice-based coarse sampling
        # Use a structured lattice to cover the search space more uniformly than random sampling.
        # This helps identify distinct basins more reliably for oscillatory functions like Katsuura.
        
        init_points = []
        
        # Priority 1: The origin
        init_points.append(np.zeros(dim))
        
        # Priority 2: Structured lattice sampling
        # Use a 3-point lattice per dimension for coarse coverage
        lattice_coords = np.array([-3.0, 0.0, 3.0])
        
        # Generate lattice points (3^5 = 243 points, but we'll limit to ~30 for budget)
        # Use a subset of the full lattice to stay within budget
        lattice_indices = list(range(3 ** dim))
        rng.shuffle(lattice_indices)
        
        # Take a reasonable number of lattice points
        num_lattice = min(30, len(lattice_indices))
        for idx in lattice_indices[:num_lattice]:
            # Convert multi-index to coordinates
            coords = []
            temp = idx
            for d in range(dim):
                coords.append(lattice_coords[temp % 3])
                temp //= 3
            p = np.array(coords, dtype=float)
            # Add small random jitter to avoid exact lattice alignment
            p = p + rng.normal(0, 0.1, dim)
            init_points.append(np.clip(p, -5, 5))
        
        # Priority 3: Additional random points for diversity
        for _ in range(10):
            p = rng.uniform(-4, 4, dim)
            init_points.append(p)
        
        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort indices by function value (ascending)
        sorted_indices = np.argsort(init_values)
        
        # Identify distinct basins: select top candidates that are sufficiently far apart
        # This ensures we explore different regions of the search space
        basin_candidates = []
        min_distance = 1.0  # Minimum distance to consider points as distinct basins
        
        for i in sorted_indices:
            if len(basin_candidates) >= 5:  # Limit to 5 distinct basins
                break
            candidate = init_points[i]
            is_distinct = True
            for existing in basin_candidates:
                if np.linalg.norm(candidate - existing) < min_distance:
                    is_distinct = False
                    break
            if is_distinct:
                basin_candidates.append(i)
        
        # 2. Tiered Local Search on distinct basins
        
        # Tier 1: Deep search on the top 3 distinct basins
        top_k_tier1 = min(3, len(basin_candidates))
        if calls < self.budget - 80:
            for i in range(top_k_tier1):
                seed_idx = basin_candidates[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    # Medium-Long Nelder-Mead to converge in the basin
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 60, 
                                       'xatol': 1e-6, 
                                       'fatol': 1e-8, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 2: Medium search on the next 2-3 distinct basins
        top_k_tier2 = min(3, len(basin_candidates) - top_k_tier1)
        if calls < self.budget - 50:
            for i in range(top_k_tier1, min(len(basin_candidates), top_k_tier1 + top_k_tier2)):
                seed_idx = basin_candidates[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 25, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 3: Final high-precision polish on the global best
        if calls < self.budget - 5 and best_x is not None:
            try:
                # Very tight tolerances for final convergence
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={
                                   'maxiter': 150, 
                                   'xatol': 1e-10, 
                                   'fatol': 1e-12, 
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Additional restarts if budget remains
        # Try a few more random restarts with short local search
        if calls < self.budget - 20 and best_x is not None:
            for _ in range(3):
                if time.time() - start_time > time_limit - 10.0:
                    break
                if calls >= self.budget - 20:
                    break
                # Restart near the current best with some perturbation
                perturbation = rng.normal(0, 0.2, dim)
                seed = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 15, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
