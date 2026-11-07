import numpy as np
import time
import scipy.optimize as opt

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We aim to finish well before.
        deadline = start_time + 280.0 
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            # Check time and budget limits
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration
            
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                # Fallback for shape mismatches
                if len(x) > dim:
                    x = x[:dim]
                else:
                    # Pad with zeros if too short (shouldn't happen with NM)
                    x_new = np.zeros(dim)
                    x_new[:len(x)] = x
                    x = x_new
            
            # Clip to valid domain [-5, 5]
            x = np.clip(x, -5.0, 5.0)
            
            try:
                val = float(func(x))
            except Exception:
                val = np.inf

            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration via Multi-Start Nelder-Mead
        # Strategy: Use diverse starting points and run short, low-precision NM
        # to quickly jump to local minima. This exploits the multimodal nature.
        
        # Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Structured points: Center, Corners, Axes
        start_points = []
        start_points.append(np.zeros(dim))
        start_points.append(np.ones(dim) * 4.0)
        start_points.append(np.ones(dim) * -4.0)
        
        # Axis-aligned points
        for d in range(dim):
            for v in [3.0, -3.0, 1.0, -1.0]:
                x = np.zeros(dim)
                x[d] = v
                start_points.append(x)
        
        # Random points to cover the space
        n_random_starts = 15
        for _ in range(n_random_starts):
            start_points.append(rng.uniform(-4.5, 4.5, size=dim))
        
        # Limit total starts to ensure we have budget for refinement
        # Each short NM run will consume some calls. 
        # Let's allocate ~50% of budget to exploration if possible, but cap starts.
        # A short NM with maxiter=5 in 5D takes ~30-50 calls.
        # 10-15 starts * 40 calls = 400-600 calls. 
        # We have 500 calls. So we should be conservative.
        # Let's do 8 starts with very low maxiter.
        
        max_short_iter = 3 # Very short, just to find the basin
        xatol_short = 1e-3
        fatol_short = 1e-3
        
        active_starts = start_points[:8] # Cap to 8 starts to save budget
        
        for x0 in active_starts:
            if calls >= budget or time.time() > deadline:
                break
            
            try:
                res = opt.minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_short_iter,
                        'xatol': xatol_short,
                        'fatol': fatol_short
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        if best_x is None:
            # Fallback if no calls were made or all failed
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except Exception:
                best_f = func(best_x)
                calls += 1

        # Phase 2: Global Refinement from the best point found
        # Use a medium-precision Nelder-Mead to converge closer to the minimum.
        
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                remaining = budget - calls
                # Estimate maxiter: NM uses dim+1 calls per iteration roughly
                # But we want to leave room for Phase 3.
                # Let's use half the remaining budget for Phase 2.
                alloc_p2 = max(10, remaining // 2)
                est_maxiter_p2 = max(1, alloc_p2 // (dim + 1))
                
                res = opt.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': est_maxiter_p2,
                        'xatol': 1e-6,
                        'fatol': 1e-6
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        # Phase 3: High-Precision Polish
        # If budget remains, run a tight NM to finalize.
        
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                remaining = budget - calls
                if remaining >= 5:
                    est_maxiter_p3 = max(1, remaining // (dim + 1))
                    
                    res = opt.minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': est_maxiter_p3,
                            'xatol': 1e-10,
                            'fatol': 1e-10
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except Exception:
                best_f = func(best_x)

        return best_x, best_f
