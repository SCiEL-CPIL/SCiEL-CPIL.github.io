import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower, upper = -5.0, 5.0
        
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        calls_made = 0
        best_x = None
        best_f = float('inf')
        all_results = []  # list of (f, x)

        def track(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lower, upper)
            
            try:
                f = func(x)
            except Exception as e:
                raise TimeoutError("Internal error or budget") from e
            
            calls_made += 1
            all_results.append((f, x.copy()))
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            return f

        # Phase 1: Coarse Grid Search
        # Minimizer is in [-4, 4]^5.
        # We use a 3-level grid: -4, 0, 4.
        # 3^5 = 243 points.
        # Sort by distance to origin to maximize early AOCC (center is likely good).
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=dim))
        origin = np.zeros(dim)
        grid_points.sort(key=lambda p: np.linalg.norm(np.array(p) - origin))

        for p in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            try:
                track(np.array(p, dtype=np.float64))
            except (TimeoutError, ValueError):
                break
            except Exception:
                break

        # Phase 2: Local Refinement
        # We have used ~243 calls. Remaining: ~257 calls.
        # Strategy: Use a fast local optimizer (Nelder-Mead style or BFGS) on the best few points.
        # Since BBOB f22 is multimodal, refining the global best found in grid is crucial.
        # Also, we might have missed the true global min if it's in a narrow basin not hit by the grid.
        # However, grid is quite dense.
        
        # Let's implement a simple, robust Nelder-Mead that respects the budget and time.
        # We will run it on the best point, and potentially a few other top points if budget allows.
        
        def nelder_mead(x0, max_evals):
            nonlocal calls_made, best_x, best_f, all_results
            
            if calls_made >= budget or time.time() > deadline:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                # Step size: 0.5 if positive, -0.5 if negative? 
                # Better: use a small fraction of the box size or absolute step.
                # Box is 10. Step 0.5 is reasonable.
                step = 0.5 if x0[i] < 0 else -0.5
                xi[i] += step
                simplex.append(np.clip(xi, lower, upper))
            
            # Evaluate simplex
            fvals = []
            for x in simplex:
                if calls_made >= budget or time.time() > deadline:
                    return
                try:
                    fvals.append(track(x))
                except (TimeoutError, ValueError):
                    return
                except Exception:
                    return
            
            if len(fvals) < dim + 1:
                return
            
            # Main loop
            for _ in range(max_evals):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Convergence check
                if max(fvals) - min(fvals) < 1e-10:
                    break
                if np.max(np.std(simplex, axis=0)) < 1e-7:
                    break
                
                x_worst = simplex[-1]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, lower, upper)
                try:
                    f_ref = track(x_ref)
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    break
                
                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = x_centroid + gamma * (x_centroid - x_worst)
                    x_exp = np.clip(x_exp, lower, upper)
                    try:
                        f_exp = track(x_exp)
                    except (TimeoutError, ValueError):
                        break
                    except Exception:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, lower, upper)
                    try:
                        f_con = track(x_con)
                    except (TimeoutError, ValueError):
                        break
                    except Exception:
                        break
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            simplex[i] = (simplex[0] + simplex[i]) / 2
                            simplex[i] = np.clip(simplex[i], lower, upper)
                            try:
                                fvals[i] = track(simplex[i])
                            except (TimeoutError, ValueError):
                                break
                            except Exception:
                                break
        
        # Identify top candidates from grid search
        # Sort all_results by f
        all_results.sort(key=lambda t: t[0])
        
        # Refine the best point
        if all_results and best_x is not None:
            remaining_budget = budget - calls_made
            # Allocate a significant chunk to the best point
            nm_budget_best = max(20, int(remaining_budget * 0.5))
            nelder_mead(best_x, nm_budget_best)
            
            # If we still have budget, refine the 2nd and 3rd best points
            if calls_made < budget - 10 and time.time() < deadline:
                for i in range(1, min(3, len(all_results))):
                    if calls_made >= budget - 10 or time.time() > deadline:
                        break
                    # Only refine if this point is significantly better than the current best?
                    # Or just refine it to see if it leads to a better basin.
                    # If f is already worse than best_f, it might still lead to a better local min?
                    # Unlikely if we are near the global min. But for multimodal, maybe.
                    # Let's refine the 2nd best if it's close to the best.
                    if all_results[i][0] < best_f + 10.0: # Threshold
                        nm_budget_other = max(10, int((budget - calls_made) * 0.25))
                        nelder_mead(all_results[i][1], nm_budget_other)

        return best_x, best_f
