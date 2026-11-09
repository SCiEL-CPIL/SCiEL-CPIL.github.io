"""Hybrid Differential Evolution + Nelder-Mead Optimizer for BBOB f7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """
    Hybrid optimizer for BBOB f7 (Step ellipsoidal).
    Uses Differential Evolution for global search and Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        total_budget = self.budget
        
        # State tracking
        calls = [0]
        best_val = [np.inf]
        best_x = [np.zeros(n)]

        def safe_call(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls[0] >= total_budget:
                raise MemoryError("Budget exhausted")
            calls[0] += 1
            val = func(x)
            if val < best_val[0]:
                best_val[0] = val
                best_x[0] = x.copy()
            return val

        # --- Phase 1: Global Search with Differential Evolution ---
        # Allocate ~85% of the budget to DE
        de_budget = int(total_budget * 0.85)
        
        bounds = [(-5.0, 5.0)] * n
        
        # DE callback to stop if we hit the budget or time limit
        def de_callback(xk, convergence):
            if calls[0] >= de_budget:
                return True
            if time.time() > deadline:
                return True
            return False

        try:
            # popsize=15 is a good balance for dim=5. 
            # DE uses popsize * dim evaluations per generation roughly, but actually popsize evaluations per generation.
            # 500 calls / 15 popsize = ~33 generations.
            # We limit maxiter high, but callback stops it.
            res = differential_evolution(
                safe_call, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-10, 
                init='latinhypercube', 
                mutation=(0.5, 1.0), 
                recombination=0.5, 
                seed=42,
                callback=de_callback,
                polish=False # We do our own polish
            )
            
            if res is not None and res.fun < best_val[0]:
                best_x[0] = res.x
                best_val[0] = res.fun
            elif res is not None:
                # Ensure we at least have the best from the wrapper
                pass
                
        except (TimeoutError, MemoryError):
            pass
        except Exception:
            # Fallback if DE fails completely
            pass

        # --- Phase 2: Local Refinement with Nelder-Mead ---
        # Use remaining budget for NM
        remaining_budget = total_budget - calls[0]
        
        if remaining_budget > 10 and time.time() < deadline:
            x_start = best_x[0].copy()
            
            # Simple Nelder-Mead implementation
            def nelder_mead(x0, max_iter=50):
                if time.time() > deadline or calls[0] >= total_budget:
                    return x0, best_val[0]

                # Initialize simplex
                x = np.array(x0, dtype=float)
                f0 = safe_call(x)
                simplex = [x.copy()]
                fvals = [f0]

                # Create initial simplex
                for i in range(n):
                    xi = x.copy()
                    # Small initial step
                    step = 0.1 if abs(xi[i]) < 0.1 else 0.1 * abs(xi[i])
                    xi[i] += step
                    xi = np.clip(xi, -5, 5)
                    fi = safe_call(xi)
                    simplex.append(xi)
                    fvals.append(fi)

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma = 0.5   # shrink

                for it in range(max_iter):
                    if time.time() > deadline or calls[0] >= total_budget:
                        break

                    # Check convergence
                    if abs(fvals[0] - fvals[-1]) < 1e-12:
                        break

                    # Centroid of all points except the worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_call(xr)

                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = safe_call(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        # Contraction
                        if fr < fvals[-1]:
                            # Outside contraction
                            xc = centroid + rho * (xr - centroid)
                        else:
                            # Inside contraction
                            xc = centroid - rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = safe_call(simplex[i])

                    # Sort again
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                return simplex[0], fvals[0]

            try:
                nm_x, nm_f = nelder_mead(x_start, max_iter=50)
                if nm_f < best_val[0]:
                    best_x[0] = nm_x
                    best_val[0] = nm_f
            except (TimeoutError, MemoryError):
                pass
            except Exception:
                pass

        return best_x[0], best_val[0]
# EVOLVE-BLOCK-END
