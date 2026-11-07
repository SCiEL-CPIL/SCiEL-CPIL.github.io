import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Global Random Search followed by Nelder-Mead Local Refinement.
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(1234)
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Random Search
        # Use more samples to better locate the basin in 5D
        n_random = 30
        
        random_points = []
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break
            random_points.append((f_val, x.copy()))
        
        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Determine adaptive step size based on spread of good random points
        # Use the top 5 points to estimate local scale
        if len(random_points) >= 5:
            random_points.sort(key=lambda t: t[0])
            top_points = np.array([p[1] for p in random_points[:5]])
            # Estimate step size from spread of top points
            spread = np.max(top_points, axis=0) - np.min(top_points, axis=0)
            step_size = max(0.1, min(1.0, np.mean(spread) * 0.5))
        else:
            step_size = 0.5

        # Phase 2: Nelder-Mead with restarts
        max_restarts = 3
        restart_count = 0
        
        while restart_count < max_restarts and calls < budget and time.time() < deadline:
            # Start NM from current best
            x_start = best_x.copy()
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Initialize simplex around x_start
            simplex = [x_start]
            f_values = [best_f]
            
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                xi = x_start.copy()
                xi[i] += step_size
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                f_values.append(fi)
            
            if len(simplex) < dim + 1:
                break
            
            simplex = np.array(simplex)
            f_values = np.array(f_values)
            
            nm_calls = 0
            max_nm_calls = 150  # Limit NM iterations per restart
            
            while calls < budget and nm_calls < max_nm_calls and time.time() < deadline:
                # Sort simplex by function value (best first)
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                # Update global best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                
                # Check for convergence
                spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                spread_f = np.max(f_values) - np.min(f_values)
                
                if np.max(spread_x) < 1e-7 or spread_f < 1e-12:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_call(xr)
                if fr is None:
                    break
                nm_calls += 1
                
                if fr < f_values[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    nm_calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_values[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fr
                elif fr < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    f_values[-1] = fr
                else:
                    # Contraction
                    if fr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    nm_calls += 1
                    if fc <= min(fr, f_values[-1]):
                        simplex[-1] = xc
                        f_values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            f_values[i] = fi
                            nm_calls += 1
                        if fi is None:
                            break
            
            # If we have budget left, try a restart with perturbation
            if calls < budget - 10 and time.time() < deadline:
                # Perturb the best point slightly to escape potential local minima
                perturbation = rng.uniform(-0.3, 0.3, size=dim)
                x_perturbed = best_x + perturbation
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                f_perturbed = safe_call(x_perturbed)
                if f_perturbed is None:
                    break
                # Only restart if perturbed point is reasonably good
                if f_perturbed < best_f + 1.0:
                    best_x = x_perturbed.copy()
                    best_f = f_perturbed
                    restart_count += 1
                else:
                    break
            else:
                break

        return best_x, best_f
