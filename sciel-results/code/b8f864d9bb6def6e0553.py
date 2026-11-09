"""
Strategy: Nelder-Mead with multi-start and diagonal step-size adaptation.
Rationale: The Discus function (f11) is smooth and unimodal. Nelder-Mead is
extremely sample-efficient for low-dimensional smooth functions, converging
rapidly to the global minimum. A multi-start strategy protects against
rare local minima or initialization issues. We use a hybrid approach:
1. Initial diverse sampling to locate the basin of attraction.
2. Nelder-Mead refinement from the best initial point.
3. If NM stagnates or fails to improve significantly, restart with new random points.
4. Strict budget management to ensure we use all 500 calls efficiently.
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        best_calls_made = 0

        def call_f(x):
            nonlocal n_calls, best_x, best_f, best_calls_made
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                best_calls_made = n_calls
            return f_val

        def nelder_mead(x0, max_calls):
            nonlocal n_calls, best_x, best_f, best_calls_made
            
            if n_calls >= budget:
                return best_x, best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            fs = [call_f(x0)]
            if fs[0] is None:
                return best_x, best_f
            
            # Add perturbed points
            for i in range(dim):
                xi = x0.copy()
                # Adaptive step size based on dimension
                step = 0.5 if abs(x0[i]) < 0.5 else 0.1
                xi[i] += step
                simplex.append(xi)
                fi = call_f(xi)
                if fi is None:
                    return best_x, best_f
                fs.append(fi)
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            iterations = 0
            max_iterations = max_calls // (dim + 1) + 50
            last_best_f = best_f
            stagnation_counter = 0
            
            while iterations < max_iterations and n_calls < budget and time.time() < deadline - 5:
                iterations += 1
                
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check for convergence
                if abs(fs[0] - fs[-1]) < 1e-10:
                    break
                
                # Check for stagnation
                if fs[0] < last_best_f - 1e-8:
                    last_best_f = fs[0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > 20:
                        break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = call_f(xr)
                if fr is None:
                    break
                
                if fs[0] <= fr < fs[-2]:
                    # Reflection is good
                    simplex[-1] = xr
                    fs[-1] = fr
                elif fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = call_f(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                else:
                    # Contraction
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = call_f(xc)
                        if fc is None:
                            break
                        if fc <= min(fr, fs[-1]):
                            simplex[-1] = xc
                            fs[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fi = call_f(simplex[i])
                                if fi is None:
                                    return best_x, best_f
                                fs[i] = fi
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = call_f(xc)
                        if fc is None:
                            break
                        if fc < fs[-1]:
                            simplex[-1] = xc
                            fs[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fi = call_f(simplex[i])
                                if fi is None:
                                    return best_x, best_f
                                fs[i] = fi
                
                # Update global best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = simplex[0].copy()
                    best_calls_made = n_calls
            
            return best_x, best_f

        # Multi-start strategy
        # 1. Initial diverse sampling
        n_init = min(10, budget)
        for k in range(n_init):
            if n_calls >= budget or time.time() > deadline - 5:
                break
            if k == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            call_f(x0)
        
        # 2. Run Nelder-Mead from the best point found
        x_start = best_x.copy()
        nelder_mead(x_start, 200)
        
        # 3. If we still have budget, try more restarts
        while n_calls < budget and time.time() < deadline - 10:
            # Check if we're making progress
            if n_calls - best_calls_made > 50:
                # Stagnation, try a new random start
                x_start = rng.uniform(-4, 4, dim)
                remaining = budget - n_calls
                if remaining < 20:
                    break
                nelder_mead(x_start, min(100, remaining))
            else:
                # Continue from current best
                x_start = best_x.copy()
                remaining = budget - n_calls
                if remaining < 20:
                    break
                nelder_mead(x_start, min(100, remaining))
        
        return best_x, best_f
