"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex method for BBOB-style continuous optimization.
    
    Uses a standard Nelder-Mead implementation with adaptive step sizes and 
    restarts based on stall detection.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        # Initial diverse seeds to find a good starting point for the simplex
        n_seeds = min(5, budget // 10)
        seeds = [rng.uniform(-5.0, 5.0, dim) for _ in range(n_seeds)]
        for s in seeds:
            f = func(s)
            record(s, f)
            if n_calls >= budget or time.time() > deadline:
                break

        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        # Start Nelder-Mead from the current best point
        x0 = x_best.copy()
        
        # Initialize simplex: 1 point at x0, others at x0 + small perturbation
        delta = 0.5
        simplex = [x0]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(np.clip(xi, -5.0, 5.0))
        
        # Evaluate initial simplex
        f_vals = []
        for x in simplex:
            f = func(x)
            record(x, f)
            f_vals.append(f)
            if n_calls >= budget or time.time() > deadline:
                break
        
        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        f_vals = np.array(f_vals)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        stall_count = 0
        max_stall = 20
        
        while n_calls < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = f_vals[order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            record(xr, fr)
            
            if n_calls >= budget or time.time() > deadline:
                break
            
            if f_vals[0] <= fr < f_vals[-2]:
                # Accept reflection
                simplex[-1] = xr
                f_vals[-1] = fr
                stall_count = 0
            elif fr < f_vals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                record(xe, fe)
                
                if n_calls >= budget or time.time() > deadline:
                    break
                
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                stall_count = 0
            else:
                # Contraction
                if fr < f_vals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                record(xc, fc)
                
                if n_calls >= budget or time.time() > deadline:
                    break
                
                if fc < min(fr, f_vals[-1]):
                    simplex[-1] = xc
                    f_vals[-1] = fc
                    stall_count = 0
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = func(simplex[i])
                        record(simplex[i], fi)
                        f_vals[i] = fi
                        
                        if n_calls >= budget or time.time() > deadline:
                            break
                    
                    if n_calls >= budget or time.time() > deadline:
                        break
                    
                    stall_count += 1
                    
                    # Restart if stalled
                    if stall_count >= max_stall:
                        # Create new simplex around current best
                        simplex = [x_best.copy()]
                        for i in range(dim):
                            xi = x_best.copy()
                            xi[i] += 0.5
                            simplex.append(np.clip(xi, -5.0, 5.0))
                        
                        f_vals = []
                        for x in simplex:
                            f = func(x)
                            record(x, f)
                            f_vals.append(f)
                            if n_calls >= budget or time.time() > deadline:
                                break
                        
                        if n_calls >= budget or time.time() > deadline:
                            break
                        
                        f_vals = np.array(f_vals)
                        stall_count = 0
                        max_stall = 10  # Increase restart frequency

        return x_best, f_best
# EVOLVE-BLOCK-END
