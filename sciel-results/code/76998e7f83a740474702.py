import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s limit
        d = self.dim
        budget = self.budget
        
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0
        
        def check_time():
            return (time.time() - start_time) > timeout
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: Multi-resolution grid scan (Halton) followed by local refinement
        # 1. Halton Sequence Scan
        # Halton sequence is a low-discrepancy sequence that covers the space uniformly.
        # This helps locate the "step" boundary in f7 more efficiently than random search.
        
        # Number of points in Halton scan
        # We allocate about 50% of the budget to the scan to ensure broad coverage
        # and leave the rest for local refinement.
        n_scan = max(50, budget // 2)
        
        # Generate Halton sequence points
        # We use a fixed number of prime bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Generate n_scan points
        scan_points = []
        for i in range(1, n_scan + 1):
            point = []
            for j in range(d):
                point.append(halton(i, primes[j]))
            # Map from [0, 1]^d to [-5, 5]^d
            x = np.array(point) * 10.0 - 5.0
            scan_points.append(x)
            
            # Check time and budget
            if check_time() or calls_made >= budget:
                break

        # Evaluate scan points
        for x in scan_points:
            if check_time() or calls_made >= budget:
                break
            evaluate(x)

        # 2. Local Refinement using Nelder-Mead
        # Use the best point found in the scan as the starting point for NM.
        # Also, consider the top few points from the scan to potentially start multiple NM runs
        # if budget allows, but for simplicity and robustness, we focus on the single best one
        # or a small set of promising ones.
        
        # Let's pick the best point found so far.
        x_start = best_x.copy()
        
        # Initialize Simplex around x_start
        # Use a smaller initial step size since we are likely near the optimum or at least in a good region
        delta = 0.1
        simplex = [x_start.copy()]
        for i in range(d):
            xi = x_start.copy()
            xi[i] += delta
            simplex.append(xi)
            
        # Evaluate initial simplex
        fvals = [evaluate(simplex[0])]
        for i in range(1, d+1):
            if check_time() or calls_made >= budget:
                return best_x, best_f
            fvals.append(evaluate(simplex[i]))
            
        # Nelder-Mead Loop
        max_nm_iter = 100 # Increased iterations for better convergence
        for _ in range(max_nm_iter):
            if check_time() or calls_made >= budget:
                break
                
            # Sort simplex by f value
            sort_idx = np.argsort(fvals)
            simplex = [simplex[i] for i in sort_idx]
            fvals = [fvals[i] for i in sort_idx]
            
            # Check convergence
            f_range = fvals[-1] - fvals[0]
            if f_range < 1e-8:
                break
            
            # Centroid of best N points
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            xr = centroid + 1.0 * (centroid - worst)
            if check_time() or calls_made >= budget:
                return best_x, best_f
            fr = evaluate(xr)
            
            if fr < fvals[0]:
                # Expansion
                if check_time() or calls_made >= budget:
                    return best_x, best_f
                xe = centroid + 2.0 * (xr - centroid)
                fe = evaluate(xe)
                if check_time() or calls_made >= budget:
                    return best_x, best_f
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
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    if check_time() or calls_made >= budget:
                        return best_x, best_f
                    xc = centroid + 0.5 * (xr - centroid)
                    fc = evaluate(xc)
                    if check_time() or calls_made >= budget:
                        return best_x, best_f
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d+1):
                            if check_time() or calls_made >= budget:
                                return best_x, best_f
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])
                else:
                    # Inside contraction
                    if check_time() or calls_made >= budget:
                        return best_x, best_f
                    xc = centroid - 0.5 * (worst - centroid)
                    fc = evaluate(xc)
                    if check_time() or calls_made >= budget:
                        return best_x, best_f
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d+1):
                            if check_time() or calls_made >= budget:
                                return best_x, best_f
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])

        return best_x, best_f
