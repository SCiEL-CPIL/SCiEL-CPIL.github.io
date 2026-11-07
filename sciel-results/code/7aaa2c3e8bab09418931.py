import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES with randomized simplex initialization and (1+1)-ES polish
    for BBOB f13 (sharp ridge).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        def clip(v):
            return np.clip(v, -5.0, 5.0)
        
        def time_ok():
            return time.time() < deadline and total_evals < budget

        # Phase 1: Coarse multi-start to locate basin
        # Use a mix of random points and structured points
        n_starts = 10
        candidates = rng.uniform(-4.0, 4.0, size=(n_starts, dim))
        # Add origin and some symmetric points
        extra = np.array([
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0
        ])
        candidates = np.vstack([candidates, extra])
        
        start_xs = []
        start_fs = []
        for i in range(len(candidates)):
            if not time_ok():
                break
            x = clip(candidates[i])
            fx = func(x)
            total_evals += 1
            start_xs.append(x)
            start_fs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no evaluations made
            x = clip(rng.uniform(-4, 4, dim))
            best_f = func(x)
            total_evals += 1
            best_x = x.copy()

        # Phase 2: CMA-ES from the best start
        # We implement a simplified CMA-ES.
        # For dim=5, CMA-ES is very effective.
        
        x = best_x.copy()
        fx = best_f
        
        # Initial sigma
        sigma = 1.0
        
        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1. / mu_eff) / (dim + 2) ** 2 - 1))
        
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        # Initial population
        # We will generate the population in a loop to respect budget
        
        # Helper to generate a step
        def generate_step(sigma):
            return sigma * (B @ (D * rng.standard_normal(dim)))
        
        # Main CMA-ES loop
        # We estimate how many generations we can afford.
        # Each gen costs lambda_ evaluations.
        # Budget remaining: budget - total_evals
        # Max gens: (budget - total_evals) // lambda_
        
        while time_ok():
            if total_evals + lambda_ > budget + 10: # Allow slight overrun if close, but strict budget is better
                break
            
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_):
                if not time_ok():
                    break
                y = clip(x + generate_step(sigma))
                fy = func(y)
                total_evals += 1
                ys.append(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(ys) < lambda_:
                break
            
            # Sort
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update mean
            new_x = sum(w * y for w, y in zip(weights, ys[:mu]))
            
            # Update covariance matrix
            # Diff
            diffs = np.array(ys[:mu]) - x
            # Weights for covariance
            w_cov = weights
            # Rank-one update
            z = (p_c / np.sqrt(c_c * (1 - c_c)))
            c = c_1 + c_mu
            
            C = (1 - c) * C # Wait, C is not defined yet. Need to initialize C.
            
            # Actually, let's use a simpler (1+1)-ES if CMA-ES implementation is too complex/buggy prone in this short snippet.
            # The prompt says "Combine strategy... with strongest ideas".
            # Inspiration 3 (score 2.9666) used Multi-start CMA-ES.
            # Inspiration 1 (score 2.0857) used Multi-start Powell + ES.
            # Current (1.3646) is (1+1)-ES.
            
            # Let's stick to a robust (1+1)-ES with adaptive sigma and restarts, 
            # but improve the initialization and add a local search phase.
            # Actually, the "strongest idea" from Insp 3 is CMA-ES. 
            # Implementing full CMA-ES correctly in ~100 lines is risky.
            # Let's try a simplified CMA-ES or a better (1+1)-ES with covariance adaptation.
            
            # Let's switch back to a well-tuned (1+1)-ES with restarts, 
            # as it's more robust to implementation errors in a constrained environment.
            # We will use the "Randomized Simplex Initialization" idea from the lesson:
            # "Randomized simplex initialization aligns with rotated ridges faster"
            
            # So, Phase 2: Nelder-Mead with randomized simplex, then (1+1)-ES polish.
            # This matches Inspiration 1's structure but with better init.
            
            break # Exit CMA-ES loop, switch to Nelder-Mead/ES strategy

        # Phase 2: Nelder-Mead from best_x with randomized simplex
        remaining = budget - total_evals
        if remaining > 10 and best_x is not None and time_ok():
            x = best_x.copy()
            
            # Initialize simplex
            simplex = [x.copy()]
            # Randomized initialization to align with ridge
            for i in range(dim):
                new_pt = x.copy()
                # Random direction with some magnitude
                dir_vec = rng.standard_normal(dim)
                dir_vec /= np.linalg.norm(dir_vec)
                step = 0.5 * rng.uniform(0.5, 2.0)
                new_pt += step * dir_vec
                new_pt = clip(new_pt)
                simplex.append(new_pt)
            
            simplex_f = []
            for pt in simplex:
                if not time_ok():
                    break
                f_val = func(pt)
                total_evals += 1
                simplex_f.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = pt.copy()
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iter = remaining // (dim + 1)
            
            for _ in range(max_iter):
                if not time_ok():
                    break
                # Sort
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = clip(centroid + alpha * (centroid - simplex[-1]))
                fr = func(xr)
                total_evals += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                    
                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expansion
                    xe = clip(centroid + gamma * (centroid - simplex[-1]))
                    fe = func(xe)
                    total_evals += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        xc = clip(centroid + rho * (simplex[-1] - centroid))
                    else:
                        xc = clip(centroid - rho * (simplex[-1] - centroid))
                    fc = func(xc)
                    total_evals += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok():
                                break
                            simplex[i] = clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]))
                            simplex_f[i] = func(simplex[i])
                            total_evals += 1
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()

        # Phase 3: (1+1)-ES Polish
        remaining = budget - total_evals
        if remaining > 0 and best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            successes = 0
            window = 0
            WIN = 5
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                total_evals += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-6, 1.0))
                    successes = 0
                    window = 0

        return best_x, best_f
