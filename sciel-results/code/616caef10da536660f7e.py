import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 320.0  # 10s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        # Phase 1: Enhanced Scout Phase
        scout_budget = 30
        best_x = None
        best_f = np.inf
        scout_points = []
        
        points = []
        points.append(np.zeros(d))
        for _ in range(5):
            points.append(rng.uniform(lo, hi, d))
        for _ in range(scout_budget - 6):
            points.append(rng.uniform(lo, hi, d))
            
        for x in points:
            if time.time() - self.start_time > self.deadline:
                break
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        # Phase 2: CMA-ES with coordinate-wise refinement for step functions
        
        lam = 4 + int(3 * np.log(d))
        if lam < 8:
            lam = 8
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1.0/(4*d) + 1.0/(21*d*d))
        
        def coordinate_refine(x, func, budget_left, lo, hi):
            """Coordinate-wise geometric step refinement for step functions."""
            if budget_left < 2 * d * 4:
                return x.copy(), budget_left
            
            x = x.copy()
            fx = func(x)
            step = 1.0
            min_step = 1e-4
            
            for coord in range(d):
                if time.time() - self.start_time > self.deadline:
                    break
                if budget_left < 4:
                    break
                    
                for direction in [1.0, -1.0]:
                    trial = x.copy()
                    trial[coord] += direction * step
                    trial = np.clip(trial, lo, hi)
                    f_trial = func(trial)
                    budget_left -= 1
                    
                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                    else:
                        break
            
            # Shrink step and repeat
            step *= 0.5
            while step > min_step and budget_left > d * 2:
                if time.time() - self.start_time > self.deadline:
                    break
                for coord in range(d):
                    if budget_left < 2:
                        break
                    for direction in [1.0, -1.0]:
                        if budget_left < 1:
                            break
                        trial = x.copy()
                        trial[coord] += direction * step
                        trial = np.clip(trial, lo, hi)
                        f_trial = func(trial)
                        budget_left -= 1
                        
                        if f_trial < fx:
                            x = trial
                            fx = f_trial
                        else:
                            break
                step *= 0.5
            
            return x, budget_left
        
        def init_cma(start_x, start_sigma):
            """Initialize CMA-ES state."""
            m = start_x.copy()
            sigma = start_sigma
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            return m, sigma, C, pc, ps
        
        def cma_step(m, sigma, C, pc, ps, rng, func, lo, hi, lam, mu, weights, 
                      mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n):
            """Perform one CMA-ES step, return updated state and best candidate."""
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)
                
            Z = rng.standard_normal((lam, d))
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)
            
            F = np.array([func(x) for x in X])
            
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]
            
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            step = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * chi_n else 0
            
            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)
                
            Y_delta = X_sorted[:mu] - m
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term
            
            C = (C + C.T) / 2.0
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)
                
            return m_new, sigma, C, pc, ps, F_sorted[0], X_sorted[0]
        
        # Initialize from best scout point
        if len(scout_points) > 2:
            scout_points.sort(key=lambda t: t[1])
            top_half = np.array([t[0] for t in scout_points[:max(1, len(scout_points)//2)]])
            sigma_init = np.linalg.norm(np.std(top_half, axis=0), axis=0) * 0.5
            if sigma_init < 0.1:
                sigma_init = 0.1
            elif sigma_init > 2.0:
                sigma_init = 2.0
        else:
            sigma_init = 1.0

        m, sigma, C, pc, ps = init_cma(best_x, sigma_init)
        
        remaining_budget = budget - scout_budget
        best_f_cma = best_f
        best_x_cma = best_x.copy()
        
        stagnation_count = 0
        stagnation_threshold = 6
        
        while remaining_budget >= lam:
            if time.time() - self.start_time > self.deadline:
                break
            
            m, sigma, C, pc, ps, f_best_candidate, x_best_candidate = cma_step(
                m, sigma, C, pc, ps, rng, func, lo, hi, lam, mu, weights,
                mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n
            )
            
            if f_best_candidate < best_f_cma:
                best_f_cma = f_best_candidate
                best_x_cma = x_best_candidate.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            remaining_budget -= lam
            
            if stagnation_count >= stagnation_threshold:
                best_restart_x = None
                best_restart_f = np.inf
                for _ in range(5):
                    if time.time() - self.start_time > self.deadline:
                        break
                    new_x = rng.uniform(lo, hi, d)
                    new_f = func(new_x)
                    remaining_budget -= 1
                    if new_f < best_restart_f:
                        best_restart_f = new_f
                        best_restart_x = new_x.copy()
                
                if best_restart_x is not None:
                    if best_restart_f < best_f_cma:
                        best_f_cma = best_restart_f
                        best_x_cma = best_restart_x.copy()
                    m, sigma, C, pc, ps = init_cma(best_restart_x, 1.0)
                    stagnation_count = 0
        
        # Phase 3: Final coordinate-wise polish
        if remaining_budget > 0 and time.time() - self.start_time < self.deadline:
            best_x_cma, remaining_budget = coordinate_refine(
                best_x_cma, func, remaining_budget, lo, hi
            )
            f_final = func(best_x_cma)
            if f_final < best_f_cma:
                best_f_cma = f_final
        
        return best_x_cma, best_f_cma
