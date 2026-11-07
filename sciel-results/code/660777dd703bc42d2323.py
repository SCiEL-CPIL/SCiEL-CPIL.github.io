"""Direct approach: combine global exploration with local CMA-ES for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with diverse random points
        # Use Latin hypercube sampling for better coverage
        n_lhs = min(50, budget // 10)
        # Add the origin and some structured points
        candidates = [np.zeros(dim)]
        
        # Latin hypercube sampling
        for _ in range(n_lhs):
            if calls_made >= budget or time_left() < 10:
                break
            # For each dimension, divide [-5,5] into n_lhs strata
            pts = []
            for d in range(dim):
                stratum = rng.integers(0, n_lhs)
                lower = -5.0 + (10.0 / n_lhs) * stratum
                upper = lower + 10.0 / n_lhs
                x_d = rng.uniform(lower, upper)
                pts.append(x_d)
            candidates.append(np.array(pts))
        
        # Evaluate initial candidates
        for x in candidates:
            if calls_made >= budget or time_left() < 5:
                break
            make_call(np.clip(x, -5.0, 5.0))
        
        if best_x is None:
            # Fallback: just return origin
            make_call(np.zeros(dim))
            return best_x, best_f

        # Phase 2: Local refinement around best point using CMA-ES
        # Use remaining budget for CMA-ES
        remaining_budget = budget - calls_made
        
        if remaining_budget < 20:
            return best_x, best_f

        # CMA-ES parameters for local search
        # Moderate population for local refinement
        lam = 20
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialize CMA-ES around best point
        m = best_x.copy()
        # Start with moderate step size
        sigma = 0.5
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        dcor = 1.0 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Eigen decomposition
        C_eigvals, C_eigvecs = np.linalg.eigh(C)
        
        def update_eig():
            nonlocal C, C_eigvals, C_eigvecs
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.maximum(C_eigvals, 1e-12)
        
        # CMA-ES loop
        while calls_made < budget and time_left() > 2:
            # Compute square root of C
            sqrt_C_eigvals = np.sqrt(C_eigvals)
            C_sqrt = C_eigvecs @ np.diag(sqrt_C_eigvals)
            
            # Generate population
            Z = rng.standard_normal((lam, dim))
            Y = m[None, :] + sigma * (C_sqrt @ Z.T).T
            Y = np.clip(Y, -5.0, 5.0)
            
            # Evaluate population
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls_made >= budget or time_left() < 1:
                    break
                f = make_call(Y[i])
                if f is not None:
                    F[i] = f
            
            if calls_made >= budget or time_left() < 1:
                break
            
            # Check if any evaluation was made
            if np.all(np.isinf(F)):
                break
                
            # Sort by fitness
            valid = ~np.isinf(F)
            if not np.any(valid):
                break
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update mean
            m_old = m.copy()
            # Only use valid evaluations
            valid_idx = [i for i in idx if not np.isinf(F[i])]
            if len(valid_idx) < mu:
                # Use all valid
                sel_Y = Y[valid_idx]
                sel_w = np.ones(len(valid_idx)) / len(valid_idx)
                m = np.sum(sel_w[:, None] * sel_Y, axis=0)
            else:
                Y_sorted_valid = Y_sorted[:mu]
                m = np.sum(w[:, None] * Y_sorted_valid, axis=0)
            
            # Update sigma, C, pc, ps
            if sigma > 1e-12:
                y = (m - m_old) / sigma
            else:
                y = np.zeros(dim)
            
            # Update ps
            hs = (math.sqrt(1 - math.exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs)
            if np.linalg.norm(y) < (1 + 5.0 / (dim + 1)) * hs * math.sqrt(dim) * sigma:
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * y
            
            # Update pc
            # Robust heuristic for h_sig condition
            hsig_threshold = 1.5 * math.sqrt(dim) * sigma
            if np.linalg.norm(ps) < hsig_threshold:
                pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mu_eff) * ps / sigma
            
            # Update C
            norm_ps_sq = np.dot(ps, ps)
            delta_h = (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs) * (norm_ps_sq / dim - 1)
            
            rank1 = np.outer(y, y)
            
            # Rank-mu update
            if len(valid_idx) >= mu:
                Y_selected = Y_sorted[:mu]
            else:
                Y_selected = Y[valid_idx]
                if len(Y_selected) < mu:
                    # Pad with mean
                    while len(Y_selected) < mu:
                        Y_selected = np.vstack([Y_selected, m_old])
            
            Y_steps = (Y_selected[:mu] - m_old[None, :]) / max(sigma, 1e-12)
            rank_mu = np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', Y_steps, Y_steps), axis=0)
            
            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
            C = (C + C.T) / 2.0
            update_eig()
            
            # Update sigma
            sigma = sigma * math.exp(cs / dcor * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            
            # Convergence check
            if sigma < 1e-10:
                break
            
            # Stagnation restart: if no improvement in last 30 evaluations, restart locally
            # For simplicity, we'll just continue but could add restart logic
            
        return best_x, best_f
# EVOLVE-BLOCK-END
