"""Rewritten optimizer for BBOB f13 (Sharp Ridge) with Nelder-Mead + CMA-ES hybrid."""
import numpy as np
import time
import random


class Optimizer:
    """
    Hybrid optimizer for BBOB f13 (Sharp Ridge):
    1. Halton low-discrepancy initialization for broad early coverage (AOCC).
    2. Nelder-Mead simplex refinement on top candidates for fast local convergence.
    3. CMA-ES (if available) or adaptive (1+1)-ES for global refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim, skip=1):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for j in range(1, dim + 1):
                base = j + 1
                f = 1.0
                result = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    result += f * (i_val % base)
                    i_val //= base
                point.append(result)
            points.append(point)
        return np.array(points)

    def _nelder_mead(self, func, x0, budget, deadline, best_tracker):
        """
        Nelder-Mead simplex method.
        Returns best x, best f found.
        """
        dim = self.dim
        rng = np.random.default_rng(123)
        
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += 0.1 if x0[i] < 4.9 else -0.1
            simplex.append(x_new)
        
        # Evaluate initial simplex
        fvals = [func(x) for x in simplex]
        for x, f in zip(simplex, fvals):
            if f < best_tracker['best_f']:
                best_tracker['best_f'] = f
                best_tracker['best_x'] = x.copy()
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        for _ in range(budget):
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            
            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
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
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
            
            # Update best
            for x, f in zip(simplex, fvals):
                if f < best_tracker['best_f']:
                    best_tracker['best_f'] = f
                    best_tracker['best_x'] = x.copy()
        
        return best_tracker['best_x'].copy(), best_tracker['best_f']

    def _cma_es(self, func, x0, sigma0, budget, deadline, best_tracker):
        """
        Simple CMA-ES implementation.
        """
        dim = self.dim
        rng = np.random.default_rng(456)
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff + c_sigma)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # State
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Evaluate initial population
        pops = [x_mean + sigma * rng.multivariate_normal(np.zeros(dim), C) for _ in range(lambda_)]
        pops = [np.clip(p, -5, 5) for p in pops]
        fvals = [func(p) for p in pops]
        
        for p, f in zip(pops, fvals):
            if f < best_tracker['best_f']:
                best_tracker['best_f'] = f
                best_tracker['best_x'] = p.copy()
        
        for it in range(budget // lambda_):
            if time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            pops = [pops[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * pops[:mu], axis=0)
            
            # Update step size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (x_mean - x_old))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * np.outer(pc, pc)
            for i in range(mu):
                z = pops[i] - x_old
                C -= c_sigma * w[i] * np.outer(z, z)
            
            # Regularize
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Generate new population
            pops = [x_mean + sigma * rng.multivariate_normal(np.zeros(dim), C) for _ in range(lambda_)]
            pops = [np.clip(p, -5, 5) for p in pops]
            fvals = [func(p) for p in pops]
            
            for p, f in zip(pops, fvals):
                if f < best_tracker['best_f']:
                    best_tracker['best_f'] = f
                    best_tracker['best_x'] = p.copy()
        
        return best_tracker['best_x'].copy(), best_tracker['best_f']

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Halton initialization
        n_init = min(30, max(10, budget // 8))
        halton_points = self._halton_sequence(n_init, dim, skip=1)
        init_points = halton_points * 10.0 - 5.0

        best_tracker = {'best_x': None, 'best_f': np.inf}
        all_points = []
        all_fvals = []

        for p in init_points:
            if time.time() > deadline:
                break
            fp = func(p)
            all_points.append(p)
            all_fvals.append(fp)
            if fp < best_tracker['best_f']:
                best_tracker['best_f'] = fp
                best_tracker['best_x'] = p.copy()

        if not all_points:
            return best_tracker['best_x'], best_tracker['best_f']

        # Phase 2: Nelder-Mead on top candidates
        remaining = budget - len(all_points)
        sorted_idx = np.argsort(all_fvals)
        top_k = min(3, len(sorted_idx))
        
        nm_budget = max(1, remaining // 2)
        
        for rank in range(top_k):
            if time.time() > deadline:
                break
            idx = sorted_idx[rank]
            x0 = all_points[idx].copy()
            self._nelder_mead(func, x0, nm_budget, deadline, best_tracker)

        # Phase 3: CMA-ES or (1+1)-ES on best point
        remaining = budget - len(all_points) - top_k * nm_budget
        if remaining > 5 and best_tracker['best_x'] is not None:
            x0 = best_tracker['best_x'].copy()
            self._cma_es(func, x0, 0.5, remaining, deadline, best_tracker)

        return best_tracker['best_x'], best_tracker['best_f']
