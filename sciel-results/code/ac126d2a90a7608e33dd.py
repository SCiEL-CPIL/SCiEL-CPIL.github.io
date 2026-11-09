"""Optimizer for BBOB f6 (Attractive Sector) combining (1+1)-ES with CMA-ES refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with CMA-ES refinement and global scan warm start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_refine(self, func, x0, sigma0, max_calls, deadline):
        """CMA-ES refinement from a given starting point. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        rng = np.random.default_rng(42)
        x = np.array(x0, dtype=float)
        fx = float(func(x))
        calls = 1
        sigma = float(sigma0)
        C = np.eye(n)
        Cinv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # CMA-ES parameters for small n
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4.0 + mueff / n) / ((n + 2.0) ** 2 + mueff / 2.0)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (2.0 - cc + cs)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) /
                 ((n + 2.0) ** 2 + mueff))
        chi = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        
        while calls < max_calls and time.time() < deadline:
            # Generate population
            batch_y = []
            batch_z = []
            batch_f = []
            
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x + sigma * z
                batch_y.append(y)
                batch_z.append(z)
                fy = float(func(y))
                batch_f.append(fy)
                calls += 1
                if calls >= max_calls or time.time() > deadline:
                    break
            
            if len(batch_y) < 2:
                break
            
            batch_y = np.array(batch_y)
            batch_z = np.array(batch_z)
            batch_f = np.array(batch_f)
            
            # Sort by fitness
            order = np.argsort(batch_f)
            batch_y = batch_y[order]
            batch_z = batch_z[order]
            batch_f = batch_f[order]
            
            # Update mean
            xmean = np.sum(w[:len(batch_y)][:, None] * batch_y, axis=0)
            
            # Update ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv @ (batch_z.mean(axis=0, keepdims=True) * np.sqrt(mueff)).T[0]
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls + 1)))) < (1.4 + 2.0 / (n + 1)) * chi
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc)) * (xmean - x) / (sigma + 1e-300)
            
            delta = (pc @ C @ pc) / (pc @ pc + 1e-300)
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) \
                + cmu * np.sum(w[:len(batch_y)][:, None, None] * np.outer(batch_y - x, batch_y - x), axis=0) / (sigma ** 2 + 1e-300)
            
            # Recompute inverse
            try:
                Cinv = np.linalg.inv(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                Cinv = np.eye(n)
            
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1.0))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            if sigma < 1e-12:
                break
                
            x = xmean
            fx = float(batch_f[0])
        
        return x, fx, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Global scan for early progress (AOCC rewards early improvement)
        n_scan = min(50, budget // 10)
        best_x = None
        best_f = np.inf
        
        for _ in range(n_scan):
            x = rng.uniform(-5.0, 5.0, dim)
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        # Phase 2: (1+1)-ES refinement
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_used = n_scan
        
        for _ in range(budget - calls_used - 100):  # Reserve 100 calls for CMA-ES if needed
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = float(func(y))
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # If sigma is very small, (1+1)-ES has likely converged
            if sigma < 1e-4:
                break
        
        # Phase 3: CMA-ES refinement if budget remains and (1+1)-ES converged
        remaining = budget - calls_used
        if remaining >= 50 and time.time() < deadline:
            try:
                x_cma, fx_cma, used = self._cmaes_refine(func, best_x, 0.5, remaining, deadline)
                calls_used += used
                if fx_cma < best_f:
                    best_f = fx_cma
                    best_x = x_cma.copy()
            except Exception:
                pass
        
        # Ensure we return a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
