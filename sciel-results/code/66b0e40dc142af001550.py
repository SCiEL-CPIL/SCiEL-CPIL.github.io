import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _katsuura_model(self, x, b, s, f):
        x = np.asarray(x, dtype=float)
        cos_terms = np.prod(np.cos(np.pi * f * x), axis=1)
        poly = 0.1 * np.sum(np.abs(x - 1.0) * (1.0 + 10.0 * np.abs(x - 1.0)) ** 2, axis=1)
        return b + s * (cos_terms + poly)

    def _fit_model(self, samples_x, samples_f):
        b = float(np.mean(samples_f))
        resid = samples_f - b
        s = float(np.sqrt(np.mean(resid ** 2)))
        if s < 1e-12:
            s = 1.0

        dim = samples_x.shape[1]
        f = np.ones(dim)
        for _ in range(3):
            poly = 0.1 * np.sum(np.abs(samples_x - 1.0) * (1.0 + 10.0 * np.abs(samples_x - 1.0)) ** 2, axis=1)
            target = resid / s - poly

            def total_cost(fv):
                cos = np.ones(len(samples_f))
                for d in range(dim):
                    cos = cos * np.cos(np.pi * fv[d] * samples_x[:, d])
                return float(np.sum((cos - target) ** 2))

            for d in range(dim):
                best_fv = f.copy()
                best_c = total_cost(f)
                for cand in np.linspace(0.3, 3.0, 14):
                    f[d] = cand
                    c = total_cost(f)
                    if c < best_c:
                        best_c = c
                        best_fv = f.copy()
                f = best_fv
        return b, s, f

    def _model_min_candidates(self, f, dim, b, s):
        cands = []
        for d in range(dim):
            fd = max(f[d], 1e-3)
            kmin = int(np.floor(-5 * fd / 2.0)) - 1
            kmax = int(np.ceil(5 * fd / 2.0)) + 1
            xs = [(2 * k + 1) / fd for k in range(kmin, kmax + 1)]
            xs = [x for x in xs if -5.0 <= x <= 5.0]
            if not xs:
                xs = [1.0 / fd if fd > 0 else 0.0]
            best = min(xs, key=lambda x: abs(x - 1.0))
            cands.append(best)
        return np.array(cands)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_c = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                val = float(func(x_c))
            except Exception:
                return np.inf
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_c.copy()
            return val

        def still_ok():
            return eval_count < budget and time.time() < deadline

        # Phase 1: Structural exploration - Katsuura has minima near x_d = 1/f_d
        # where f_d is the frequency. Without knowing f_d, explore a grid of
        # plausible minima locations.
        
        # Generate candidate points based on the known structure:
        # For each dimension, minima occur at x = (2k+1)/f for integer k
        # We don't know f, but we can try a range of plausible values
        
        # First, do a coarse random sampling to get a sense of the landscape
        rng = np.random.RandomState(42)
        n_coarse = 30
        coarse_pts = rng.uniform(-5, 5, size=(n_coarse, dim))
        # Add some structural points: try x_d = 1.0 (poly minimum)
        for d in range(dim):
            pt = rng.uniform(-5, 5, size=dim)
            pt[d] = 1.0
            coarse_pts = np.vstack([coarse_pts, pt])
        
        for pt in coarse_pts:
            if not still_ok():
                break
            counting_func(pt)
        
        if best_x is None:
            best_x = np.ones(dim)
            best_f = counting_func(best_x)
        
        # Phase 2: Multi-start local optimization from diverse starting points
        # Use the best point and several perturbed versions
        
        # Generate diverse starting points around the best found
        n_starts = 8
        starts = [best_x.copy()]
        
        # Add perturbations of different magnitudes
        for scale in [0.5, 1.0, 2.0]:
            for i in range(2):
                if not still_ok():
                    break
                pert = best_x + scale * rng.standard_normal(dim)
                pert = np.clip(pert, -5, 5)
                starts.append(pert)
        
        # Also try points based on the structure: x_d = 1.0 for some dimensions
        for d in range(dim):
            if not still_ok():
                break
            pt = best_x.copy()
            pt[d] = 1.0
            starts.append(pt)
        
        # Local refinement from each starting point
        for start in starts:
            if not still_ok():
                break
            remaining = budget - eval_count
            if remaining < 20:
                break
            
            # Try Nelder-Mead with tight tolerances
            try:
                scipy.optimize.minimize(
                    counting_func,
                    start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(100, max(20, remaining // 3)),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
            
            # Follow up with L-BFGS-B if we have budget
            if still_ok() and (budget - eval_count) > 15:
                try:
                    scipy.optimize.minimize(
                        counting_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 30,
                            'maxls': 10,
                            'maxfun': min(50, max(10, (budget - eval_count) // 2))
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Final polish with very tight tolerances
        if best_x is not None and still_ok() and (budget - eval_count) > 10:
            try:
                scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.ones(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = 0.0

        return best_x, best_f
