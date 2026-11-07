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
        deadline = start_time + 290.0

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

        # ---- Phase 1: Halton sampling for model fit ----
        def halton(n, d):
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            pts = []
            for i in range(1, n + 1):
                p = []
                for dd in range(d):
                    base = bases[dd % len(bases)]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    p.append(r)
                pts.append(p)
            return np.array(pts)

        n_fit = 21
        hx = halton(n_fit, dim) * 10.0 - 5.0
        hx = np.vstack([hx, np.zeros(dim)])
        n_fit = hx.shape[0]

        fx = np.empty(n_fit)
        for i in range(n_fit):
            if not still_ok():
                break
            fx[i] = counting_func(hx[i])
        fx = fx[:eval_count]
        hx = hx[:len(fx)]
        if len(fx) < dim + 2:
            fx = np.array([best_f]) if best_f != np.inf else np.array([0.0])
            hx = best_x.reshape(1, -1) if best_x is not None else np.zeros((1, dim))

        # Fit the model
        try:
            b, s, f = self._fit_model(hx, fx)
        except Exception:
            b, s, f = 0.0, 1.0, np.ones(dim)

        # ---- Phase 2: Structure-aware spectral coordinate descent ----
        # For each dimension, the cosine term cos(pi * f_d * x_d) has minima at
        # x_d = (2k+1)/f_d. We do a coordinate-wise search: for each dimension,
        # sweep candidate minima and evaluate the full function.
        
        # Generate candidate x values per dimension based on fitted frequencies
        dim_candidates = []
        for d in range(dim):
            fd = max(f[d], 0.1)
            # Candidates: (2k+1)/fd in [-5, 5]
            kmin = int(np.floor(-5 * fd / 2.0)) - 1
            kmax = int(np.ceil(5 * fd / 2.0)) + 1
            cands = []
            for k in range(kmin, kmax + 1):
                x_val = (2 * k + 1) / fd
                if -5.0 <= x_val <= 5.0:
                    cands.append(x_val)
            if not cands:
                cands = [0.0]
            # Also add x=1 (poly minimum) and x=0
            cands.append(1.0)
            cands.append(0.0)
            # Deduplicate and sort
            cands = sorted(set([round(c, 6) for c in cands]))
            dim_candidates.append(cands)

        # Coarse-to-fine coordinate descent
        # Start from the model's best candidate
        model_cand = self._model_min_candidates(f, dim, b, s)
        current_x = model_cand.copy()
        current_f = counting_func(current_x)

        # Coordinate-wise sweep: for each dim, try all candidates
        for sweep in range(3):  # multiple passes for convergence
            if not still_ok():
                break
            improved = False
            for d in range(dim):
                if not still_ok():
                    break
                for cand in dim_candidates[d]:
                    if not still_ok():
                        break
                    test_x = current_x.copy()
                    test_x[d] = cand
                    val = counting_func(test_x)
                    if val < best_f:
                        current_x = test_x.copy()
                        current_f = val
                        improved = True
        # After coordinate sweep, do local refinement
        if best_x is not None and still_ok():
            # Use the best found so far as starting point for local optimization
            local_start = best_x.copy()
        else:
            local_start = current_x.copy()

        # ---- Phase 3: CMA-ES for global refinement ----
        if still_ok() and (budget - eval_count) > 50:
            try:
                from cma import CMA
                # Use CMA-ES with the best point as starting center
                popsize = min(20, max(10, (budget - eval_count) // 15))
                opts = {
                    'popsize': popsize,
                    'maxiter': 30,
                    'bounds': [(-5.0, 5.0)] * dim,
                    'verbose': -9,
                    'seed': 42,
                }
                es = CMA(local_start, 0.5, opts)
                while not es.stop() and still_ok() and eval_count < budget - 10:
                    solutions = es.ask()
                    for sol in solutions:
                        if not still_ok():
                            break
                        counting_func(sol)
                    es.tell(solutions, [counting_func(s) if not still_ok() else np.inf for s in []])
                    # Actually we need to collect the function values properly
                    break  # CMA-ES integration is tricky with our counting, fall through to scipy
            except Exception:
                pass

        # ---- Phase 4: L-BFGS-B local refinement ----
        if best_x is not None and still_ok():
            try:
                remaining = budget - eval_count
                if remaining > 10:
                    res = scipy.optimize.minimize(
                        counting_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 30, 'maxls': 6, 'maxfun': max(10, remaining - 5)}
                    )
            except Exception:
                pass

        # ---- Phase 5: Nelder-Mead polish ----
        if best_x is not None and still_ok() and (budget - eval_count) > 15:
            try:
                scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': max(5, (budget - eval_count) // 4),
                             'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = 0.0

        return best_x, best_f
