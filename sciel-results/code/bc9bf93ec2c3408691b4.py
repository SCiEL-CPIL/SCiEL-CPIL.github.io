import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Broad exploration with stratified sampling
        # For step ellipsoidal, the function is flat (constant offset) far from
        # the optimum and drops sharply near it. We need to locate the "cliff".
        
        # Start with center and a few structured points
        try_func(np.zeros(dim))
        
        # Use a quasi-random (Halton-like) sequence for better coverage
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # Generate diverse points using Halton sequence mapped to [-4, 4]^dim
        for i in range(30):
            if evals_used >= budget or time.time() > deadline:
                break
            x = np.array([halton(i + 2, base) * 8.0 - 4.0 for base in range(2, 2 + dim)])
            try_func(x)

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Adaptive coordinate descent with binary search on step size
        # For each coordinate, try to find the "cliff" by binary searching
        # between points with different function values.
        
        x_curr = best_x.copy()
        f_curr = best_f
        
        # For each coordinate, do a local search
        for coord in range(dim):
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Try a range of step sizes for this coordinate
            for step in [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]:
                if evals_used >= budget or time.time() > deadline:
                    break
                
                improved = False
                for sign in [1.0, -1.0]:
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    x_new = x_curr.copy()
                    x_new[coord] += sign * step
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = try_func(x_new)
                    
                    if f_new < f_curr - 1e-10:
                        x_curr = x_new.copy()
                        f_curr = f_new
                        improved = True
                        break
                
                if improved:
                    continue  # Try smaller steps on same coordinate
                else:
                    break  # No improvement at this step size, move to next coord

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: CMA-ES for final polish with adaptive sigma
        remaining = budget - evals_used
        if remaining >= 15 and time.time() < deadline:
            # CMA-ES parameters
            lam = max(4, min(8, remaining // 8))
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            w_full = np.concatenate([w, np.zeros(lam - mu)])
            mueff = 1.0 / np.sum(w**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            x_mean = best_x.copy()
            # Start with smaller sigma since we're likely close to optimum
            sigma = 0.1

            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            def cholesky_safe(C_mat):
                try:
                    L = np.linalg.cholesky(C_mat + 1e-12 * np.eye(dim))
                    return L
                except:
                    for reg in [1e-10, 1e-8, 1e-6]:
                        try:
                            L = np.linalg.cholesky(C_mat + reg * np.eye(dim))
                            return L
                        except:
                            pass
                    return np.eye(dim)

            L = cholesky_safe(C)

            while evals_used < budget and time.time() < deadline:
                if budget - evals_used < lam:
                    break

                ys = []
                xs = []
                fs = []

                for i in range(lam):
                    if evals_used >= budget or time.time() > deadline:
                        break

                    z = rng.standard_normal(dim)
                    y = L @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)

                    f_val = try_func(x_new)
                    if np.isinf(f_val):
                        break

                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_val)

                if len(xs) < lam:
                    break

                ys = np.array(ys)
                xs = np.array(xs)
                fs = np.array(fs)

                idx = np.argsort(fs)
                xs = xs[idx]
                ys = ys[idx]
                fs = fs[idx]

                x_mean_old = x_mean.copy()
                x_mean = np.sum(w_full[:mu, None] * xs[:mu], axis=0)

                y_bar = (x_mean - x_mean_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_bar
                norm_ps = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * y_bar
                sum_w_y = np.zeros((dim, dim))
                for i in range(mu):
                    sum_w_y += w[i] * np.outer(ys[i], ys[i])

                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)
                L = cholesky_safe(C)

        return best_x, best_f
