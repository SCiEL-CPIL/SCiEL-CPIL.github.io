"""Model-and-optimize for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-and-optimize with Gaussian Process Surrogate + DE-based local search:
    1) Halton-based DOE for better space-filling coverage.
    2) Fit a GP surrogate per objective.
    3) Propose points by minimizing weighted sum of GP means.
    4) Differential Evolution local search on real problem.
    5) Interpolation and remaining budget for DE restarts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- GP Surrogate Implementation ---
        def matern_kernel(X, Y, length_scale=1.0, variance=1.0, nu=2.5):
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            if nu == 2.5:
                kernel = np.sqrt(3) * dist / length_scale
                return variance * (1 + kernel) * np.exp(-kernel)
            elif nu == 5.0:
                kernel = np.sqrt(5) * dist / length_scale
                return variance * (1 + kernel + kernel**2/3) * np.exp(-kernel)
            else:
                return variance * np.exp(-0.5 * dist**2 / length_scale**2)

        class GPSurrogate:
            def __init__(self, X, Y, length_scale=0.5):
                self.X = X
                self.Y = Y
                self.mu = np.mean(Y, axis=0)
                self.sigma = np.std(Y, axis=0)
                self.Y_scaled = (Y - self.mu) / (self.sigma + 1e-10)
                self.length_scale = length_scale
                self.variance = 1.0
                self.K = matern_kernel(X, X, length_scale, self.variance) + 1e-6 * np.eye(X.shape[0])
                try:
                    self.alpha = np.linalg.solve(self.K, self.Y_scaled)
                except np.linalg.LinAlgError:
                    self.K += 1e-4 * np.eye(X.shape[0])
                    self.alpha = np.linalg.solve(self.K, self.Y_scaled)

            def predict(self, X_new):
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance)
                means = self.mu + K_s @ self.alpha * self.sigma
                v = matern_kernel(X_new, X_new, self.length_scale, self.variance)
                v = np.diag(v) - np.sum(K_s @ self.K @ K_s.T, axis=1)
                v = np.maximum(v, 1e-10)
                stds = np.sqrt(v) * self.sigma
                return means, stds

        # --- 1) DOE: Halton sequence + special points ---
        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            def halton_one(i, base):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, ...
                for i in range(1, n + 1):
                    points[i-1, d] = halton_one(i, base)
            return points

        n_halton = 40
        halton_pts = halton_sequence(n_halton, dim)
        
        specials = [lo, hi, (lo+hi)/2.0,
                    np.array([0,0,1,1]), np.array([0,1,0,1]),
                    np.array([0,1,1,0]), np.array([1,0,0,1]),
                    np.array([1,0,1,0]), np.array([1,1,0,0])]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)
        
        grid = np.vstack([specials, halton_pts])
        rng.shuffle(grid)

        model_X = []
        model_Y = []
        pareto = {}

        def add_point(x, f):
            key = tuple(np.round(np.asarray(f, dtype=float), 12))
            if key not in pareto:
                pareto[key] = np.asarray(x, dtype=float).copy()

        for x in grid:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                model_X.append(np.asarray(x, dtype=float))
                model_Y.append(np.asarray(f, dtype=float))
                add_point(x, f)
            except Exception:
                break

        if len(model_X) == 0:
            return None

        Xtr = np.vstack(model_X)
        Ytr = np.vstack(model_Y)

        # Fit GPs
        gps = []
        for k in range(n_obj):
            try:
                gps.append(GPSurrogate(Xtr, Ytr[:, k], length_scale=0.5))
            except Exception:
                gps.append(None)

        def surrogate_predict(X_new):
            if all(g is None for g in gps):
                return np.tile(np.mean(Ytr, axis=0), (X_new.shape[0], 1))
            
            means = np.zeros((X_new.shape[0], n_obj))
            for k, g in enumerate(gps):
                if g is not None:
                    m, _ = g.predict(X_new)
                    means[:, k] = m
                else:
                    means[:, k] = np.mean(Ytr[:, k])
            return means

        # --- 2) Weighted Sum Surrogate Search ---
        weights = []
        for k in range(n_obj):
            w = np.zeros(n_obj); w[k] = 1.0; weights.append(w)
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                w = np.zeros(n_obj); w[i] = 0.5; w[j] = 0.5; weights.append(w)
        for _ in range(10):
            w = rng.dirichlet([1.0, 1.0, 1.0]); weights.append(w)

        proposals = []
        cand = rng.uniform(lo, hi, size=(500, dim))
        Yc = surrogate_predict(cand)
        
        for w in weights:
            s = Yc @ w
            idx = np.argsort(s)[:10]
            for i in idx:
                proposals.append(cand[i])

        for w in weights:
            s = Ytr @ w
            idx = np.argsort(s)[:5]
            for i in idx:
                proposals.append(Xtr[i])

        seen = set()
        unique_prop = []
        for x in proposals:
            k = tuple(np.round(np.clip(x, lo, hi), 6))
            if k not in seen:
                seen.add(k)
                unique_prop.append(np.clip(x, lo, hi))
        rng.shuffle(unique_prop)

        # --- 3) DE-based Local Optimization ---
        def local_de(x, pop_size=8, max_gen=15):
            """Differential Evolution for multi-objective (scalarized via random weights)."""
            x0 = np.asarray(x, dtype=float).copy()
            pop = np.clip(x0 + rng.normal(0, 0.1, size=(pop_size, dim)), lo, hi)
            pop_f = np.array([eval_real(p) for p in pop])
            best_idx = np.argmin(pop_f.sum(axis=1))
            best = pop[best_idx].copy()
            bestf = pop_f[best_idx].copy()
            
            for gen in range(max_gen):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                for i in range(pop_size):
                    if self.eval_count >= self.budget or time.time() > deadline:
                        break
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    trial = np.clip(pop[a] + 0.8 * (pop[b] - pop[c]), lo, hi)
                    trial_f = eval_real(trial)
                    if np.any(trial_f < pop_f[i]) and not np.all(trial_f > pop_f[i]):
                        pop[i] = trial
                        pop_f[i] = trial_f
                        if np.all(trial_f <= bestf) and np.any(trial_f < bestf):
                            best = trial.copy()
                            bestf = trial_f.copy()
            return best, bestf

        for x in unique_prop:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            b, bf = local_de(x, pop_size=6, max_gen=12)
            add_point(b, bf)

        # --- 4) Interpolation ---
        def pareto_points():
            pts = []
            for f, x in pareto.items():
                pts.append((x, np.asarray(f, dtype=float)))
            return pts

        used = set()
        for _ in range(30):
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            pts = pareto_points()
            if len(pts) < 2:
                break
            i = rng.integers(0, len(pts))
            j = rng.integers(0, len(pts))
            if i == j:
                continue
            xi, fi = pts[i]; xj, fj = pts[j]
            t = rng.uniform(0.15, 0.85)
            cand = np.clip((1.0-t)*xi + t*xj, lo, hi)
            key = tuple(np.round(cand, 6))
            if key in used:
                continue
            used.add(key)
            f = eval_real(cand)
            add_point(cand, f)

        # --- 5) Remaining Budget ---
        while self.eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            try:
                b, bf = local_de(x, pop_size=5, max_gen=8)
            except Exception:
                break
            add_point(b, bf)

        return None
# EVOLVE-BLOCK-END
