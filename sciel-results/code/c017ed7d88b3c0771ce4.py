"""Model-and-optimize for RE37 multi-objective task.
Combines GP surrogate with weighted-sum decomposition and local exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """GP-based model-and-optimize with weighted-sum decomposition:
    1) Space-filling DOE for initialization.
    2) Fit GP surrogates per objective.
    3) Propose points via weighted-sum minimization on GP means.
    4) Local exploitation with Nelder-Mead on promising points.
    5) Track Pareto front and use remaining budget for random restarts.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0  # safety margin before 330s kill
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        rng = np.random.default_rng(42)

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

        # --- GP Surrogate ---
        def matern_kernel(X, Y, length_scale=1.0, variance=1.0, nu=2.5):
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            if nu == 2.5:
                kernel = np.sqrt(3) * dist / length_scale
                return variance * (1 + kernel) * np.exp(-kernel)
            else:
                return variance * np.exp(-0.5 * (dist / length_scale)**2)

        class GPSurrogate:
            def __init__(self, X, Y, length_scale=0.5):
                self.X = X.copy()
                self.Y = Y.copy()
                self.mu = np.mean(Y)
                self.sigma = np.std(Y) + 1e-10
                self.Y_scaled = (Y - self.mu) / self.sigma
                self.length_scale = length_scale
                self.variance = 1.0
                self.K = matern_kernel(X, X, length_scale, self.variance) + 1e-6 * np.eye(X.shape[0])
                try:
                    self.L = np.linalg.cholesky(self.K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_scaled))
                except np.linalg.LinAlgError:
                    self.K += 1e-4 * np.eye(X.shape[0])
                    self.L = np.linalg.cholesky(self.K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_scaled))

            def predict(self, X_new):
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance)
                v = np.linalg.solve(self.L, K_s.T)
                means = self.mu + v.T @ self.alpha * self.sigma
                var = self.variance - np.sum(v**2, axis=1)
                var = np.maximum(var, 1e-10)
                stds = np.sqrt(var) * self.sigma
                return means, stds

        # --- 1) DOE: structured grid + special points ---
        levels = np.linspace(0.0, 1.0, 4)
        grid = np.array(np.meshgrid(*([levels]*dim)), dtype=float)
        grid = grid.reshape(dim, -1).T
        grid = grid[::2]  # reduce density
        rng.shuffle(grid)

        specials = [lo, hi, (lo+hi)/2.0,
                    np.array([0,0,1,1]), np.array([0,1,0,1]),
                    np.array([0,1,1,0]), np.array([1,0,0,1]),
                    np.array([1,0,1,0]), np.array([1,1,0,0])]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)
        doe = np.vstack([specials, grid])

        model_X = []
        model_Y = []

        for x in doe:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                model_X.append(np.asarray(x, dtype=float))
                model_Y.append(np.asarray(f, dtype=float))
            except Exception:
                break

        if len(model_X) == 0:
            return None

        Xtr = np.vstack(model_X)
        Ytr = np.vstack(model_Y)

        def fit_gps():
            gps = []
            for k in range(n_obj):
                try:
                    gps.append(GPSurrogate(Xtr, Ytr[:, k], length_scale=0.5))
                except Exception:
                    gps.append(None)
            return gps

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

        # --- 2) Weighted-sum decomposition search ---
        # Generate diverse weight vectors
        weights = []
        # Axis-aligned weights
        for k in range(n_obj):
            w = np.zeros(n_obj); w[k] = 1.0
            weights.append(w)
        # Pairwise equal weights
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                w = np.zeros(n_obj); w[i] = 0.5; w[j] = 0.5
                weights.append(w)
        # Random Dirichlet weights
        for _ in range(15):
            w = rng.dirichlet([1.0]*n_obj)
            weights.append(w)
        # Normalize
        weights = np.array(weights)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        def get_pareto_front(Y):
            """Extract non-dominated points."""
            if len(Y) == 0:
                return np.array([]).reshape(0, n_obj)
            Y = np.asarray(Y)
            n = len(Y)
            nondom = []
            for i in range(n):
                is_nondom = True
                for j in range(n):
                    if i != j and np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                        is_nondom = False
                        break
                if is_nondom:
                    nondom.append(Y[i])
            if len(nondom) == 0:
                return np.array([]).reshape(0, n_obj)
            return np.array(nondom)

        def nelder_mead_local(x0, max_evals=8):
            """Nelder-Mead local search on scalarized objective."""
            if self.eval_count >= self.budget or time.time() > deadline:
                return None, None
            
            best_x = np.asarray(x0, dtype=float).copy()
            best_f = eval_real(best_x)
            
            # Initial simplex
            simplex = [best_x.copy()]
            for d in range(dim):
                xc = best_x.copy()
                step = 0.05
                xc[d] += step
                xc = np.clip(xc, lo, hi)
                simplex.append(xc)
            simplex = np.array(simplex)
            
            simplex_f = [best_f]
            for xc in simplex[1:]:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                try:
                    fc = eval_real(xc)
                except Exception:
                    fc = best_f + 1.0
                simplex_f.append(fc)
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            
            for _ in range(max_evals):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                
                # Sort by scalarized value (sum of objectives)
                idx = np.argsort([np.sum(f) for f in simplex_f])
                simplex = simplex[idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, lo, hi)
                try:
                    f_ref = eval_real(x_ref)
                except Exception:
                    break
                
                if np.sum(f_ref) < np.sum(simplex_f[0]):
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, lo, hi)
                    try:
                        f_exp = eval_real(x_exp)
                    except Exception:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                        continue
                    if np.sum(f_exp) < np.sum(f_ref):
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif np.sum(f_ref) < np.sum(simplex_f[-1]):
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    try:
                        f_con = eval_real(x_con)
                    except Exception:
                        break
                    if np.sum(f_con) < np.sum(simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            try:
                                simplex_f[i] = eval_real(simplex[i])
                            except Exception:
                                simplex_f[i] = simplex_f[0] + 1.0
                        break
                
                # Update best
                best_idx = np.argmin([np.sum(f) for f in simplex_f])
                if np.sum(simplex_f[best_idx]) < np.sum(best_f):
                    best_x = simplex[best_idx].copy()
                    best_f = simplex_f[best_idx].copy()
            
            return best_x, best_f

        # Main loop: fit GP, propose via weighted sums, exploit locally
        gps = fit_gps()
        
        # Generate candidate pool
        n_cand = 300
        cand_pool = rng.uniform(lo, hi, size=(n_cand, dim))
        
        # Add structured candidates
        structured = []
        for i in range(dim):
            for sign in [-1, 1]:
                xc = (lo + hi) / 2.0
                xc[i] = lo[i] if sign < 0 else hi[i]
                structured.append(xc)
        cand_pool = np.vstack([cand_pool, np.array(structured)])
        
        n_iter = 0
        while self.eval_count < self.budget and time.time() < deadline:
            n_iter += 1
            
            # Refit GP periodically
            if n_iter % 5 == 1:
                gps = fit_gps()
            
            # Predict on candidate pool
            Yc = surrogate_predict(cand_pool)
            
            # For each weight vector, find best candidate
            proposals = []
            for w in weights:
                s = Yc @ w
                idx = np.argmin(s)
                proposals.append(cand_pool[idx])
            
            # Deduplicate proposals
            seen = set()
            unique_prop = []
            for x in proposals:
                k = tuple(np.round(np.clip(x, lo, hi), 6))
                if k not in seen:
                    seen.add(k)
                    unique_prop.append(np.clip(x, lo, hi))
            
            if len(unique_prop) == 0:
                break
            
            rng.shuffle(unique_prop)
            
            # Evaluate and exploit the top proposals
            n_exploit = min(3, len(unique_prop))
            for i in range(n_exploit):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                
                x_next = unique_prop[i]
                try:
                    f_next = eval_real(x_next)
                except Exception:
                    break
                
                model_X.append(x_next)
                model_Y.append(f_next)
                
                # Local exploitation
                if self.eval_count < self.budget - 10 and time.time() < deadline - 5:
                    bx, bf = nelder_mead_local(x_next, max_evals=5)
                    if bx is not None:
                        model_X.append(bx)
                        model_Y.append(bf)
                
                # Remove evaluated point from pool and add new random
                mask = np.array([not np.allclose(cand_pool[j], x_next, atol=1e-6) for j in range(len(cand_pool))])
                cand_pool = cand_pool[mask]
                new_cand = rng.uniform(lo, hi)
                cand_pool = np.vstack([cand_pool, new_cand.reshape(1, -1)])
        
        return None
# EVOLVE-BLOCK-END
