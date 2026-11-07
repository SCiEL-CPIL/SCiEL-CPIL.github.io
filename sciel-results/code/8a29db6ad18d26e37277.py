"""Multi-objective optimizer on RE37 using DNN surrogate + L-BFGS-B local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import torch
import torch.nn as nn
from scipy.optimize import minimize


class Optimizer:
    """DNN surrogate ensemble + L-BFGS-B local search for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 285.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        max_evals = self.budget

        used = 0
        archive_X = []
        archive_F = []

        def is_pareto(f, archive_F):
            if len(archive_F) == 0:
                return True
            A = np.asarray(archive_F)
            return not np.any(np.all(A <= f, axis=1) & np.any(A < f, axis=1))

        def add_to_archive(x, f):
            new_X = []
            new_F = []
            for xi, fi in zip(archive_X, archive_F):
                if not (np.all(f <= fi) and np.any(f < fi)):
                    new_X.append(xi)
                    new_F.append(fi)
            archive_X[:] = new_X
            archive_F[:] = new_F
            if is_pareto(f, archive_F):
                archive_X.append(np.array(x, dtype=float))
                archive_F.append(np.array(f, dtype=float))

        def eval_point(x):
            nonlocal used
            if used >= max_evals:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        # DNN surrogate model
        class Surrogate(nn.Module):
            def __init__(self, dim, n_obj):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_obj)
                )

            def forward(self, x):
                return self.net(x)

        def train_surrogate(X, F, epochs=50):
            model = Surrogate(dim, n_obj)
            X_t = torch.tensor(X, dtype=torch.float32)
            F_t = torch.tensor(F, dtype=torch.float32)
            # Normalize inputs to [0,1]
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1.0
            X_norm = (X_t - X_min) / X_range

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                pred = model(X_norm)
                loss = criterion(pred, F_t)
                loss.backward()
                optimizer.step()
            return model, X_min, X_range

        def predict_surrogate(model, X_min, X_range, x):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_norm = (x_t - X_min) / X_range
            model.eval()
            with torch.no_grad():
                pred = model(x_norm).numpy()[0]
            return pred

        # Phase 1: Initial random sampling
        n_init = min(15, max_evals)
        for i in range(n_init):
            x = lo + span * np.random.rand(dim)
            res = eval_point(x)
            if res is None:
                break
            x, f = res
            add_to_archive(x, f)

        # Phase 2: Train surrogate and use for guided search
        if len(archive_X) >= 8 and time.time() < deadline:
            X_data = np.array(archive_X)
            F_data = np.array(archive_F)
            model, X_min, X_range = train_surrogate(X_data, F_data, epochs=100)

            # Generate candidate points from surrogate
            n_candidates = 20
            for _ in range(n_candidates):
                if used >= max_evals or time.time() > deadline:
                    break

                # Random point in [0,1]^dim
                x_norm = np.random.rand(dim)
                x = X_min + x_norm * X_range
                x = np.clip(x, lo, hi)

                # Predict objectives
                pred_f = predict_surrogate(model, X_min, X_range, x)

                # Check if predicted point is potentially non-dominated
                if is_pareto(pred_f, archive_F):
                    res = eval_point(x)
                    if res is not None:
                        xp, fp = res
                        add_to_archive(xp, fp)

        # Phase 3: L-BFGS-B local search from archive points
        def local_search_lbfgsb(x0, w, max_iter=20):
            nonlocal used
            if used >= max_evals:
                return None
            x0 = np.clip(x0, lo, hi)

            def obj(x):
                nonlocal used
                if used >= max_evals:
                    return 1e9
                f = np.asarray(problem(x), dtype=float)
                used += 1
                return float(np.dot(w, f))

            try:
                bounds_list = list(zip(lo, hi))
                res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds_list,
                               options={'maxiter': max_iter, 'gtol': 1e-6})
                xbest = np.clip(res.x, lo, hi)
                fbest = np.asarray(problem(xbest), dtype=float)
                used += 1
                return xbest, fbest
            except Exception:
                return None

        # Weight vectors for local search
        weight_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.25, 0.25]),
            np.array([0.25, 0.5, 0.25]),
            np.array([0.25, 0.25, 0.5]),
            np.array([1.0/3, 1.0/3, 1.0/3]),
        ]

        # Add random weights
        for _ in range(10):
            w = np.random.rand(n_obj)
            w = w / w.sum()
            weight_vectors.append(w)

        # Local search from archive points
        for w in weight_vectors:
            if used >= max_evals or time.time() > deadline:
                break
            # Pick a random archive point as start
            idx = np.random.randint(len(archive_X))
            x0 = archive_X[idx]
            res = local_search_lbfgsb(x0, w, max_iter=15)
            if res is not None:
                x, f = res
                add_to_archive(x, f)

        # Phase 4: More random local searches if budget remains
        while used < max_evals and time.time() < deadline:
            w = np.random.rand(n_obj)
            w = w / w.sum()
            x0 = lo + span * np.random.rand(dim)
            res = local_search_lbfgsb(x0, w, max_iter=10)
            if res is not None:
                x, f = res
                add_to_archive(x, f)

        # Ensure at least something returned
        if len(archive_X) == 0:
            x = lo + span * np.random.rand(dim)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            archive_X.append(np.array(x))
            archive_F.append(np.array(f))

        return np.array(archive_X)
# EVOLVE-BLOCK-END
