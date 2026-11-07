# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


class _Tree:
    """Shallow regression tree fitted to residuals via least-squares at leaves."""

    def __init__(self, max_depth=4, min_samples_leaf=5, max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.nodes = []  # list of dicts
        self.n_features = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.nodes = []
        self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node_idx = len(self.nodes)
        self.nodes.append({'leaf': True, 'value': 0.0})

        n = len(y)
        d = X.shape[1]

        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.std(y) < 1e-12:
            self.nodes[node_idx]['value'] = np.mean(y)
            return node_idx

        # Determine feature subset
        if self.max_features is not None and self.max_features < d:
            feat_idx = np.random.choice(d, size=self.max_features, replace=False)
        else:
            feat_idx = np.arange(d)

        best_mse = np.var(y) * n
        best_feat = -1
        best_thr = 0.0

        for f in feat_idx:
            x_vals = X[:, f]
            if np.std(x_vals) < 1e-12:
                continue
            # Candidate thresholds: midpoints of sorted unique values
            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            y_sorted = y[order]

            # Efficiently compute MSE for all splits
            # Use cumulative sums
            cum_sum = np.cumsum(y_sorted)
            cum_sq = np.cumsum(y_sorted ** 2)
            total_sum = cum_sum[-1]
            total_sq = cum_sq[-1]

            # Only consider splits where both sides have >= min_samples_leaf
            left_counts = np.arange(1, n)
            right_counts = n - left_counts
            valid = (left_counts >= self.min_samples_leaf) & (right_counts >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            lc = left_counts[valid]
            rc = right_counts[valid]
            ls = cum_sum[lc - 1]
            rs = total_sum - ls
            lq = cum_sq[lc - 1]
            rq = total_sq - lq

            # SSE = sum_left - (sum_left^2)/n_left + sum_right - (sum_right^2)/n_right
            sse = (lq - ls ** 2 / lc) + (rq - rs ** 2 / rc)

            min_sse = np.min(sse)
            if min_sse < best_mse:
                best_mse = min_sse
                # Find the threshold
                valid_indices = np.where(valid)[0]
                best_local = valid_indices[np.argmin(sse)]
                # Threshold is midpoint between x_sorted[best_local] and x_sorted[best_local+1]
                if best_local < n - 1:
                    best_thr = (x_sorted[best_local] + x_sorted[best_local + 1]) / 2.0
                else:
                    best_thr = x_sorted[best_local]
                best_feat = f

        if best_feat == -1:
            self.nodes[node_idx]['value'] = np.mean(y)
            return node_idx

        # Split
        mask = X[:, best_feat] <= best_thr
        self.nodes[node_idx] = {'leaf': False, 'feature': best_feat, 'threshold': best_thr,
                                 'left': -1, 'right': -1}
        left_idx = self._build(X[mask], y[mask], depth + 1)
        right_idx = self._build(X[~mask], y[~mask], depth + 1)
        self.nodes[node_idx]['left'] = left_idx
        self.nodes[node_idx]['right'] = right_idx

        return node_idx

    def predict(self, X):
        """Vectorized prediction."""
        preds = np.zeros(X.shape[0])
        for row_idx in range(X.shape[0]):
            x = X[row_idx]
            node = 0
            while not self.nodes[node]['leaf']:
                if x[self.nodes[node]['feature']] <= self.nodes[node]['threshold']:
                    node = self.nodes[node]['left']
                else:
                    node = self.nodes[node]['right']
            preds[row_idx] = self.nodes[node]['value']
        return preds


def _gradient_boosting(X, y, n_trees=200, learning_rate=0.1, max_depth=4,
                       min_samples_leaf=5, max_features=None, time_limit=25.0):
    """Gradient boosting for regression (L2 loss)."""
    n = X.shape[0]
    d = X.shape[1]
    start = time.time()

    # Initialize with mean
    F = np.full(n, np.mean(y))
    trees = []

    for t in range(n_trees):
        if time.time() - start > time_limit or _time_left() < 5.0:
            break

        # Residuals (negative gradient of L2 loss)
        residuals = y - F

        # Fit tree to residuals
        tree = _Tree(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                     max_features=max_features)
        tree.fit(X, residuals)

        # Update
        tree_pred = tree.predict(X)
        F += learning_rate * tree_pred
        trees.append(tree)

        # Check convergence
        if np.std(residuals) < 1e-10:
            break

    return F, trees


def _make_predictor(trees, base_value, learning_rate=0.1):
    """Create a predict callable from trained trees."""
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = np.full(X.shape[0], base_value)
        for tree in trees:
            preds += learning_rate * tree.predict(X)
        return preds
    return predict


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    if N == 0:
        return lambda X: np.zeros(len(X))

    # Normalize inputs for better tree splitting
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std

    # Determine time budget per equation
    # We have 10 equations, 330s total. Reserve ~30s for overhead.
    # So ~28s per equation. Use most of it for training.
    time_budget = min(28.0, max(5.0, _time_left() / 10.0))

    # Gradient boosting parameters tuned for small datasets
    n_trees = 300
    learning_rate = 0.05
    max_depth = 5
    min_samples_leaf = 3
    max_features = None  # use all features (d is small: 1-3)

    # Train
    base_value = np.mean(y_train)
    F_train, trees = _gradient_boosting(
        X_norm, y_train,
        n_trees=n_trees,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        time_limit=time_budget
    )

    # Check training MSE
    train_mse = np.mean((F_train - y_train) ** 2)
    y_var = np.var(y_train)
    if y_var > 0:
        norm_mse = train_mse / y_var
    else:
        norm_mse = train_mse

    # If fit is very poor, try a second round with different hyperparams
    if norm_mse > 0.1 and _time_left() > 15.0:
        time_budget2 = min(15.0, _time_left() / 3.0)
        F_train2, trees2 = _gradient_boosting(
            X_norm, y_train,
            n_trees=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_leaf=2,
            max_features=None,
            time_limit=time_budget2
        )
        train_mse2 = np.mean((F_train2 - y_train) ** 2)
        if train_mse2 < train_mse:
            F_train = F_train2
            trees = trees2
            learning_rate = 0.1
            base_value = np.mean(y_train)

    # Capture everything needed for prediction
    _trees = trees
    _base = base_value
    _lr = learning_rate
    _X_mean = X_mean
    _X_std = X_std

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - _X_mean) / _X_std
        preds = np.full(X.shape[0], _base)
        for tree in _trees:
            preds += _lr * tree.predict(X_norm)
        return preds

    return predict
# EVOLVE-BLOCK-END
