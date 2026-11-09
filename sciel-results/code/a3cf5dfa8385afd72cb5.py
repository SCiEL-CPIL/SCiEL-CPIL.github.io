# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    def check_time():
        return time.time() < deadline
    
    best_predict = None
    best_val_mse = np.inf
    
    def safe_evaluate(model, X, y, val_frac=0.2):
        if n >= 20:
            val_idx = np.arange(int(n * (1 - val_frac)), n)
            tr_idx = np.arange(0, int(n * (1 - val_frac)))
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_val, y_val = X, y
        try:
            preds = model(X_val)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y_val.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y_val) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    def update_best(model):
        nonlocal best_predict, best_val_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_val_mse:
            best_val_mse = mse
            best_predict = model
    
    # Sum-of-Sines Kernel Ridge Regression
    def make_sines_kernel_model(n_freqs_per_dim=8, include_cross=True, cross_order=2):
        """
        Kernel: K(x, x') = sum over all frequency combinations of 
                  sin(f_i * x_i) * sin(f_j * x_j) etc.
        We build a feature matrix with sin and cos terms for each dim at each frequency,
        plus cross terms for pairs/triples of dimensions.
        """
        # Determine frequency grid based on input ranges
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        ranges = X_max - X_min
        ranges[ranges < 1e-6] = 1.0
        
        # Frequencies: use a grid that captures the range well
        # For range R, fundamental frequency ~ pi/R, then harmonics
        freqs_per_dim = []
        for j in range(d):
            R = ranges[j]
            # Use frequencies from ~1/R to ~n_freqs_per_dim/R
            base_freq = np.pi / R
            freqs = base_freq * np.arange(1, n_freqs_per_dim + 1)
            freqs_per_dim.append(freqs)
        
        def build_features(X):
            cols = [np.ones(len(X))]
            # Sin and cos for each dimension at each frequency
            for j in range(d):
                for f in freqs_per_dim[j]:
                    cols.append(np.sin(f * X[:, j]))
                    cols.append(np.cos(f * X[:, j]))
            
            # Cross terms for pairs of dimensions
            if include_cross and d >= 2:
                max_pairs = min(cross_order, d)
                for i in range(d):
                    for j in range(i + 1, d):
                        if j - i >= max_pairs:
                            continue
                        # Use low frequencies for cross terms
                        fi = freqs_per_dim[i][:2]
                        fj = freqs_per_dim[j][:2]
                        for f1 in fi:
                            for f2 in fj:
                                cols.append(np.sin(f1 * X[:, i] + f2 * X[:, j]))
                                cols.append(np.cos(f1 * X[:, i] + f2 * X[:, j]))
                                cols.append(np.sin(f1 * X[:, i] - f2 * X[:, j]))
                                cols.append(np.cos(f1 * X[:, i] - f2 * X[:, j]))
                                cols.append(np.sin(f1 * X[:, i]) * np.sin(f2 * X[:, j]))
                                cols.append(np.cos(f1 * X[:, i]) * np.cos(f2 * X[:, j]))
                                cols.append(np.sin(f1 * X[:, i]) * np.cos(f2 * X[:, j]))
                                cols.append(np.cos(f1 * X[:, i]) * np.sin(f2 * X[:, j]))
            
            # Triple cross terms for 3D
            if include_cross and d >= 3 and cross_order >= 3:
                for i in range(d):
                    for j in range(i + 1, d):
                        for k in range(j + 1, d):
                            fi = freqs_per_dim[i][0]
                            fj = freqs_per_dim[j][0]
                            fk = freqs_per_dim[k][0]
                            cols.append(np.sin(fi * X[:, i] + fj * X[:, j] + fk * X[:, k]))
                            cols.append(np.cos(fi * X[:, i] + fj * X[:, j] + fk * X[:, k]))
                            cols.append(np.sin(fi * X[:, i]) * np.sin(fj * X[:, j]) * np.sin(fk * X[:, k]))
            
            return np.column_stack(cols)
        
        F_train = build_features(X_train)
        n_features = F_train.shape[1]
        
        # Ridge regression with adaptive regularization
        # Use stronger regularization for more features
        lam = 1e-4 * max(1, n_features / 50)
        
        try:
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            try:
                w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
            except Exception:
                return None
        
        def predict(X):
            F = build_features(X)
            return F @ w
        return predict
    
    # 1. Try Sum-of-Sines kernel with various settings
    sine_configs = [
        (4, True, 2),   # 4 freqs per dim, cross pairs
        (6, True, 2),   # 6 freqs per dim, cross pairs
        (8, True, 2),   # 8 freqs per dim, cross pairs
        (4, True, 3),   # 4 freqs per dim, cross pairs+triples
        (6, True, 3),   # 6 freqs per dim, cross pairs+triples
        (3, False, 0),  # 3 freqs per dim, no cross
        (5, False, 0),  # 5 freqs per dim, no cross
    ]
    
    for n_freq, inc_cross, cross_ord in sine_configs:
        if not check_time():
            break
        model = make_sines_kernel_model(n_freqs_per_dim=n_freq, 
                                         include_cross=inc_cross, 
                                         cross_order=cross_ord)
        update_best(model)
    
    # 2. Polynomial fallback
    def make_poly_model(degree):
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            if degree >= 3:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            return np.column_stack(cols)
        
        F_train = features(X_train)
        try:
            lam = 1e-6
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
        def predict(X):
            return features(X) @ w
        return predict
    
    for deg in [1, 2, 3]:
        if not check_time():
            break
        model = make_poly_model(deg)
        update_best(model)
    
    # 3. RBF fallback
    def make_rbf_model(gamma, n_centers=50):
        if n > n_centers:
            idx = np.linspace(0, n - 1, n_centers, dtype=int)
            centers = X_train[idx]
        else:
            centers = X_train
        
        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F_train = np.exp(-gamma * dists)
        
        try:
            lam = 1e-6
            A = F_train.T @ F_train
            b = F_train.T @ y_train
            A_reg = A.copy()
            A_reg[1:, 1:] += lam
            w = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
        def predict(X):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            return F_test @ w
        return predict
    
    X_range = np.ptp(X_train, axis=0)
    avg_range = np.mean(X_range)
    if avg_range == 0:
        avg_range = 1.0
    
    base_gamma = 1.0 / (d * avg_range ** 2)
    for g in [base_gamma * 0.5, base_gamma, base_gamma * 2]:
        if not check_time():
            break
        model = make_rbf_model(g, n_centers=50)
        update_best(model)

    # Fallback
    if best_predict is None:
        return lambda X: np.full(X.shape[0], y_mean)
        
    return best_predict

# EVOLVE-BLOCK-END
