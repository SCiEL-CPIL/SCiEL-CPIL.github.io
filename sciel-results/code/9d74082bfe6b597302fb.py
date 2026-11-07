# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X, d):
    """Build a rich set of candidate basis functions evaluated on X. Returns (N, p) array."""
    N = X.shape[0]
    cols = []

    # Constant
    cols.append(np.ones(N))

    # Linear
    for j in range(d):
        cols.append(X[:, j])

    # Powers up to 6
    for j in range(d):
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
        cols.append(X[:, j] ** 4)
        cols.append(X[:, j] ** 5)
        cols.append(X[:, j] ** 6)

    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])

    # Triple products
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])

    # Trig
    for j in range(d):
        xj = X[:, j]
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        cols.append(np.sin(2.0 * xj))
        cols.append(np.cos(2.0 * xj))
        cols.append(np.sin(3.0 * xj))
        cols.append(np.cos(3.0 * xj))

    # Exp, log, sqrt, sigmoid, tanh
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        cols.append(np.exp(-xj))
        cols.append(np.exp(2.0 * xj))
        cols.append(np.exp(-2.0 * xj))
        cols.append(np.log1p(xj))
        cols.append(np.sqrt(xj))
        cols.append(np.tanh(xj))
        cols.append(np.tanh(2.0 * xj))
        cols.append(1.0 / (1.0 + np.exp(-xj)))

    # Cross terms with trig and exp
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                xi, xj = X[:, i], X[:, j]
                cols.append(np.sin(xi) * xj)
                cols.append(np.cos(xi) * xj)
                cols.append(np.sin(xi) * np.sin(xj))
                cols.append(np.cos(xi) * np.cos(xj))
                cols.append(xi * np.exp(xj))
                cols.append(np.exp(xi) * xj)
                cols.append(xi * np.sqrt(xj))
                cols.append(np.sqrt(xi) * xj)
                cols.append(np.sin(xi) * np.exp(xj))
                cols.append(np.cos(xi) * np.exp(xj))

    # Global features
    cols.append(np.sum(X, axis=1))
    cols.append(np.prod(X, axis=1))
    cols.append(np.sum(X ** 2, axis=1))
    cols.append(np.sum(X ** 3, axis=1))
    cols.append(np.linalg.norm(X, axis=1))
    cols.append(np.linalg.norm(X, axis=1) ** 2)
    cols.append(np.min(X, axis=1))
    cols.append(np.max(X, axis=1))

    # Differences
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] - X[:, j])
                cols.append(X[:, i] * (X[:, i] - X[:, j]))

    return np.column_stack(cols)


def _fit_ridge(F, y, alpha=1e-8):
    n, p = F.shape
    A = F.T @ F + alpha * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _greedy_forward_select(F, y, max_features=20, tol=1e-10):
    """
    Greedy forward selection using precomputed Gram matrix.
    Returns list of selected feature indices.
    """
    n, p = F.shape
    if p == 0:
        return []
    
    # Precompute Gram matrix and y projections
    G = F.T @ F
    gy = F.T @ y
    y_norm_sq = y @ y
    
    selected = []
    residual_ss = y_norm_sq  # Total sum of squares
    
    for step in range(min(max_features, p)):
        if _time_left() < 3.0:
            break
        
        best_idx = -1
        best_improvement = tol
        
        # For each unselected feature, compute the improvement
        # If we add feature j to current model, the new coefficient is:
        # w_j = (gy_j - G[j, selected] @ w_selected) / G[j, j]
        # and the improvement in explained variance is:
        # delta = (w_j)^2 * G[j, j] = (gy_j - G[j, selected] @ w_selected)^2 / G[j, j]
        
        if len(selected) == 0:
            # First feature: just pick the one with max |gy_j| / G[j,j]
            improvements = gy ** 2 / np.maximum(np.diag(G), 1e-15)
            # Exclude already selected (none yet)
            best_idx = int(np.argmax(improvements))
            best_improvement = improvements[best_idx]
        else:
            # Compute the projection of gy onto selected features
            G_sel = G[np.ix_(selected, selected)]
            gy_sel = gy[selected]
            
            try:
                w_sel = np.linalg.solve(G_sel + 1e-10 * np.eye(len(selected)), gy_sel)
            except np.linalg.LinAlgError:
                w_sel = np.linalg.lstsq(G_sel, gy_sel, rcond=None)[0]
            
            # For each candidate, compute the residual correlation
            # rho_j = gy_j - G[j, selected] @ w_sel
            G_j_sel = G[:, selected]  # (p, k)
            rho = gy - G_j_sel @ w_sel  # (p,)
            
            # Improvement for adding feature j: rho_j^2 / G[j, j]
            improvements = rho ** 2 / np.maximum(np.diag(G), 1e-15)
            
            # Mask out already selected
            for s in selected:
                improvements[s] = 0.0
            
            best_idx = int(np.argmax(improvements))
            best_improvement = improvements[best_idx]
        
        if best_improvement < tol:
            break
        
        selected.append(best_idx)
    
    return selected


def _rbf_predict(X_train, y_train, X, gamma, alpha=1e-6):
    Xa = X_train
    Xb = X
    sq_a = np.sum(Xa ** 2, axis=1)[:, None]
    sq_b = np.sum(Xb ** 2, axis=1)[None, :]
    K_train = np.exp(-gamma * (sq_a + sq_a.T - 2.0 * Xa @ Xa.T))
    K_test = np.exp(-gamma * (sq_a + sq_b - 2.0 * Xa @ X.T))

    n = X_train.shape[0]
    A = K_train + alpha * np.eye(n)
    try:
        theta = np.linalg.solve(A, y_train)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, y_train, rcond=None)[0]
    return K_test @ theta


def _gp_predict(X_train, y_train, X, length_scale, noise, amplitude=1.0):
    """
    Gaussian Process prediction with RBF kernel.
    Returns predictions for X.
    """
    Xa = X_train
    Xb = X
    
    # RBF kernel
    def rbf_kernel(X1, X2, ls):
        sq1 = np.sum(X1 ** 2, axis=1)[:, None]
        sq2 = np.sum(X2 ** 2, axis=1)[None, :]
        sq_dist = sq1 + sq2 - 2.0 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0.0)
        return amplitude ** 2 * np.exp(-0.5 * sq_dist / (ls ** 2))
    
    K_train = rbf_kernel(Xa, Xa, length_scale) + noise * np.eye(len(Xa))
    K_test = rbf_kernel(Xa, Xb, length_scale)
    K_test_X = rbf_kernel(Xb, Xa, length_scale)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_test_X @ alpha_vec
        v = np.linalg.solve(L, K_test)
        var = amplitude ** 2 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
    except np.linalg.LinAlgError:
        # Fallback to regularized solve
        K_reg = K_train + 1e-6 * np.eye(len(Xa))
        try:
            alpha_vec = np.linalg.solve(K_reg, y_train)
            mu = K_test_X @ alpha_vec
        except np.linalg.LinAlgError:
            mu = np.full(len(Xb), np.mean(y_train))
    
    return mu


def _optimize_gp_hyperparams(X_train, y_train, d, max_evals=30):
    """
    Optimize GP hyperparameters (length_scale, noise) using L-BFGS.
    Returns best (length_scale, noise, amplitude) and a predict function.
    """
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    # Initial guesses
    X_scale = np.std(X_train, axis=0)
    X_scale[X_scale < 1e-12] = 1.0
    d_eff = np.sqrt(np.sum(X_scale ** 2))
    
    best_params = None
    best_negloglik = np.inf
    best_predict = None
    
    # Try several initial points
    init_ls = [0.5 * d_eff, 1.0 * d_eff, 2.0 * d_eff, 0.1 * d_eff]
    init_noise = [1e-4, 1e-3, 1e-2]
    
    for ls0 in init_ls:
        for noise0 in init_noise:
            if _time_left() < 5.0:
                break
            
            def negloglik(params):
                ls = np.exp(params[0])
                noise = np.exp(params[1])
                amp = np.exp(params[2])
                
                if ls < 1e-6 or noise < 1e-10:
                    return 1e10
                
                try:
                    K_train = amp ** 2 * np.exp(-0.5 * cdist(X_train, X_train) ** 2 / (2 * ls ** 2)) + noise * np.eye(n)
                    L = np.linalg.cholesky(K_train)
                    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    logdet = 2.0 * np.sum(np.log(np.diag(L)))
                    nll = 0.5 * y_norm @ alpha_vec + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
                    return float(nll)
                except np.linalg.LinAlgError:
                    return 1e10
            
            x0 = np.array([np.log(max(ls0, 1e-3)), np.log(max(noise0, 1e-6)), 0.0])
            
            try:
                result = minimize(negloglik, x0, method='L-BFGS-B', 
                                  bounds=[(np.log(1e-3), np.log(10 * d_eff)), 
                                         (np.log(1e-8), np.log(1.0)),
                                         (-2.0, 2.0)],
                                  options={'maxiter': 50, 'maxfun': 200})
                if result.fun < best_negloglik:
                    best_negloglik = result.fun
                    best_params = result.x
            except Exception:
                continue
    
    if best_params is None:
        # Default
        best_params = np.array([np.log(d_eff), np.log(1e-3), 0.0])
    
    best_ls = np.exp(best_params[0])
    best_noise = np.exp(best_params[1])
    best_amp = np.exp(best_params[2])
    
    # Create predict function
    def gp_pred(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mu = _gp_predict(X_train, y_norm, X, best_ls, best_noise, best_amp)
        return (mu * y_std + y_mean).flatten()
    
    return gp_pred, best_negloglik


def _train_neural_net(X_train, y_train, X_test, hidden_sizes=(64, 32), lr=0.005, 
                       epochs=5000, batch_size=32, weight_decay=1e-4, patience=100):
    """
    Train a small fully-connected neural network using Adam optimizer with LR scheduling.
    Returns the trained model parameters and a predict function.
    """
    N, d = X_train.shape
    y_train = y_train.reshape(-1, 1)
    
    # Standardize inputs and outputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-12:
        y_std = 1.0
    
    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std
    
    # Define architecture
    layers = [d] + list(hidden_sizes) + [1]
    
    # Initialize weights with He initialization
    rng = np.random.RandomState(42)
    W = []
    b = []
    for i in range(len(layers) - 1):
        fan_in = layers[i]
        fan_out = layers[i + 1]
        W.append(rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
        b.append(np.zeros(fan_out))
    
    # Adam optimizer state
    m_W = [np.zeros_like(w) for w in W]
    v_W = [np.zeros_like(w) for w in W]
    m_b = [np.zeros_like(bi) for bi in b]
    v_b = [np.zeros_like(bi) for bi in b]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    def forward(X):
        activations = [X]
        z = X
        for i in range(len(W)):
            z = z @ W[i] + b[i]
            if i < len(W) - 1:
                z = np.tanh(z)
            activations.append(z)
        return activations
    
    def backward(activations, dL_dy):
        dW = [None] * len(W)
        db = [None] * len(b)
        dz = dL_dy
        for i in range(len(W) - 1, -1, -1):
            dW[i] = activations[i].T @ dz
            db[i] = dz.sum(axis=0)
            if i > 0:
                dz = (dz @ W[i].T) * (1 - activations[i + 1] ** 2)
        return dW, db
    
    # Training loop with early stopping
    best_loss = np.inf
    best_params = (W.copy(), b.copy())
    patience_counter = 0
    
    for epoch in range(epochs):
        if _time_left() < 5.0:
            break
        
        # Shuffle
        idx = np.random.permutation(N)
        
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = idx[start:end]
            X_batch = X_norm[batch_idx]
            y_batch = y_norm[batch_idx]
            
            # Forward
            activations = forward(X_batch)
            y_pred = activations[-1]
            
            # Loss and gradient
            loss = np.mean((y_pred - y_batch) ** 2)
            epoch_loss += loss
            n_batches += 1
            
            # Weight decay
            dL_dy = 2.0 * (y_pred - y_batch) / len(batch_idx)
            dW, db = backward(activations, dL_dy)
            for i in range(len(W)):
                dW[i] += weight_decay * W[i]
            
            # Adam update with LR decay
            t += 1
            lr_current = lr / (1.0 + 0.001 * t)
            for i in range(len(W)):
                m_W[i] = beta1 * m_W[i] + (1 - beta1) * dW[i]
                v_W[i] = beta2 * v_W[i] + (1 - beta2) * dW[i] ** 2
                m_hat = m_W[i] / (1 - beta1 ** t)
                v_hat = v_W[i] / (1 - beta2 ** t)
                W[i] -= lr_current * m_hat / (np.sqrt(v_hat) + eps)
                
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * db[i] ** 2
                m_hat_b = m_b[i] / (1 - beta1 ** t)
                v_hat_b = v_b[i] / (1 - beta2 ** t)
                b[i] -= lr_current * m_hat_b / (np.sqrt(v_hat_b) + eps)
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_params = (W.copy(), b.copy())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            break
        
        # Early stopping if loss is very small
        if avg_loss < 1e-8:
            break
    
    # Use best parameters
    W_best, b_best = best_params
    
    # Create predict function
    def nn_predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - X_mean) / X_std
        z = X_norm
        for i in range(len(W_best)):
            z = z @ W_best[i] + b_best[i]
            if i < len(W_best) - 1:
                z = np.tanh(z)
        return (z * y_std + y_mean).flatten()
    
    return nn_predict, avg_loss


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    if N == 0:
        return lambda X: np.zeros(np.asarray(X).shape[0])

    # Build basis
    F = _build_basis(X_train, d)
    p = F.shape[1]

    # Ensure finite
    finite_cols = np.all(np.isfinite(F), axis=0)
    F = F[:, finite_cols]
    p = F.shape[1]
    finite_mask = finite_cols

    if p == 0:
        return lambda X: np.zeros(np.asarray(X).shape[0])

    # Standardize features (keep constant)
    F_mean = np.mean(F, axis=0)
    F_std = np.std(F, axis=0)
    F_std[F_std < 1e-12] = 1.0
    F_std[0] = 1.0
    F_mean[0] = 0.0
    F_norm = (F - F_mean) / F_std

    # y stats
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-12:
        return lambda X: np.full(np.asarray(X).shape[0], y_mean)
    y_norm = (y_train - y_mean) / y_std

    # Greedy forward selection
    active = _greedy_forward_select(F_norm, y_norm, max_features=25)

    if len(active) == 0:
        active = [0]

    F_active = F_norm[:, active]
    w_active = _fit_ridge(F_active, y_norm, alpha=1e-8)

    pred_train_norm = F_active @ w_active
    train_mse_norm = np.mean((pred_train_norm - y_norm) ** 2)
    train_mse = train_mse_norm * y_std ** 2

    best_mse = train_mse
    best_type = "basis"
    best_gp_predict = None
    best_nn_predict = None

    # GP fallback - try if basis didn't fit well
    if train_mse_norm > 0.02 and _time_left() > 20.0:
        try:
            gp_pred, gp_nll = _optimize_gp_hyperparams(X_train, y_train, d)
            pred_gp = gp_pred(X_train)
            mse_gp = np.mean((pred_gp - y_train) ** 2)
            if mse_gp < best_mse:
                best_mse = mse_gp
                best_type = "gp"
                best_gp_predict = gp_pred
        except Exception:
            pass

    # Neural network fallback if other methods didn't fit well
    if train_mse_norm > 0.01 and _time_left() > 25.0:
        try:
            nn_predict, nn_loss = _train_neural_net(X_train, y_train, X_train, 
                                                     hidden_sizes=(64, 32), 
                                                     lr=0.005, epochs=5000, 
                                                     batch_size=32)
            pred_nn = nn_predict(X_train)
            mse_nn = np.mean((pred_nn - y_train) ** 2)
            if mse_nn < best_mse:
                best_mse = mse_nn
                best_type = "nn"
                best_nn_predict = nn_predict
        except Exception:
            pass

    # Capture
    sel_indices = active
    F_mean_c = F_mean.copy()
    F_std_c = F_std.copy()
    w_c = w_active.copy()
    y_mean_c = y_mean
    y_std_c = y_std
    best_type_c = best_type
    X_train_c = X_train.copy()
    y_train_c = y_train.copy()
    finite_mask_c = finite_mask.copy()
    best_gp_predict_c = best_gp_predict
    best_nn_predict_c = best_nn_predict

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if best_type_c == "nn" and best_nn_predict_c is not None:
            return best_nn_predict_c(X)
        elif best_type_c == "gp" and best_gp_predict_c is not None:
            return best_gp_predict_c(X)
        else:
            F_pred = _build_basis(X, d)
            F_pred = F_pred[:, finite_mask_c]
            F_pred_norm = (F_pred - F_mean_c) / F_std_c
            F_pred_sel = F_pred_norm[:, sel_indices]
            return F_pred_sel @ w_c * y_std_c + y_mean_c

    return predict
# EVOLVE-BLOCK-END
