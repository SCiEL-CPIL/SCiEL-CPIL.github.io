"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 310.0  # safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Normalize y for better conditioning? No, KRR handles scale with lambda.
    # But we need to be careful with the kernel scaling.
    
    # Define Kernel Functions
    def kernel_poly(X1, X2, degree=3, gamma=1.0, coef0=1.0):
        return (X1 @ X2.T + coef0) ** degree

    def kernel_rbf(X1, X2, gamma=1.0):
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-gamma * sq_dists)

    def kernel_sine(X1, X2, gamma=1.0):
        # Periodic kernel, good for oscillatory functions
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.sin(gamma * sq_dists)

    def combined_kernel(X1, X2, params):
        # params: dict with keys 'w_poly', 'w_rbf', 'w_sin', 'gamma_poly', 'gamma_rbf', 'gamma_sin'
        K = np.zeros((X1.shape[0], X2.shape[0]))
        if params.get('w_poly', 0) > 0:
            K += params['w_poly'] * kernel_poly(X1, X2, degree=params.get('deg_poly', 3), gamma=1.0, coef0=params.get('coef0', 1.0))
        if params.get('w_rbf', 0) > 0:
            K += params['w_rbf'] * kernel_rbf(X1, X2, gamma=params.get('gamma_rbf', 1.0))
        if params.get('w_sin', 0) > 0:
            K += params['w_sin'] * kernel_sine(X1, X2, gamma=params.get('gamma_sin', 1.0))
        return K

    # Grid search for hyperparameters
    # We use a simple train/val split for speed, or LOOCV if time permits.
    # Given N=200, LOOCV is feasible for a small grid.
    
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size = max(20, int(n * 0.2))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    
    var_y = np.var(y)
    if var_y < 1e-15:
        var_y = 1e-15

    def eval_pred(pred, y_eval):
        pred = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            return np.inf
        return np.mean((pred - y_eval) ** 2) / var_y

    best_score = np.inf
    best_params = None
    best_alpha = None

    # Hyperparameter grid
    # Lambdas (regularization)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    # Kernel weights and gammas
    # We will try a few combinations of kernels
    kernel_configs = [
        # (w_poly, deg_poly, coef0, w_rbf, gamma_rbf, w_sin, gamma_sin)
        (1.0, 2, 1.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 3, 1.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 4, 1.0, 0.0, 1.0, 0.0, 1.0),
        (0.0, 2, 1.0, 1.0, 0.1, 0.0, 1.0),
        (0.0, 2, 1.0, 1.0, 0.5, 0.0, 1.0),
        (0.0, 2, 1.0, 1.0, 1.0, 0.0, 1.0),
        (0.0, 2, 1.0, 1.0, 2.0, 0.0, 1.0),
        (0.0, 2, 1.0, 0.0, 1.0, 1.0, 0.5),
        (0.0, 2, 1.0, 0.0, 1.0, 1.0, 1.0),
        (0.0, 2, 1.0, 0.0, 1.0, 1.0, 2.0),
        (1.0, 2, 1.0, 1.0, 1.0, 0.0, 1.0),
        (1.0, 2, 1.0, 0.0, 1.0, 1.0, 1.0),
        (1.0, 3, 1.0, 1.0, 0.5, 1.0, 1.0),
    ]

    for config in kernel_configs:
        w_poly, deg_poly, coef0, w_rbf, gamma_rbf, w_sin, gamma_sin = config
        params = {
            'w_poly': w_poly, 'deg_poly': deg_poly, 'coef0': coef0,
            'w_rbf': w_rbf, 'gamma_rbf': gamma_rbf,
            'w_sin': w_sin, 'gamma_sin': gamma_sin
        }
        
        if time.time() > deadline - 30:
            break

        for lam in lambdas:
            try:
                K_tr = combined_kernel(X_tr, X_tr, params)
                # Add regularization
                A = K_tr + lam * np.eye(X_tr.shape[0])
                
                # Solve for alpha
                alpha = np.linalg.solve(A, y_tr)
                
                # Predict on validation
                K_val = combined_kernel(X_val, X_tr, params)
                pred_val = K_val @ alpha
                
                score = eval_pred(pred_val, y_val)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_alpha = alpha
                    best_lam = lam

            except Exception:
                continue

    # Refit on all data with best parameters
    if best_params is not None:
        try:
            K_full = combined_kernel(X, X, best_params)
            A_full = K_full + best_lam * np.eye(n)
            alpha_full = np.linalg.solve(A_full, y)
            
            def predict(X_pred):
                Xp = np.asarray(X_pred, dtype=np.float64)
                K_test = combined_kernel(Xp, X, best_params)
                return K_test @ alpha_full
            
            return predict

        except Exception:
            pass

    # Fallback: Simple Linear Regression (Polynomial degree 1)
    try:
        # Add bias
        X_ext = np.hstack([X, np.ones((n, 1))])
        w, *_ = np.linalg.lstsq(X_ext, y, rcond=None)
        
        def predict(X_pred):
            Xp = np.asarray(X_pred, dtype=np.float64)
            Xp_ext = np.hstack([Xp, np.ones((Xp.shape[0], 1))])
            return Xp_ext @ w
        return predict
    except Exception:
        mean_y = np.mean(y)
        def predict(X_pred):
            return np.full(len(X_pred), mean_y)
        return predict
# EVOLVE-BLOCK-END
