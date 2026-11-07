"""
Symbolic regression using Local Polynomial Regression (LOESS).
Strategy:
1. For each test point, find k nearest neighbors in the training set.
2. Fit a low-degree polynomial (degree 1 or 2) to these neighbors using weighted least squares,
   where weights are Gaussian or tricube functions of the distance.
3. This approximates the function as a piecewise polynomial that adapts to local geometry.
4. We perform a small grid search over the number of neighbors (k) and polynomial degree,
   using Leave-One-Out Cross-Validation (LOOCV) on a subsample to select the best hyperparameters.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0


def _tricube_weights(distances, bandwidth):
    """
    Compute tricube weights: w = (1 - (d/bw)^6)^3 if d < bw, else 0.
    distances: (n,)
    bandwidth: scalar
    """
    u = distances / bandwidth
    u = np.minimum(u, 1.0)
    w = (1 - u**6)**3
    return w


def _gaussian_weights(distances, bandwidth):
    """
    Compute Gaussian weights: w = exp(-(d/bw)^2).
    """
    u = distances / bandwidth
    return np.exp(-u**2)


def _fit_loess_predictor(X_train, y_train, k_neighbors, poly_degree, weight_type='tricube'):
    """
    Create a predictor function using LOESS.
    X_train: (n, d)
    y_train: (n,)
    k_neighbors: number of nearest neighbors to use
    poly_degree: degree of local polynomial
    weight_type: 'tricube' or 'gaussian'
    """
    n_train, d = X_train.shape
    
    # Pre-compute distances is too expensive for full matrix if n is large, 
    # but here n=200, so we can compute on the fly or precompute.
    # Since predict is called on 200 test points, and we do this for 4 equations,
    # total operations are manageable.
    
    def predict(X_test):
        n_test = X_test.shape[0]
        preds = np.zeros(n_test)
        
        # Design matrix for local polynomial
        # If d=1: [1, x, x^2, ...]
        # If d>1: We only have 1D inputs in this task, so d=1.
        # But let's be general. For d=1, features are powers of x.
        
        for i in range(n_test):
            x_test = X_test[i]
            
            # Compute distances to all training points
            # X_train shape (n_train, d), x_test shape (d,)
            dists = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
            
            # Find k nearest neighbors
            if k_neighbors >= n_train:
                idx = np.arange(n_train)
                # Bandwidth is max distance
                bw = np.max(dists[idx]) + 1e-8
            else:
                # Use argsort for small n
                idx = np.argpartition(dists, k_neighbors)[:k_neighbors]
                # Get the k-th smallest distance as bandwidth
                bw = np.partition(dists, k_neighbors-1)[k_neighbors-1] + 1e-8
            
            # Get neighbors
            x_neigh = X_train[idx]
            y_neigh = y_train[idx]
            d_neigh = dists[idx]
            
            # Compute weights
            if weight_type == 'tricube':
                w = _tricube_weights(d_neigh, bw)
            else:
                w = _gaussian_weights(d_neigh, bw)
            
            # Avoid zero weights causing issues
            w = np.maximum(w, 1e-10)
            
            # Build local design matrix
            # For d=1, we use powers of x.
            # For d>1, we would need multivariate polynomials, but task says 1 input.
            # Let's assume d=1 for simplicity as per task description.
            # If d > 1, we fallback to linear basis [1, x1, x2, ...] for safety?
            # No, task says "1 input(s)". So d=1.
            
            if d == 1:
                x_vals = x_neigh[:, 0]
                # Columns: 1, x, x^2, ..., x^degree
                Phi = np.zeros((len(x_neigh), poly_degree + 1))
                for k in range(poly_degree + 1):
                    Phi[:, k] = x_vals ** k
            else:
                # Fallback for multi-dim (not expected): use linear + pairwise products?
                # Just use linear for now to avoid complexity, as d=1 is guaranteed.
                Phi = np.hstack([np.ones((len(x_neigh), 1)), x_neigh])
                # Adjust degree to 1 effectively
                # But let's stick to the 1D assumption.
                
            # Weighted Least Squares
            # W = diag(w)
            # Solve (Phi^T W Phi) beta = Phi^T W y
            sqrt_w = np.sqrt(w)
            Phi_w = Phi * sqrt_w[:, None]
            y_w = y_neigh * sqrt_w
            
            try:
                # Add small ridge for stability
                lam = 1e-8
                A = Phi_w.T @ Phi_w + lam * np.eye(Phi.shape[1])
                b = Phi_w.T @ y_w
                beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(Phi_w, y_w, rcond=None)[0]
            
            # Predict
            if d == 1:
                x_test_val = x_test[0]
                Phi_test = np.array([x_test_val ** k for k in range(poly_degree + 1)])
            else:
                Phi_test = np.hstack([1, x_test])
                
            preds[i] = Phi_test @ beta
            
        return preds
    
    return predict


def _loocv_score(X, y, k_neighbors, poly_degree, weight_type='tricube'):
    """
    Estimate LOOCV score for a LOESS configuration.
    Since n=200, we can do actual LOOCV by refitting, but that's O(n^2 * k).
    Alternatively, we can use a subsample for speed.
    Given n=200, 200 fits of ~10-50 neighbors is fast enough.
    """
    n = X.shape[0]
    errors = []
    
    # To speed up, we might subsample the validation set if time is tight,
    # but 200 is small. Let's do all.
    for i in range(n):
        # Leave out i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X[mask]
        y_tr = y[mask]
        
        # Fit LOESS on X_tr, y_tr
        # We need to predict X[i]
        # Instead of building a full predictor, we do the local fit directly
        
        x_test = X[i]
        dists = np.sqrt(np.sum((X_tr - x_test)**2, axis=1))
        
        if k_neighbors >= len(X_tr):
            idx = np.arange(len(X_tr))
            bw = np.max(dists[idx]) + 1e-8
        else:
            idx = np.argpartition(dists, k_neighbors)[:k_neighbors]
            bw = np.partition(dists, k_neighbors-1)[k_neighbors-1] + 1e-8
            
        x_neigh = X_tr[idx]
        y_neigh = y_tr[idx]
        d_neigh = dists[idx]
        
        if weight_type == 'tricube':
            w = _tricube_weights(d_neigh, bw)
        else:
            w = _gaussian_weights(d_neigh, bw)
            
        w = np.maximum(w, 1e-10)
        
        d = X.shape[1]
        if d == 1:
            x_vals = x_neigh[:, 0]
            Phi = np.zeros((len(x_neigh), poly_degree + 1))
            for k in range(poly_degree + 1):
                Phi[:, k] = x_vals ** k
        else:
            Phi = np.hstack([np.ones((len(x_neigh), 1)), x_neigh])
            
        sqrt_w = np.sqrt(w)
        Phi_w = Phi * sqrt_w[:, None]
        y_w = y_neigh * sqrt_w
        
        try:
            lam = 1e-8
            A = Phi_w.T @ Phi_w + lam * np.eye(Phi.shape[1])
            b = Phi_w.T @ y_w
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Phi_w, y_w, rcond=None)[0]
            
        if d == 1:
            x_test_val = x_test[0]
            Phi_test = np.array([x_test_val ** k for k in range(poly_degree + 1)])
        else:
            Phi_test = np.hstack([1, x_test])
            
        pred = Phi_test @ beta
        errors.append((pred - y[i])**2)
        
    return np.mean(errors)


def fit(X_train, y_train):
    t_start = time.time()
    X = X_train
    y = y_train
    n, d = X.shape
    
    def time_left():
        return TIME_LIMIT - SAFETY_MARGIN - (time.time() - t_start)
    
    # Grid search for hyperparameters
    # k_neighbors: fraction of n or absolute number
    # Since n=200, let's try k = 10% to 50% of n
    k_options = [int(n * 0.1), int(n * 0.2), int(n * 0.3), int(n * 0.4), int(n * 0.5)]
    degree_options = [1, 2, 3]
    weight_options = ['tricube', 'gaussian']
    
    best_score = np.inf
    best_params = None
    
    # We will evaluate each configuration using LOOCV on the full training set.
    # 5 k's * 3 degrees * 2 weights = 30 configurations.
    # Each LOOCV takes ~200 local fits.
    # 30 * 200 = 6000 local fits.
    # Each local fit is a small linear system (size ~10-50).
    # This should be very fast (< 5 seconds).
    
    for k in k_options:
        if time_left() < 10:
            break
        for deg in degree_options:
            if time_left() < 10:
                break
            for wt in weight_options:
                if time_left() < 10:
                    break
                try:
                    score = _loocv_score(X, y, k, deg, wt)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_params = (k, deg, wt)
                except Exception:
                    continue
    
    # If no valid params found, fallback to reasonable defaults
    if best_params is None:
        best_params = (int(n * 0.3), 2, 'tricube')
    
    k_best, deg_best, wt_best = best_params
    
    # Create the final predictor
    try:
        predictor = _fit_loess_predictor(X, y, k_best, deg_best, wt_best)
        # Test prediction on a few points to ensure it works
        test_pred = predictor(X[:5])
        if not np.all(np.isfinite(test_pred)):
            raise ValueError("Non-finite prediction")
        return predictor
    except Exception:
        # Fallback: Simple linear regression
        A = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            def predict_linear(X_test):
                A_test = np.column_stack([np.ones(X_test.shape[0])] + [X_test[:, j] for j in range(d)])
                return A_test @ w
            return predict_linear
        except Exception:
            # Last resort: mean
            mean_y = np.mean(y)
            return lambda X_test: np.full(X_test.shape[0], mean_y)
