"""Symbolic regression via ensemble of GP (Matern-52) and polynomial features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Center and scale inputs for numerical stability
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    def time_check():
        return time.time() < deadline

    # --- Component 1: GP with Matern-52 kernel ---
    def matern52_kernel(X1, X2, length_scale, sigma_f):
        """Matern-52 kernel: k(r) = sigma_f^2 * (1 + sqrt(3)*r/ls) * exp(-sqrt(3)*r/ls)"""
        D = cdist(X1, X2)  # Euclidean distances
        r = D / length_scale
        return (sigma_f ** 2) * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

    def gp_predict(X_pred, X_train_scaled, y_train_scaled, length_scale, sigma_f, sigma_n):
        """Predict using GP with Matern-52 kernel."""
        K = matern52_kernel(X_train_scaled, X_train_scaled, length_scale, sigma_f)
        K[np.diag_indices_from(K)] += sigma_n ** 2
        Kpred = matern52_kernel(X_pred, X_train_scaled, length_scale, sigma_f)
        try:
            alpha = np.linalg.solve(K, y_train_scaled)
            return Kpred @ alpha
        except np.linalg.LinAlgError:
            # Fallback: add jitter
            K[np.diag_indices_from(K)] += 1e-6
            try:
                alpha = np.linalg.solve(K, y_train_scaled)
                return Kpred @ alpha
            except np.linalg.LinAlgError:
                return np.zeros(len(X_pred))

    # Fit GP with a few hyperparameter settings
    gp_models = []
    
    # Try a few reasonable hyperparameter combinations
    ls_candidates = [0.3, 0.5, 1.0, 2.0, 5.0]
    sf_candidates = [0.5, 1.0, 2.0]
    sn_candidates = [0.01, 0.05, 0.1]

    best_gp_score = np.inf
    best_gp_params = None

    # Evaluate on training data to pick best hyperparams (quick check)
    for ls in ls_candidates:
        for sf in sf_candidates:
            for sn in sn_candidates:
                if not time_check():
                    break
                try:
                    pred = gp_predict(Xs, Xs, ys, ls, sf, sn)
                    score = np.mean((pred - ys) ** 2)
                    if score < best_gp_score:
                        best_gp_score = score
                        best_gp_params = (ls, sf, sn)
                except Exception:
                    pass
            if not time_check():
                break
        if not time_check():
            break

    if best_gp_params is not None:
        ls, sf, sn = best_gp_params
        gp_models.append((ls, sf, sn))
    
    # Also keep a few diverse models for ensemble
    diverse_params = [(0.5, 1.0, 0.05), (1.0, 1.0, 0.1), (2.0, 1.5, 0.05)]
    for p in diverse_params:
        if p not in gp_models:
            gp_models.append(p)

    # Precompute GP predictions on training for weight calculation
    gp_train_preds = []
    for (ls, sf, sn) in gp_models:
        try:
            pred = gp_predict(Xs, Xs, ys, ls, sf, sn)
            gp_train_preds.append(pred)
        except Exception:
            gp_train_preds.append(np.zeros(n))

    # --- Component 2: Polynomial features (degree 2 and 3) ---
    def make_poly_features(X_scaled, degree):
        """Create polynomial features up to given degree."""
        cols = [np.ones(len(X_scaled))]
        # Linear terms
        for j in range(X_scaled.shape[1]):
            cols.append(X_scaled[:, j])
        # Quadratic terms
        if degree >= 2:
            for i in range(X_scaled.shape[1]):
                for j in range(i, X_scaled.shape[1]):
                    cols.append(X_scaled[:, i] * X_scaled[:, j])
        # Cubic terms
        if degree >= 3:
            for i in range(X_scaled.shape[1]):
                for j in range(i, X_scaled.shape[1]):
                    for k in range(j, X_scaled.shape[1]):
                        cols.append(X_scaled[:, i] * X_scaled[:, j] * X_scaled[:, k])
        return np.column_stack(cols)

    poly_models = []
    poly_train_preds = []

    for degree in [2, 3]:
        if not time_check():
            break
        try:
            F = make_poly_features(Xs, degree)
            # Use ridge regression for stability
            for alpha in [0.01, 0.1, 1.0]:
                if not time_check():
                    break
                w, *_ = np.linalg.lstsq(F + alpha * np.eye(F.shape[1]), ys, rcond=None)
                pred = F @ w
                poly_models.append((degree, alpha, w))
                poly_train_preds.append(pred)
        except Exception:
            pass

    # --- Component 3: Radial basis function (RBF) network ---
    def rbf_predict(X_pred, centers, weights, gamma):
        """Simple RBF network."""
        D = cdist(X_pred, centers)
        phi = np.exp(-gamma * D ** 2)
        return phi @ weights

    rbf_models = []
    rbf_train_preds = []

    # Use a subset of training points as centers
    n_centers = min(50, n)
    idx = np.random.choice(n, n_centers, replace=False)
    centers = Xs[idx]
    y_centers = ys[idx]

    for gamma in [0.5, 1.0, 2.0, 5.0]:
        if not time_check():
            break
        try:
            Kc = np.exp(-gamma * cdist(centers, centers) ** 2)
            Kc[np.diag_indices_from(Kc)] += 1e-6
            w_rbf = np.linalg.solve(Kc, y_centers)
            rbf_models.append((gamma, w_rbf))
            pred = rbf_predict(Xs, centers, w_rbf, gamma)
            rbf_train_preds.append(pred)
        except Exception:
            pass

    # --- Ensemble: combine all models with optimal weights ---
    all_train_preds = gp_train_preds + poly_train_preds + rbf_train_preds
    all_models = gp_models + poly_models + rbf_models

    if len(all_train_preds) == 0:
        # Fallback: simple mean
        y_pred_mean = y_mean
        return lambda X: np.full(len(np.asarray(X)), y_pred_mean)

    # Stack all predictions
    P = np.column_stack(all_train_preds)  # shape (n, n_models)

    # Find optimal ensemble weights via ridge regression
    best_ensemble_weights = np.ones(len(all_train_preds)) / len(all_train_preds)
    best_ensemble_score = np.inf

    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        if not time_check():
            break
        try:
            w_ens, *_ = np.linalg.lstsq(P + alpha * np.eye(P.shape[1]), ys, rcond=None)
            pred_ens = P @ w_ens
            score = np.mean((pred_ens - ys) ** 2)
            if score < best_ensemble_score:
                best_ensemble_score = score
                best_ensemble_weights = w_ens
        except Exception:
            pass

    # Also try simple average
    pred_avg = P.mean(axis=1)
    score_avg = np.mean((pred_avg - ys) ** 2)
    if score_avg < best_ensemble_score:
        best_ensemble_weights = np.ones(len(all_train_preds)) / len(all_train_preds)
        best_ensemble_score = score_avg

    # Final predict function
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        Xs_pred = (X - X_mean) / X_std
        
        preds = []
        
        # GP predictions
        for i, (ls, sf, sn) in enumerate(gp_models):
            try:
                pred = gp_predict(Xs_pred, Xs, ys, ls, sf, sn)
                preds.append(pred)
            except Exception:
                preds.append(np.zeros(len(Xs_pred)))
        
        # Polynomial predictions
        for j, (degree, alpha, w) in enumerate(poly_models):
            try:
                F = make_poly_features(Xs_pred, degree)
                pred = F @ w
                preds.append(pred)
            except Exception:
                preds.append(np.zeros(len(Xs_pred)))
        
        # RBF predictions
        for k, (gamma, w_rbf) in enumerate(rbf_models):
            try:
                pred = rbf_predict(Xs_pred, centers, w_rbf, gamma)
                preds.append(pred)
            except Exception:
                preds.append(np.zeros(len(Xs_pred)))
        
        if len(preds) == 0:
            return np.full(len(Xs_pred), y_mean)
        
        P_pred = np.column_stack(preds)
        pred_final = P_pred @ best_ensemble_weights
        
        # Rescale back
        pred_final = pred_final * y_std + y_mean
        
        # Ensure finite values
        pred_final = np.nan_to_num(pred_final, nan=y_mean, posinf=y_mean, neginf=y_mean)
        
        return pred_final

    return predict
# EVOLVE-BLOCK-END
