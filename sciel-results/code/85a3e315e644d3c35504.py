"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Pre-compute non-linear base features for efficiency
    def get_base_nonlinear(X):
        """
        Returns a list of non-linear feature arrays for each input dimension.
        Each input gets: log, sqrt, inv, x*log(x), log^2, sqrt^3, sin, cos, exp(-x/10)
        """
        base_features = []
        for j in range(d):
            xj = X[:, j]
            # Safe operations
            xj_safe = np.clip(xj, 1e-10, None)
            feat_j = [
                np.log(xj_safe),
                np.sqrt(xj_safe),
                1.0 / xj_safe,
                xj * np.log(xj_safe),
                np.log(xj_safe) ** 2,
                np.sqrt(xj_safe) ** 3,
                np.sin(xj),
                np.cos(xj),
                np.exp(-xj / 10.0)
            ]
            base_features.append(feat_j)
        return base_features

    def build_features(X, max_deg, use_nonlinear=True):
        """
        Builds a feature matrix combining polynomials up to max_deg and optional non-linear terms.
        """
        cols = [np.ones(len(X))]
        
        # 1. Polynomial terms
        for degree in range(1, max_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        
        # 2. Non-linear terms
        if use_nonlinear:
            base_feats = get_base_nonlinear(X)
            # Individual non-linear terms
            for j in range(d):
                for feat in base_feats[j]:
                    cols.append(feat)
            
            # Pairwise products of non-linear terms (to capture interactions)
            # Limit to first 3 non-linear types per dimension to avoid explosion
            key_feats = []
            for j in range(d):
                # log, sqrt, inv are the most common
                key_feats.append([base_feats[j][0], base_feats[j][1], base_feats[j][2]])
                
            for i in range(d):
                for j in range(i + 1, d):
                    for fi in range(3):
                        for fj in range(3):
                            cols.append(key_feats[i][fi] * key_feats[j][fj])
                            
            # Triple products for d=3 (if applicable and time permits)
            if d == 3:
                cols.append(key_feats[0][0] * key_feats[1][0] * key_feats[2][0]) # log(x)*log(y)*log(z)
                cols.append(key_feats[0][1] * key_feats[1][1] * key_feats[2][1]) # sqrt(x)*sqrt(y)*sqrt(z)

        return np.column_stack(cols)

    def ridge_fit(X, y, max_deg, use_nonlinear, lam=1e-6):
        F = build_features(X, max_deg, use_nonlinear)
        n_features = F.shape[1]
        try:
            A = F.T @ F + lam * np.eye(n_features)
            b = F.T @ y
            w = np.linalg.solve(A, b)
        except:
            try:
                w, *_ = np.linalg.lstsq(F, y, rcond=None)
            except:
                w = np.zeros(n_features)
        
        def predict(X_pred):
            F_pred = build_features(X_pred, max_deg, use_nonlinear)
            return F_pred @ w
        return predict

    def cross_validate(X, y, max_deg, use_nonlinear, lam=0.0, n_folds=5):
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        errors = []
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]
            
            F_train = build_features(X_train_cv, max_deg, use_nonlinear)
            n_features = F_train.shape[1]
            
            try:
                if lam == 0.0:
                    w, *_ = np.linalg.lstsq(F_train, y_train_cv, rcond=None)
                else:
                    A = F_train.T @ F_train + lam * np.eye(n_features)
                    b = F_train.T @ y_train_cv
                    w = np.linalg.solve(A, b)
            except:
                continue
            
            F_val = build_features(X_val_cv, max_deg, use_nonlinear)
            pred = F_val @ w
            mse = np.mean((pred - y_val_cv) ** 2)
            errors.append(mse)
        
        if len(errors) == 0:
            return np.inf
        return np.mean(errors)

    best_predictor = None
    best_cv_score = np.inf
    
    # Search Strategy:
    # 1. Try polynomial-only models with degrees 3, 4, 5, 6
    # 2. Try extended (non-linear) models with degrees 2, 3, 4
    # Regularization is crucial for higher degrees.
    
    search_space = [
        # (max_deg, use_nonlinear, [lambdas])
        (3, False, [0.0, 1e-6, 1e-4, 1e-2]),
        (4, False, [0.0, 1e-6, 1e-4, 1e-2, 1e-1]),
        (5, False, [1e-4, 1e-2, 1e-1, 1.0]),
        (6, False, [1e-2, 1e-1, 1.0, 10.0]),
        
        (2, True, [0.0, 1e-6, 1e-4, 1e-2]),
        (3, True, [0.0, 1e-6, 1e-4, 1e-2]),
        (4, True, [1e-6, 1e-4, 1e-2, 1e-1]),
    ]
    
    for max_deg, use_nl, lams in search_space:
        for lam in lams:
            if time.time() > deadline - 20:
                break
            try:
                cv_score = cross_validate(X, y, max_deg, use_nl, lam)
                if cv_score < best_cv_score:
                    best_cv_score = cv_score
                    best_predictor = ridge_fit(X, y, max_deg, use_nl, lam)
            except Exception:
                pass

    # Fallback if nothing worked
    if best_predictor is None:
        best_predictor = ridge_fit(X, y, 3, False, 1e-6)
    
    return best_predictor
# EVOLVE-BLOCK-END
