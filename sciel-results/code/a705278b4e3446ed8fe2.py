"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations_with_replacement

def fit(X_train, y_train):
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize inputs to [0, 1] to stabilize polynomial terms
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range

    Xn_train = normalize(X_train)
    
    def poly_features(X, degree):
        """Generate polynomial features of given degree."""
        if degree == 0:
            return np.ones((X.shape[0], 1))
        
        # Generate all monomials of degree <= 'degree'
        # We include the bias term (degree 0) in the feature matrix
        # Number of features for degree D is C(d+D, D)
        # We will generate columns for all combinations
        
        cols = [np.ones((X.shape[0], 1))]
        
        for deg in range(1, degree + 1):
            # Get all combinations of indices with replacement of length 'deg'
            # e.g. for d=2, deg=2: (0,0), (0,1), (1,1)
            for comb in combinations_with_replacement(range(d), deg):
                # Create the feature vector for this monomial
                # e.g. comb=(0,1) -> X[:,0] * X[:,1]
                feat = np.ones((X.shape[0], 1))
                for idx in comb:
                    feat = feat * X[:, idx].reshape(-1, 1)
                cols.append(feat)
                
        return np.hstack(cols)

    # Hyperparameter search
    degrees = [1, 2, 3, 4, 5]
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_degree = 1
    best_alpha = 1.0
    best_cv_mse = np.inf
    
    # Pre-compute features for all degrees to save time? 
    # No, we need sub-sets for CV. But feature generation is fast for small n.
    
    for deg in degrees:
        # Generate full feature matrix for this degree
        Phi_full = poly_features(Xn_train, deg)
        
        for alpha in alphas:
            cv_mses = []
            for k in range(n_folds):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
                
                Phi_tr = Phi_full[train_idx]
                Phi_val = Phi_full[val_idx]
                y_tr = y_train[train_idx]
                y_val = y_train[val_idx]
                
                # Solve Ridge Regression: (Phi^T Phi + alpha I) beta = Phi^T y
                # Add alpha to diagonal, but not for bias term? 
                # Usually we don't regularize the bias. 
                # Let's assume the first column is bias (degree 0 term).
                # We will subtract mean from y and center X? 
                # Simpler: Just add alpha to all diagonals. With normalized X, it's okay.
                # Or better: standardize y? No, just solve.
                
                try:
                    # Construct system
                    A = Phi_tr.T @ Phi_tr + alpha * np.eye(Phi_tr.shape[1])
                    b = Phi_tr.T @ y_tr
                    beta = np.linalg.solve(A, b)
                    pred = Phi_val @ beta
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(Phi_tr.T @ Phi_tr + alpha * np.eye(Phi_tr.shape[1]), Phi_tr.T @ y_tr, rcond=None)[0]
                    pred = Phi_val @ beta
                    
                mse = np.mean((pred - y_val) ** 2)
                cv_mses.append(mse)
            
            mean_cv_mse = np.mean(cv_mses)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_degree = deg
                best_alpha = alpha

    # Final fit
    Phi_full = poly_features(Xn_train, best_degree)
    try:
        A = Phi_full.T @ Phi_full + best_alpha * np.eye(Phi_full.shape[1])
        b = Phi_full.T @ y_train
        beta_final = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta_final = np.linalg.lstsq(Phi_full.T @ Phi_full + best_alpha * np.eye(Phi_full.shape[1]), Phi_full.T @ y_train, rcond=None)[0]

    def predict(X):
        Xn = normalize(X)
        Phi_test = poly_features(Xn, best_degree)
        return Phi_test @ beta_final

    return predict
# EVOLVE-BLOCK-END
