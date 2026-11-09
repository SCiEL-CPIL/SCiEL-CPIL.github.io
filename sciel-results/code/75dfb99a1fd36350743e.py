import numpy as np
import time
from sklearn.cluster import KMeans

def fit(X_train, y_train):
    """
    Fits a regression model using a hybrid basis of polynomials, trigonometric functions,
    and Radial Basis Functions (RBFs), solved via Ridge Regression.
    """
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs for better numerical conditioning of basis functions
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std

    def build_features(X_input):
        """
        Constructs the feature matrix for the hybrid basis.
        """
        N = X_input.shape[0]
        D = X_input.shape[1]
        
        # 1. Polynomial features (degree 1 to 3)
        # We generate all monomials of degree 1, 2, 3
        poly_feats = [np.ones((N, 1))]
        
        # Degree 1
        for i in range(D):
            poly_feats.append(X_input[:, i:i+1])
            
        # Degree 2
        for i in range(D):
            for j in range(i, D):
                if i == j:
                    poly_feats.append(X_input[:, i:i+1] ** 2)
                else:
                    poly_feats.append(X_input[:, i:i+1] * X_input[:, j:j+1])
                    
        # Degree 3
        for i in range(D):
            for j in range(i, D):
                for k in range(j, D):
                    if i == j == k:
                        poly_feats.append(X_input[:, i:i+1] ** 3)
                    elif i == j != k:
                        poly_feats.append(X_input[:, i:i+1] ** 2 * X_input[:, k:k+1])
                    elif i != j == k:
                        poly_feats.append(X_input[:, i:i+1] * X_input[:, j:j+1] ** 2)
                    else:
                        poly_feats.append(X_input[:, i:i+1] * X_input[:, j:j+1] * X_input[:, k:k+1])

        # 2. Trigonometric features
        # Frequencies: 1, 2, 4 (powers of 2)
        freqs = [1, 2, 4]
        trig_feats = []
        for i in range(D):
            x_col = X_input[:, i:i+1]
            for f in freqs:
                trig_feats.append(np.sin(f * x_col))
                trig_feats.append(np.cos(f * x_col))
                
        # Interactions for low dims (if D <= 2, add cross terms)
        if D == 2:
            x1, x2 = X_input[:, 0:1], X_input[:, 1:2]
            # Simple interactions
            trig_feats.append(np.sin(x1) * np.cos(x2))
            trig_feats.append(np.cos(x1) * np.sin(x2))
            trig_feats.append(np.sin(2*x1) * np.cos(2*x2))
            trig_feats.append(np.cos(2*x1) * np.sin(2*x2))

        # 3. Radial Basis Functions (RBFs)
        # Use 10 RBF centers
        n_rbf = 10
        # Select centers using KMeans on normalized data
        if time.time() < DEADLINE - 5:
            try:
                kmeans = KMeans(n_clusters=n_rbf, n_init=1, random_state=42)
                kmeans.fit(X_input)
                centers = kmeans.cluster_centers_
            except Exception:
                # Fallback: random sampling
                idx = np.random.choice(N, n_rbf, replace=False)
                centers = X_input[idx]
        else:
            idx = np.random.choice(N, n_rbf, replace=False)
            centers = X_input[idx]
        
        # Calculate distances to centers: (N, n_rbf)
        # dist_sq[i, j] = ||X_input[i] - centers[j]||^2
        dist_sq = np.sum((X_input[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        
        # Width parameter: Heuristic based on median distance between centers
        # or simply a fixed value relative to the normalized space.
        # Normalized space is roughly in [-2, 2]. Median distance ~ 1.0-1.5.
        # Let's use width such that exp(-0.5 * d^2 / l^2) decays reasonably.
        # l = 0.5 to 1.0 is typical. Let's fix l=0.7 (gamma = 1/(2*l^2) ~ 1)
        # Actually, standard RBF: exp(-gamma * dist_sq).
        # If dist_sq ~ 1, gamma ~ 1 gives exp(-1) ~ 0.37.
        # If dist_sq ~ 4, gamma ~ 1 gives exp(-4) ~ 0.018.
        gamma = 1.0 
        rbf_feats = [np.exp(-gamma * dist_sq)]  # Shape (N, n_rbf)

        # Combine all features
        features = np.hstack([np.concatenate(poly_feats, axis=1), 
                              np.concatenate(trig_feats, axis=1),
                              np.concatenate(rbf_feats, axis=1)])
        return features

    # Build training features
    Phi_train = build_features(X_norm)
    
    # Solve Ridge Regression: (Phi^T Phi + lambda I) beta = Phi^T y
    # We use a small lambda for stability, as data is noiseless, we want low bias.
    # If overfitting occurs, we might need to tune lambda, but 200 points / ~200 features is balanced.
    lambda_reg = 1e-6
    
    try:
        # Efficient solve
        A = Phi_train.T @ Phi_train
        A[np.diag_indices_from(A)] += lambda_reg
        b = Phi_train.T @ y
        
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        beta, _, _, _ = np.linalg.lstsq(Phi_train, y, rcond=None)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_mean) / X_std
        
        # Ensure we don't exceed deadline (unlikely for prediction, but safe)
        if time.time() > DEADLINE:
            # Return mean if timeout (should not happen)
            return np.full(X_test.shape[0], np.mean(y_train))
            
        Phi_test = build_features(X_test_norm)
        return Phi_test @ beta

    return predict
