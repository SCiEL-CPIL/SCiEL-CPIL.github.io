"""Symbolic regression via Local Polynomial Regression (LOESS)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1] to make distance metrics comparable across dimensions
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability in local fits
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Determine polynomial degree based on dimensionality
    # Higher dimensions require lower degrees to avoid overfitting in local windows
    if d == 1:
        degree = 5
    elif d == 2:
        degree = 4
    else:
        degree = 3
        
    # Precompute basis functions for training points
    # Basis: 1, x, x^2, ..., x^degree, x_i*x_j, ...
    # To keep matrix size manageable, we use a truncated basis.
    # For d=1, degree=5: 6 terms
    # For d=2, degree=4: 1 + 2 + (2+1) + (2+1+1) + (2+1+1+1) = 1 + 2 + 3 + 4 + 5 = 15 terms?
    # Actually, let's construct the basis explicitly.
    
    def build_basis(X_in):
        """Construct polynomial basis features for X_in."""
        X_in = np.asarray(X_in)
        n_pts = X_in.shape[0]
        
        # Start with constant term
        features = [np.ones(n_pts)]
        
        # Linear terms
        for j in range(d):
            features.append(X_in[:, j])
            
        # Higher order terms up to 'degree'
        # We will generate all monomials of degree 2 up to 'degree'
        # To avoid explosion, we limit the total number of features.
        
        # Generate indices for monomials
        monomials = []
        
        # Helper to get monomial values
        def get_monomial_values(x, powers):
            val = np.ones(x.shape[0])
            for j in range(d):
                if powers[j] > 0:
                    val = val * (x[:, j] ** powers[j])
            return val
            
        # We'll iterate through possible power combinations
        # This is a bit slow if done naively, but n=200 is small.
        
        # Instead of generating all combinations which can be huge,
        # let's use a simpler strategy:
        # If d=1: [1, x, x^2, x^3, x^4, x^5]
        # If d=2: [1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3, ...]
        
        # Let's just use a fixed set of features that covers common interactions.
        # For d=1: up to x^6
        # For d=2: up to degree 4 (full polynomial)
        # For d=3: up to degree 3 (full polynomial) -> 20 terms.
        
        if d == 1:
            for p in range(2, degree + 1):
                features.append(X_in[:, 0] ** p)
        elif d == 2:
            # Full polynomial up to 'degree'
            # Degree 2: x1^2, x1x2, x2^2
            # Degree 3: x1^3, x1^2x2, x1x2^2, x2^3
            # Degree 4: x1^4, x1^3x2, x1^2x2^2, x1x2^3, x2^4
            for p1 in range(2, degree + 1):
                for p2 in range(0, degree - p1 + 1):
                    if p1 + p2 >= 2:
                        features.append((X_in[:, 0] ** p1) * (X_in[:, 1] ** p2))
        elif d == 3:
            # Full polynomial up to degree 3
            # Total terms: C(3+3, 3) = 20.
            # We already have 1, x1, x2, x3 (4 terms).
            # Need terms of degree 2 and 3.
            for p1 in range(0, degree + 1):
                for p2 in range(0, degree - p1 + 1):
                    for p3 in range(0, degree - p1 - p2 + 1):
                        if p1 + p2 + p3 >= 2:
                            features.append((X_in[:, 0] ** p1) * (X_in[:, 1] ** p2) * (X_in[:, 1+1] ** p3) if False else 
                                            (X_in[:, 0] ** p1) * (X_in[:, 1] ** p2) * (X_in[:, 2] ** p3))
        
        return np.column_stack(features)

    # Precompute basis for training set
    B_train = build_basis(X_norm)
    n_feat = B_train.shape[1]
    
    # Precompute pairwise distances in normalized space
    # D2[i, j] = ||x_i - x_j||^2
    diff = X_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
    D2 = np.sum(diff * diff, axis=-1)
    
    # Determine local window size (fraction of data)
    # Smaller windows allow higher effective degrees of freedom locally
    # But too small leads to noise sensitivity.
    # With 200 points, a window of 20-30 points is reasonable.
    # Let's try a fixed number of neighbors for stability.
    k_neighbors = 25 if d <= 2 else 35
    
    # Sort distances for each point to find k-th nearest neighbor
    # This allows us to set a local bandwidth for each prediction point
    # based on the k-th nearest neighbor distance.
    
    # We will use a weighted least squares approach.
    # Weights: Gaussian kernel based on distance, or uniform within k-nearest.
    # Uniform within k-nearest is robust to distance scale issues if k is fixed.
    
    # Precompute the k-th nearest neighbor distance for each training point
    # This defines the local scale.
    # For prediction, we find the k-th nearest neighbor in the training set for the test point.
    
    # To speed up prediction, we can precompute the inverse of (B^T W B) for each local window?
    # No, W changes per point.
    # However, we can use the "local fit" directly.
    
    # Alternative: Use a fixed global bandwidth sigma?
    # Since inputs are normalized to [0,1], a fixed sigma might work.
    # But LOESS with k-nearest is more adaptive.
    
    # Let's implement prediction using k-nearest neighbor weighting.
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        n_test = X_test_norm.shape[0]
        
        # Compute distances between test and train
        diff_test = X_test_norm[:, np.newaxis, :] - X_norm[np.newaxis, :, :]
        D2_test = np.sum(diff_test * diff_test, axis=-1)
        
        predictions = np.zeros(n_test)
        
        for i in range(n_test):
            # Check time
            if time.time() > DEADLINE:
                # Fallback: return mean
                return np.full(n_test, y_mean)
                
            dists = np.sqrt(D2_test[i, :])
            
            # Find k-nearest neighbors
            # Use argsort for small n
            idx_sorted = np.argsort(dists)
            k_idx = idx_sorted[:k_neighbors]
            
            # Get local data
            X_local = X_norm[k_idx, :]
            y_local = y_norm[k_idx]
            
            # Build local basis
            B_local = build_basis(X_local)
            
            # Weights: Inverse distance squared? Or uniform?
            # Uniform is simpler and robust.
            # Let's use uniform weights for the k-nearest neighbors.
            # This is equivalent to fitting a polynomial to the k nearest points.
            
            # Solve least squares: B_local w = y_local
            # Use lstsq for stability
            try:
                # Regularize slightly to handle collinearity in high-degree polynomials
                # Ridge regression: (B^T B + lambda I) w = B^T y
                # Lambda small
                lam = 1e-6
                A = B_local.T @ B_local + lam * np.eye(n_feat)
                b_vec = B_local.T @ y_local
                w = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                # Fallback to global mean
                predictions[i] = 0.0
                continue
            
            # Predict at x_test
            x_test_local = X_test_norm[i, :].reshape(1, -1)
            b_test = build_basis(x_test_local)
            
            pred_norm = b_test @ w
            predictions[i] = pred_norm[0] * y_std + y_mean
            
        return predictions

    return predict
# EVOLVE-BLOCK-END
