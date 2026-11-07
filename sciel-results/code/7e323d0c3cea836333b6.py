"""
Strategy: Ensemble of Chebyshev Polynomial Regressors.
Uses a library of Chebyshev polynomials of different degrees (up to 30) as basis functions.
Chebyshev polynomials are well-conditioned for polynomial approximation on a bounded interval.
We train multiple models with different degrees and select the top 3 based on 5-fold cross-validation.
The final prediction is the weighted average of these top models.
This approach handles smooth functions well (low degrees) and can capture oscillatory behavior (high degrees) 
without the risk of overfitting noise (since data is noiseless, overfitting is less of a concern than underfitting, 
but we still use CV to prevent wild oscillations in high degrees if the function is smooth).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fit an ensemble of Chebyshev polynomial regressors.
    Returns a predict function.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin
    
    # Ensure X is 2D
    X = np.atleast_2d(X_train)
    if X.shape[0] == 1:
        X = X.T
    y = np.asarray(y_train)
    
    n, d = X.shape
    
    # Standardize X to [-1, 1] for Chebyshev polynomials
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    
    X_norm = (X - X_min) / X_range
    X_norm = 2 * X_norm - 1  # Map to [-1, 1]
    
    # Precompute Chebyshev basis for various degrees
    # We will try degrees from 1 to 30
    max_degree = 30
    
    # Store models: list of (degree, coefficients)
    models = []
    
    # Cross-validation setup
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
    
    cv_scores = []
    
    for degree in range(1, max_degree + 1):
        if time.time() > deadline - 10:
            break
            
        # Build Chebyshev design matrix for this degree
        # For 1D input, we use T_k(x). For d>1, we can use tensor product or just 1D since all tasks are 1D.
        # The task description says "1 input(s)" for all 4 equations. So d=1.
        # However, to be safe, if d > 1, we might need to handle it differently. 
        # Given the contract, d=1 is guaranteed.
        
        # Compute Chebyshev polynomials T_k(x) for x in X_norm
        # T_0(x) = 1
        # T_1(x) = x
        # T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
        
        T = np.zeros((n, degree + 1))
        T[:, 0] = 1.0
        if degree >= 1:
            T[:, 1] = X_norm[:, 0]
            for k in range(2, degree + 1):
                T[:, k] = 2 * X_norm[:, 0] * T[:, k-1] - T[:, k-2]
        
        # Perform Cross-Validation
        cv_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_train_cv = T[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = T[val_idx]
            y_val_cv = y[val_idx]
            
            # Solve least squares
            try:
                # Add small regularization for stability if degree is high
                reg = 1e-6 * np.eye(degree + 1)
                A = X_train_cv.T @ X_train_cv + reg
                b = X_train_cv.T @ y_train_cv
                w_cv = np.linalg.solve(A, b)
                
                y_pred_cv = X_val_cv @ w_cv
                mse = np.mean((y_pred_cv - y_val_cv) ** 2)
                cv_errors.append(mse)
            except np.linalg.LinAlgError:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        cv_scores.append((mean_cv_error, degree))
        
        # Fit on all data for the candidate model
        try:
            reg = 1e-6 * np.eye(degree + 1)
            A = T.T @ T + reg
            b = T.T @ y
            w_all = np.linalg.solve(A, b)
            models.append((degree, w_all, T)) # Store T for potential reuse? No, T depends on X.
            # We just need w_all and degree to reconstruct T for prediction.
        except np.linalg.LinAlgError:
            continue
    
    if not models:
        # Fallback: simple linear regression
        X_aug = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict_fallback(X_pred):
            X_p = np.atleast_2d(X_pred)
            if X_p.shape[0] == 1:
                X_p = X_p.T
            X_p_norm = (X_p - X_min) / X_range
            X_p_norm = 2 * X_p_norm - 1
            X_p_aug = np.column_stack([np.ones(len(X_p)), X_p_norm])
            return X_p_aug @ w
        return predict_fallback
    
    # Select top 3 models by CV error
    cv_scores.sort(key=lambda x: x[0])
    top_models = cv_scores[:3]
    
    # We need to re-fit or retrieve the weights for these top models.
    # Since we didn't store the weights for all models in a lookup, let's just re-fit the top ones.
    # Actually, we can store the weights in a dict during the loop.
    
    # Let's re-do the loop but store weights.
    # Optimization: The loop above was expensive. Let's assume the top degrees are small enough 
    # or we can just pick the best one if time is tight. 
    # Given 300s, fitting 30 models of size 200x31 is very fast (<1s).
    # The CV loop is 30 * 5 = 150 fits. Still fast.
    
    # Let's collect the weights for the top degrees.
    top_weights = []
    top_degrees = [deg for _, deg in top_models]
    
    # We need to recompute T for these degrees and solve.
    # To save time, we can just re-run the fit for the top degrees.
    for deg in top_degrees:
        T = np.zeros((n, deg + 1))
        T[:, 0] = 1.0
        if deg >= 1:
            T[:, 1] = X_norm[:, 0]
            for k in range(2, deg + 1):
                T[:, k] = 2 * X_norm[:, 0] * T[:, k-1] - T[:, k-2]
        
        reg = 1e-6 * np.eye(deg + 1)
        A = T.T @ T + reg
        b = T.T @ y
        w_all = np.linalg.solve(A, b)
        top_weights.append(w_all)
    
    # If we have fewer than 3 valid models, adjust
    if len(top_weights) < 3:
        # Pad with the best one
        while len(top_weights) < 3:
            top_weights.append(top_weights[0])
            
    def predict(X_pred):
        X_p = np.atleast_2d(X_pred)
        if X_p.shape[0] == 1:
            X_p = X_p.T
            
        X_p_norm = (X_p - X_min) / X_range
        X_p_norm = 2 * X_p_norm - 1
        
        preds = np.zeros(len(X_p))
        
        for i, w in enumerate(top_weights):
            deg = top_degrees[i]
            T_p = np.zeros((len(X_p), deg + 1))
            T_p[:, 0] = 1.0
            if deg >= 1:
                T_p[:, 1] = X_p_norm[:, 0]
                for k in range(2, deg + 1):
                    T_p[:, k] = 2 * X_p_norm[:, 0] * T_p[:, k-1] - T_p[:, k-2]
            
            preds += T_p @ w
        
        preds /= len(top_weights)
        
        return preds

    return predict
# EVOLVE-BLOCK-END
