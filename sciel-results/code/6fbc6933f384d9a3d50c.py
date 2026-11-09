import numpy as np

def fit(X_train, y_train):
        X = X_train[:, 0]
        y = y_train
        n = len(X)
        
        # Normalize X to [0, 1]
        x_min = np.min(X)
        x_max = np.max(X)
        if x_max == x_min:
            X_norm = np.zeros_like(X)
        else:
            X_norm = (X - x_min) / (x_max - x_min)
            
        # Estimate spacing
        x_sorted = np.sort(X_norm)
        diffs = np.diff(x_sorted)
        diffs = diffs[diffs > 1e-9]
        if len(diffs) > 0:
            median_spacing = np.median(diffs)
        else:
            median_spacing = 1.0 / n
            
        # Target gamma: width ~ 2 * spacing
        # exp(-gamma * (2*spacing)^2) ~ 0.1 -> gamma ~ 2.3 / (4 * spacing^2) ~ 0.5 / spacing^2
        base_gamma = 1.0 / (median_spacing ** 2)
        
        gammas = [base_gamma * f for f in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]]
        alphas = [1e-10, 1e-8, 1e-6, 1e-4]
        
        best_w = None
        best_gamma = None
        best_alpha = None
        best_score = np.inf
        
        for gamma in gammas:
            # Compute Kernel Matrix
            # Use broadcasting: (N, 1) - (1, N)
            diff = X_norm[:, np.newaxis] - X_norm[np.newaxis, :]
            K = np.exp(-gamma * (diff ** 2))
            
            for alpha in alphas:
                # Regularize
                K_reg = K + alpha * np.eye(n)
                try:
                    w = np.linalg.solve(K_reg, y)
                except np.linalg.LinAlgError:
                    continue
                    
                # Check for finite
                if not np.all(np.isfinite(w)):
                    continue
                    
                # Training Score
                y_pred = K @ w
                score = np.mean((y_pred - y) ** 2)
                
                # Prefer lower score. If scores are very close (noiseless), 
                # we might prefer the one with smaller alpha (less smoothing) 
                # or larger gamma (more local)?
                # Actually, for noiseless, we want exact interpolation.
                # Smaller alpha -> closer to interpolation.
                # So we prefer smaller alpha if score is negligible.
                
                if score < best_score:
                    best_score = score
                    best_w = w
                    best_gamma = gamma
                    best_alpha = alpha
        
        if best_w is None:
            # Fallback to linear regression
            X_lin = np.column_stack([np.ones(n), X])
            w_lin, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
            def predict(X_test):
                X_test = X_test[:, 0]
                X_test_norm = (X_test - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(X_test)
                X_test_lin = np.column_stack([np.ones(len(X_test)), X_test])
                return X_test_lin @ w_lin
            return predict

        def predict(X_test):
            X_test = X_test[:, 0]
            X_test_norm = (X_test - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(X_test)
            
            diff = X_test_norm[:, np.newaxis] - X_norm[np.newaxis, :]
            K_test = np.exp(-best_gamma * (diff ** 2))
            
            y_pred = K_test @ best_w
            return y_pred

        return predict
