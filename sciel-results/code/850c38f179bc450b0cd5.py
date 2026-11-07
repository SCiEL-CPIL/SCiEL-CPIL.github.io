import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s buffer for evaluation overhead
    
    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape
    
    # Generate Basis
    # We will store features in a list, then stack them
    features = [np.ones(n)]
    feature_names = ['1']
    
    # Univariate features
    for j in range(d):
        x = X[:, j]
        feats = [
            x,
            x**2,
            x**3,
            np.sin(x),
            np.cos(x),
            np.exp(x),
            np.log(x),
            np.tanh(x)
        ]
        names = [f'x{j}', f'x{j}^2', f'x{j}^3', f'sin(x{j})', f'cos(x{j})', f'exp(x{j})', f'log(x{j})', f'tanh(x{j})']
        features.extend(feats)
        feature_names.extend(names)
        
    # Bivariate interaction features
    if d > 1:
        for i in range(d):
            for j in range(i+1, d):
                xi = X[:, i]
                xj = X[:, j]
                feats = [
                    xi * xj,
                    np.sin(xi + xj),
                    np.cos(xi + xj),
                    np.tanh(xi + xj)
                ]
                names = [f'x{i}*x{j}', f'sin(x{i}+x{j})', f'cos(x{i}+x{j})', f'tanh(x{i}+x{j})']
                features.extend(feats)
                feature_names.extend(names)
                
    # Trivariate interaction features (if d >= 3)
    if d > 2:
        # Just simple product of all 3? Or sum?
        # Let's add product of all 3
        prod = X[:, 0] * X[:, 1] * X[:, 2]
        features.append(prod)
        feature_names.append('x0*x1*x2')
        
        # Sum interactions
        s = X[:, 0] + X[:, 1] + X[:, 2]
        features.append(np.sin(s))
        features.append(np.cos(s))
        feature_names.append('sin(sum)')
        feature_names.append('cos(sum)')

    B = np.column_stack(features)
    # Replace inf/nan with 0 to be safe
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    
    num_features = B.shape[1]
    
    # Forward Selection
    selected_indices = [0] # Always start with intercept
    remaining_indices = list(range(1, num_features))
    
    # Precompute B.T @ B and B.T @ y for speed? 
    # Since we add one feature at a time, we can update residuals.
    # But with small N=200 and M~40, direct lstsq is fast enough.
    
    # To speed up forward selection, we can use the fact that we are adding one column.
    # However, simple approach: iterate through remaining features, add to selected, calculate RSS.
    # RSS calculation: np.sum((y - B_sel @ w)**2)
    
    # Optimization: Use QR decomposition or normal equations update.
    # Given the small size, let's just do a simple loop but with a time check.
    
    best_rss = np.sum((y - B[:, [0]] @ np.linalg.lstsq(B[:, [0]], y, rcond=None)[0])**2)
    
    max_steps = 15 # Don't select more than 15 features to avoid overfitting
    step = 0
    
    while remaining_indices and step < max_steps:
        if time.time() > deadline - 5.0:
            break
            
        best_idx = -1
        best_new_rss = best_rss
        
        for idx in remaining_indices:
            # Add feature idx
            new_indices = selected_indices + [idx]
            B_sel = B[:, new_indices]
            
            # Solve least squares
            try:
                w, *_ = np.linalg.lstsq(B_sel, y, rcond=None)
                pred = B_sel @ w
                rss = np.sum((y - pred)**2)
                
                if rss < best_new_rss:
                    best_new_rss = rss
                    best_idx = idx
            except np.linalg.LinAlgError:
                continue
                
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        best_rss = best_new_rss
        step += 1
        
    # Final Fit with Ridge Regression for stability
    B_sel = B[:, selected_indices]
    
    # Add small ridge lambda
    lambda_val = 1e-4
    I = np.eye(B_sel.shape[1])
    w = np.linalg.solve(B_sel.T @ B_sel + lambda_val * I, B_sel.T @ y)
    
    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        n_test = X_test.shape[0]
        
        # Reconstruct basis for test set
        # This is expensive to do inside predict if called many times, 
        # but the interface implies predict is called on the test set (200 points).
        # We can cache the basis generation logic or just recompute.
        # Recomputing is safer and fast enough for 200 points.
        
        feats = [np.ones(n_test)]
        
        for j in range(d):
            x = X_test[:, j]
            feats.extend([
                x,
                x**2,
                x**3,
                np.sin(x),
                np.cos(x),
                np.exp(x),
                np.log(x),
                np.tanh(x)
            ])
            
        if d > 1:
            for i in range(d):
                for j in range(i+1, d):
                    xi = X_test[:, i]
                    xj = X_test[:, j]
                    feats.extend([
                        xi * xj,
                        np.sin(xi + xj),
                        np.cos(xi + xj),
                        np.tanh(xi + xj)
                    ])
                    
        if d > 2:
            prod = X_test[:, 0] * X_test[:, 1] * X_test[:, 2]
            feats.append(prod)
            s = X_test[:, 0] + X_test[:, 1] + X_test[:, 2]
            feats.append(np.sin(s))
            feats.append(np.cos(s))

        B_test = np.column_stack(feats)
        B_test = np.nan_to_num(B_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Select columns
        B_test_sel = B_test[:, selected_indices]
        
        return B_test_sel @ w

    return predict
