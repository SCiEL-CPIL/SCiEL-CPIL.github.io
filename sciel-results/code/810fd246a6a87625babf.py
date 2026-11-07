# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 280  # safety margin before 330s kill

    def time_left():
        return deadline - time.time()

    n, d = X_train.shape
    y = np.asarray(y_train, dtype=np.float64).ravel()

    # Global stats
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0

    # y stats
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0

    # Build a rich feature set. This is the core of the symbolic regression approach.
    # We build features based on the raw inputs, since the hidden equations likely
    # operate on the raw values (e.g., sin(x), x^2, etc.).
    def build_features(X_raw):
        Xr = np.asarray(X_raw, dtype=np.float64)
        nr = Xr.shape[0]
        dr = Xr.shape[1]
        
        feats = [np.ones(nr)]
        
        # 1. Polynomial terms (up to degree 4 for d<=2, degree 3 for d>=3)
        # This captures x^2, x^3, x_i*x_j, x_i^2*x_j, etc.
        max_deg = 4 if dr <= 2 else 3
        for deg in range(1, max_deg + 1):
            # Generate all monomials of degree 'deg'
            # Using recursion or itertools.product, but we do it manually for clarity and control
            if deg == 1:
                for j in range(dr):
                    feats.append(Xr[:, j])
            elif deg == 2:
                for i in range(dr):
                    for j in range(i, dr):
                        feats.append(Xr[:, i] * Xr[:, j])
            elif deg == 3:
                for i in range(dr):
                    for j in range(i, dr):
                        for k in range(j, dr):
                            feats.append(Xr[:, i] * Xr[:, j] * Xr[:, k])
            elif deg == 4:
                for i in range(dr):
                    for j in range(i, dr):
                        for k in range(j, dr):
                            for m in range(k, dr):
                                feats.append(Xr[:, i] * Xr[:, j] * Xr[:, k] * Xr[:, m])

        # 2. Trigonometric terms
        # sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
        for j in range(dr):
            xj = Xr[:, j]
            feats.append(np.sin(xj))
            feats.append(np.cos(xj))
            feats.append(np.sin(2 * xj))
            feats.append(np.cos(2 * xj))
            # Higher harmonics if time permits or if d is small
            if dr <= 2:
                feats.append(np.sin(3 * xj))
                feats.append(np.cos(3 * xj))

        # 3. Exponential and Logarithmic terms
        # These are important for non-polynomial shapes
        for j in range(dr):
            xj = Xr[:, j]
            # Safety checks for log/exp
            if np.min(xj) > 0:
                feats.append(np.log(xj))
                feats.append(np.log(xj) ** 2)
            # exp can blow up, so we only include it if values are bounded
            if np.max(np.abs(xj)) < 15:
                feats.append(np.exp(xj))
                feats.append(np.exp(-xj))
            # Sqrt
            if np.min(xj) >= 0:
                feats.append(np.sqrt(xj))
            # Tanh (bounded, smooth)
            feats.append(np.tanh(xj))
            feats.append(np.tanh(2 * xj))

        # 4. Interaction terms with Trig/Exp
        if dr >= 2:
            for i in range(dr):
                for j in range(i + 1, dr):
                    xi = Xr[:, i]
                    xj = Xr[:, j]
                    # Linear interactions with trig
                    feats.append(xi * np.sin(xj))
                    feats.append(xi * np.cos(xj))
                    feats.append(xj * np.sin(xi))
                    feats.append(xj * np.cos(xi))
                    
                    # Trig interactions
                    feats.append(np.sin(xi) * np.sin(xj))
                    feats.append(np.cos(xi) * np.cos(xj))
                    feats.append(np.sin(xi) * np.cos(xj))
                    
                    # Sum/Diff trig
                    feats.append(np.sin(xi + xj))
                    feats.append(np.cos(xi + xj))
                    feats.append(np.sin(xi - xj))
                    feats.append(np.cos(xi - xj))
                    
                    # Product of inputs with trig
                    feats.append(xi * xj * np.sin(xi))
                    feats.append(xi * xj * np.cos(xj))
                    
                    # Exp interactions (if safe)
                    if np.max(np.abs(xi)) < 15 and np.max(np.abs(xj)) < 15:
                        feats.append(np.exp(xi) * xj)
                        feats.append(xi * np.exp(xj))
                        feats.append(np.exp(xi + xj))
                        feats.append(np.exp(xi - xj))

        # 5. Global features
        if dr > 1:
            feats.append(np.sum(Xr, axis=1))
            feats.append(np.prod(Xr, axis=1))
            feats.append(np.sum(Xr ** 2, axis=1))
            feats.append(np.linalg.norm(Xr, axis=1))
            feats.append(np.linalg.norm(Xr, axis=1) ** 2)
            feats.append(np.min(Xr, axis=1))
            feats.append(np.max(Xr, axis=1))
            feats.append(np.sum(np.sin(Xr), axis=1))
            feats.append(np.sum(np.cos(Xr), axis=1))

        # 6. Rational terms (1/x) if safe
        for j in range(dr):
            xj = Xr[:, j]
            if np.min(np.abs(xj)) > 0.1:
                feats.append(1.0 / xj)

        F = np.column_stack(feats)
        return F

    # Build features for training
    F_train = build_features(X_train)
    p = F_train.shape[1]

    # Handle potential NaNs/Infs from feature construction (e.g., log(neg), exp(huge))
    mask_finite = np.all(np.isfinite(F_train), axis=1)
    if not np.all(mask_finite):
        # This should rarely happen if we have safety checks, but just in case
        F_train = F_train[mask_finite]
        y_train_clean = y[mask_finite]
        n = len(y_train_clean)
    else:
        y_train_clean = y

    # Center and scale features for numerical stability
    F_mean = F_train.mean(axis=0)
    F_std = F_train.std(axis=0)
    F_std[F_std < 1e-12] = 1.0
    F_centered = (F_train - F_mean) / F_std

    y_centered = (y_train_clean - y_mean) / y_std

    best_mse = np.inf
    best_predictor = None

    # Strategy 1: Ridge Regression with various alphas
    # This is robust to collinearity in the rich feature set
    alphas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for alpha in alphas:
        if time_left() < 10:
            break
        try:
            if alpha == 0.0:
                w, *_ = np.linalg.lstsq(F_centered, y_centered, rcond=None)
            else:
                A = F_centered.T @ F_centered + alpha * np.eye(p)
                b = F_centered.T @ y_centered
                w = np.linalg.solve(A, b)
            
            pred_c = F_centered @ w
            pred = pred_c * y_std + y_mean
            mse = np.mean((pred - y_train_clean) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                # Store a closure for prediction
                w_copy = w.copy()
                def make_predictor_ridge(w, F_mean, F_std, y_mean, y_std, alpha_unused):
                    def predict(X):
                        Xr = np.asarray(X, dtype=np.float64)
                        F_pred = build_features(Xr)
                        F_pred_c = (F_pred - F_mean) / F_std
                        pred_c = F_pred_c @ w
                        return pred_c * y_std + y_mean
                    return predict
                best_predictor = make_predictor_ridge(w_copy, F_mean, F_std, y_mean, y_std, alpha)
        except:
            pass

    # Strategy 2: Sparse Selection (LASSO-like via coordinate descent or simple thresholding)
    # If the true equation is sparse, this can help generalization if ridge overfits
    # But since we have 200 points and potentially <200 features, ridge is usually safe.
    # Let's try a simple thresholding approach on the ridge solution.
    if best_mse < 1e-10:
        # If we already have near-perfect fit, don't bother with sparsity
        pass
    else:
        # Get the least regularized solution (alpha=0 or small)
        try:
            w_full, *_ = np.linalg.lstsq(F_centered, y_centered, rcond=None)
            abs_w = np.abs(w_full)
            
            # Try different sparsity levels
            for frac in [0.1, 0.2, 0.3, 0.5]:
                if time_left() < 5:
                    break
                threshold = np.percentile(abs_w, frac * 100)
                mask = abs_w > threshold
                if mask.sum() < 2:
                    mask = abs_w >= np.max(abs_w) * 0.1
                
                F_sub = F_centered[:, mask]
                w_sub, *_ = np.linalg.lstsq(F_sub, y_centered, rcond=None)
                
                pred_c = F_sub @ w_sub
                pred = pred_c * y_std + y_mean
                mse = np.mean((pred - y_train_clean) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    w_sub_copy = w_sub.copy()
                    mask_copy = mask.copy()
                    def make_predictor_sparse(w, mask, F_mean, F_std, y_mean, y_std):
                        def predict(X):
                            Xr = np.asarray(X, dtype=np.float64)
                            F_pred = build_features(Xr)
                            F_pred_c = (F_pred - F_mean) / F_std
                            pred_c = F_pred_c[:, mask] @ w
                            return pred_c * y_std + y_mean
                        return predict
                    best_predictor = make_predictor_sparse(w_sub_copy, mask_copy, F_mean, F_std, y_mean, y_std)
        except:
            pass

    # Strategy 3: Neural Network (if time permits and previous methods didn't get perfect fit)
    # This is a fallback for highly non-linear functions that might not be well-captured
    # by the fixed feature set.
    if best_mse > 1e-5 and time_left() > 50:
        try:
            # Simple MLP using NumPy
            X_norm = (X_train - X_min) / X_range
            y_norm = (y_train_clean - y_mean) / y_std
            
            # Architecture: d -> 64 -> 64 -> 1
            hidden1 = 64
            hidden2 = 64
            lr = 0.01
            epochs = 1000
            
            # Initialize
            np.random.seed(42)
            W1 = np.random.randn(d, hidden1) * np.sqrt(2.0 / d)
            b1 = np.zeros(hidden1)
            W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
            b2 = np.zeros(hidden2)
            W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
            b3 = np.zeros(1)
            
            # Adam optimizer state
            mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
            mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
            mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
            mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
            mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)
            mb3 = np.zeros_like(b3); vb3 = np.zeros_like(b3)
            
            beta1 = 0.9; beta2 = 0.999; eps = 1e-8
            t = 0
            
            best_nn_loss = np.inf
            best_nn_params = (W1, b1, W2, b2, W3, b3)
            
            for epoch in range(epochs):
                if time_left() < 10:
                    break
                
                # Forward
                z1 = X_norm @ W1 + b1
                a1 = np.tanh(z1)
                z2 = a1 @ W2 + b2
                a2 = np.tanh(z2)
                z3 = a2 @ W3 + b3
                pred_norm = z3.ravel()
                
                # Loss
                loss = np.mean((pred_norm - y_norm) ** 2)
                if loss < best_nn_loss:
                    best_nn_loss = loss
                    best_nn_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
                
                # Backward
                dz3 = 2 * (pred_norm - y_norm) / n
                dW3 = a2.T @ dz3
                db3 = dz3.sum(axis=0)
                da2 = dz3 @ W3.T
                dz2 = da2 * (1 - a2 ** 2)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ W2.T
                dz1 = da1 * (1 - a1 ** 2)
                dW1 = X_norm.T @ dz1
                db1 = dz1.sum(axis=0)
                
                # Adam Update
                t += 1
                mW1 = beta1 * mW1 + (1 - beta1) * dW1; vW1 = beta2 * vW1 + (1 - beta2) * dW1 ** 2
                mW1_hat = mW1 / (1 - beta1 ** t); vW1_hat = vW1 / (1 - beta2 ** t)
                W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)
                
                mb1 = beta1 * mb1 + (1 - beta1) * db1; vb1 = beta2 * vb1 + (1 - beta2) * db1 ** 2
                mb1_hat = mb1 / (1 - beta1 ** t); vb1_hat = vb1 / (1 - beta2 ** t)
                b1 -= lr * mb1_hat / (np.sqrt(vb1_hat) + eps)
                
                mW2 = beta1 * mW2 + (1 - beta1) * dW2; vW2 = beta2 * vW2 + (1 - beta2) * dW2 ** 2
                mW2_hat = mW2 / (1 - beta1 ** t); vW2_hat = vW2 / (1 - beta2 ** t)
                W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)
                
                mb2 = beta1 * mb2 + (1 - beta1) * db2; vb2 = beta2 * vb2 + (1 - beta2) * db2 ** 2
                mb2_hat = mb2 / (1 - beta1 ** t); vb2_hat = vb2 / (1 - beta2 ** t)
                b2 -= lr * mb2_hat / (np.sqrt(vb2_hat) + eps)
                
                mW3 = beta1 * mW3 + (1 - beta1) * dW3; vW3 = beta2 * vW3 + (1 - beta2) * dW3 ** 2
                mW3_hat = mW3 / (1 - beta1 ** t); vW3_hat = vW3 / (1 - beta2 ** t)
                W3 -= lr * mW3_hat / (np.sqrt(vW3_hat) + eps)
                
                mb3 = beta1 * mb3 + (1 - beta1) * db3; vb3 = beta2 * vb3 + (1 - beta2) * db3 ** 2
                mb3_hat = mb3 / (1 - beta1 ** t); vb3_hat = vb3 / (1 - beta2 ** t)
                b3 -= lr * mb3_hat / (np.sqrt(vb3_hat) + eps)
            
            # Evaluate best NN
            W1, b1, W2, b2, W3, b3 = best_nn_params
            z1 = X_norm @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            a2 = np.tanh(z2)
            z3 = a2 @ W3 + b3
            pred_norm = z3.ravel()
            pred = pred_norm * y_std + y_mean
            mse = np.mean((pred - y_train_clean) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                W1, b1, W2, b2, W3, b3 = best_nn_params
                def make_predictor_nn(W1, b1, W2, b2, W3, b3, X_min, X_range, y_mean, y_std):
                    def predict(X):
                        Xr = np.asarray(X, dtype=np.float64)
                        Xn = (Xr - X_min) / X_range
                        z1 = Xn @ W1 + b1
                        a1 = np.tanh(z1)
                        z2 = a1 @ W2 + b2
                        a2 = np.tanh(z2)
                        z3 = a2 @ W3 + b3
                        return (z3.ravel() * y_std + y_mean)
                    return predict
                best_predictor = make_predictor_nn(W1, b1, W2, b2, W3, b3, X_min, X_range, y_mean, y_std)
        except:
            pass

    # Fallback if something went wrong
    if best_predictor is None:
        # Simple linear regression
        F_lin = np.column_stack([np.ones(n), X_train])
        w_lin, *_ = np.linalg.lstsq(F_lin, y, rcond=None)
        def predict(X):
            Xr = np.asarray(X, dtype=np.float64)
            return np.column_stack([np.ones(Xr.shape[0]), Xr]) @ w_lin
        return predict

    return best_predictor

# EVOLVE-BLOCK-END
