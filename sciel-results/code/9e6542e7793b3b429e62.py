import numpy as np
import time
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin for 330s limit
    
    n, d = X_train.shape
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_mean = np.mean(y_train)
        return lambda X: np.full(len(X), y_mean)
        
    # Normalize X to [0, 1] for stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    
    X_norm = (X_train - X_min) / X_range
    
    # Feature Construction
    def build_features(X):
        Xn = (X - X_min) / X_range
        Xn = np.clip(Xn, 0.0, 1.0)
        
        cols = [np.ones(len(Xn))]
        
        # 1. Linear terms
        for j in range(d):
            cols.append(Xn[:, j])
            
        # 2. Chebyshev Polynomials up to degree 12
        # T_k(x) = cos(k * arccos(x))
        # We use the recurrence relation for stability:
        # T_0(x) = 1
        # T_1(x) = x
        # T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
        
        max_deg = 12
        for j in range(d):
            x = Xn[:, j]
            T0 = np.ones_like(x)
            T1 = x
            # T0 is already in cols[0] effectively, but let's add T1..Tmax
            # Actually, T1 is the linear term, already added.
            # So we start from T2
            if max_deg >= 2:
                T_prev = T0
                T_curr = T1
                for k in range(2, max_deg + 1):
                    T_next = 2 * x * T_curr - T_prev
                    cols.append(T_next)
                    T_prev = T_curr
                    T_curr = T_next
                    
        # 3. Fourier Terms
        # Frequencies: 1, 2, 4, 8
        freqs = [1.0, 2.0, 4.0, 8.0]
        for j in range(d):
            x = Xn[:, j]
            for f in freqs:
                cols.append(np.sin(2 * np.pi * f * x))
                cols.append(np.cos(2 * np.pi * f * x))
                
        # 4. Pairwise Interactions of Linear Terms (Simple cross terms)
        # This helps with multiplicative structures like x1*x2
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Xn[:, i] * Xn[:, j])
                
        # 5. RBFs with a few fixed centers
        # Centers: grid points in [0.2, 0.5, 0.8]
        centers = []
        if d == 1:
            centers = [0.2, 0.5, 0.8]
            for c in centers:
                gamma = 10.0
                cols.append(np.exp(-gamma * (Xn[:, 0] - c)**2))
        elif d == 2:
            # A few 2D centers
            centers = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)]
            for c in centers:
                gamma = 5.0
                dist_sq = (Xn[:, 0] - c[0])**2 + (Xn[:, 1] - c[1])**2
                cols.append(np.exp(-gamma * dist_sq))
        elif d == 3:
            # A few 3D centers
            centers = [(0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.8, 0.8, 0.8)]
            for c in centers:
                gamma = 3.0
                dist_sq = (Xn[:, 0] - c[0])**2 + (Xn[:, 1] - c[1])**2 + (Xn[:, 2] - c[2])**2
                cols.append(np.exp(-gamma * dist_sq))

        return np.column_stack(cols)
    
    F_train = build_features(X_train)
    
    # Ridge Regression
    # Use a small lambda to allow flexibility but prevent overfitting
    # With ~50-100 features and 200 samples, lambda=1.0 is reasonable.
    # We can try a few lambdas and pick the best based on training MSE? 
    # No, training MSE will always favor smaller lambda. 
    # Since we don't have test data, we rely on the basis being good enough.
    # Let's use lambda = 1.0.
    
    lambda_val = 1.0
    n_features = F_train.shape[1]
    
    # Solve (F^T F + lambda I) w = F^T y
    # Add regularization to diagonal
    I = np.eye(n_features)
    A = F_train.T @ F_train + lambda_val * I
    b = F_train.T @ y_train
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        
    # Check time
    if time.time() > deadline:
        # If we are running out of time, return what we have
        pass

    def predict(X):
        F_test = build_features(X)
        return F_test @ w

    return predict
