import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline

# EVOLVE-BLOCK-START

def fit(X_train, y_train):
    start_time = time.time()
    TIME_LIMIT = 300.0
    SAFETY_MARGIN = 10.0
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    n, d = X_train.shape
    u_train = X_train[:, 0]
    
    # Sort by x for potential smoothing
    sort_idx = np.argsort(u_train)
    u_sorted = u_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # We will evaluate a set of candidate models and pick the best one based on
    # 5-fold cross-validation MSE.
    
    candidates = []
    
    # 1. Polynomial Regression (Degrees 1-8)
    for deg in range(1, 9):
        V = np.vander(u_train, N=deg + 1, increasing=True)
        # Ridge regression with small lambda for stability at high degrees
        # We'll use lstsq for simplicity in candidate generation, but refine with CV
        try:
            # Use a fixed small ridge for stability
            lam = 1e-6 * (deg > 5)
            A = V.T @ V + lam * np.eye(deg + 1)
            b = V.T @ y_train
            coeffs = np.linalg.solve(A, b)
            
            def make_poly_pred(coeffs):
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    V_test = np.vander(u, N=len(coeffs), increasing=True)
                    return V_test @ coeffs
                return predict
            candidates.append(make_poly_pred(coeffs))
        except Exception:
            pass
            
    # 2. Fourier Series (Frequencies 1-10)
    # Represented as sum of A_k * cos(k*u) + B_k * sin(k*u)
    # We need to determine the fundamental frequency. 
    # Since x is in [-3,3] or [-2,2] or [0.1,6], the period might be 2*pi, pi, etc.
    # Let's try a few base frequencies: 1, 0.5, 2, pi, 2*pi
    base_freqs = [1.0, 0.5, 2.0, np.pi, 2 * np.pi, 1.0 / (2 * np.pi)]
    
    for base_f in base_freqs:
        for max_k in range(1, 6):
            # Build design matrix
            cols = [np.ones(n)]
            for k in range(1, max_k + 1):
                cols.append(np.cos(k * base_f * u_train))
                cols.append(np.sin(k * base_f * u_train))
            V = np.column_stack(cols)
            
            try:
                lam = 1e-6
                A = V.T @ V + lam * np.eye(V.shape[1])
                b = V.T @ y_train
                coeffs = np.linalg.solve(A, b)
                
                def make_fourier_pred(coeffs, base_f, max_k):
                    def predict(X):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        u = X[:, 0]
                        cols = [np.ones(len(u))]
                        for k in range(1, max_k + 1):
                            cols.append(np.cos(k * base_f * u))
                            cols.append(np.sin(k * base_f * u))
                        V_test = np.column_stack(cols)
                        return V_test @ coeffs
                    return predict
                candidates.append(make_fourier_pred(coeffs, base_f, max_k))
            except Exception:
                pass
                
    # 3. Exponential / Logarithmic / Power
    # y = a * exp(b * x)
    # y = a * x^b
    # y = a * log(x + b)
    # y = a / (x + b)
    
    # 3a. Exponential
    # Try fitting y = a * exp(b * x)
    # Linearize: log(y) = log(a) + b*x  (only if y > 0)
    # If y has mixed signs, this is tricky. 
    # Alternative: Grid search for b, then linear fit for a.
    b_vals = np.linspace(-2, 2, 100)
    for b in b_vals:
        y_pred_basis = np.exp(b * u_train)
        # Linear fit: y = a * y_pred_basis
        # a = sum(y * basis) / sum(basis^2)
        denom = np.sum(y_pred_basis ** 2)
        if denom < 1e-10:
            continue
        a = np.sum(y_train * y_pred_basis) / denom
        if not np.isfinite(a):
            continue
            
        def make_exp_pred(a, b):
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                u = X[:, 0]
                return a * np.exp(b * u)
            return predict
        candidates.append(make_exp_pred(a, b))
        
    # 3b. Power Law y = a * x^b
    # Only valid if x > 0. Check if all x > 0.
    if np.min(u_train) > 0:
        b_vals = np.linspace(-3, 3, 100)
        for b in b_vals:
            y_pred_basis = u_train ** b
            if not np.all(np.isfinite(y_pred_basis)):
                continue
            denom = np.sum(y_pred_basis ** 2)
            if denom < 1e-10:
                continue
            a = np.sum(y_train * y_pred_basis) / denom
            if not np.isfinite(a):
                continue
                
            def make_pow_pred(a, b):
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    # Handle negative u if possible? No, power law usually defined for x>0
                    # If test x < 0, we might get NaNs. 
                    # Since training x > 0, test x is likely > 0.
                    # But to be safe, if u < 0, use sign(u) * |u|^b?
                    # The task says ranges are like [0.1, 6], so x is always positive.
                    return a * (u ** b)
                return predict
            candidates.append(make_pow_pred(a, b))
            
    # 3c. Logarithmic y = a * log(x + b)
    # Grid search for b (offset)
    # b should be such that x+b > 0
    min_x = np.min(u_train)
    b_vals = np.linspace(min_x + 0.1, min_x + 5.0, 50)
    for b in b_vals:
        y_pred_basis = np.log(u_train + b)
        denom = np.sum(y_pred_basis ** 2)
        if denom < 1e-10:
            continue
        a = np.sum(y_train * y_pred_basis) / denom
        if not np.isfinite(a):
            continue
            
        def make_log_pred(a, b):
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                u = X[:, 0]
                return a * np.log(u + b)
            return predict
        candidates.append(make_log_pred(a, b))
        
    # 3d. Rational y = a / (x + b)
    b_vals = np.linspace(min_x + 0.1, min_x + 5.0, 50)
    for b in b_vals:
        y_pred_basis = 1.0 / (u_train + b)
        denom = np.sum(y_pred_basis ** 2)
        if denom < 1e-10:
            continue
        a = np.sum(y_train * y_pred_basis) / denom
        if not np.isfinite(a):
            continue
            
        def make_rat_pred(a, b):
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                u = X[:, 0]
                return a / (u + b)
            return predict
        candidates.append(make_rat_pred(a, b))
        
    # 4. Spline Interpolation
    # Use UnivariateSpline with different smoothing factors s
    # s=0 is interpolation, larger s is smoother
    s_vals = [0, 1, 10, 100, 1000, 10000]
    for s in s_vals:
        try:
            spl = UnivariateSpline(u_sorted, y_sorted, s=s, k=3)
            def make_spline_pred(spl):
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    return spl(u)
                return predict
            candidates.append(make_spline_pred(spl))
        except Exception:
            pass
            
    # 5. Gaussian Process / RBF Kernel Regression
    # Simple RBF: y(x) = sum_i alpha_i * exp(-gamma * (x - x_i)^2)
    # Solve for alpha using kernel matrix K.
    # K_ij = exp(-gamma * (u_i - u_j)^2)
    # alpha = (K + lambda I)^-1 y
    
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    for gamma in gammas:
        # Compute kernel matrix
        # (n, 1) - (1, n) -> (n, n)
        diff = u_train[:, np.newaxis] - u_train[np.newaxis, :]
        K = np.exp(-gamma * diff ** 2)
        
        try:
            lam = 1e-6
            A = K + lam * np.eye(n)
            alpha = np.linalg.solve(A, y_train)
            
            def make_rbf_pred(alpha, u_train, gamma):
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    # (m, 1) - (n, 1) -> (m, n)
                    diff = u[:, np.newaxis] - u_train[np.newaxis, :]
                    K_test = np.exp(-gamma * diff ** 2)
                    return K_test @ alpha
                return predict
            candidates.append(make_rbf_pred(alpha, u_train, gamma))
        except Exception:
            pass
            
    # Evaluate candidates using 5-Fold CV
    # To save time, we'll use a subset of folds or just one holdout set if too many candidates
    # But 200 points is small. Let's do 5-fold CV.
    
    # Stratified folds? No, just random split.
    # Use fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    best_mse = np.inf
    best_pred = None
    
    # If we have too many candidates, we might run out of time.
    # Let's limit the number of candidates to evaluate if it's huge.
    # Currently we have:
    # Polys: 8
    # Fourier: 6 * 5 = 30
    # Exp: 100
    # Pow: 100 (if x>0)
    # Log: 50
    # Rat: 50
    # Splines: 6
    # RBF: 5
    # Total ~ 350 candidates.
    # 5 folds * 350 candidates * (fit + predict)
    # Fitting is done above. We just need to predict on test folds and calculate MSE.
    # Predicting is fast.
    
    for pred_func in candidates:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        fold_mses = []
        for fold_idx in range(5):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(5) if i != fold_idx])
            
            # We already fitted on ALL data. 
            # Wait, for proper CV, we should refit on the training subset.
            # However, refitting 350 models * 5 folds = 1750 fits.
            # Most fits are cheap (linear algebra).
            # But RBF and Spline refits are more expensive.
            # Given the time limit, and that the data is noiseless, 
            # fitting on all data and evaluating on holdout might be biased (overfitting).
            # But for symbolic regression, we want the true function.
            # If we fit on all data, the model that fits the noise (if any) or the specific points best might not generalize.
            # But the problem states "noiseless training points".
            # So fitting on all data should recover the function exactly if the model family contains it.
            # The risk is overfitting to the specific sampling if the model is too complex.
            # With noiseless data, a complex model (like high degree poly or RBF) will interpolate perfectly, MSE=0.
            # Then all such models have MSE=0 on training.
            # We need to distinguish between the "true" simple model and the "complex" interpolating model.
            # CV is the way: the complex model will likely have higher error on unseen points if it's not the true function.
            
            # So we MUST refit on train_idx for each fold.
            
            u_tr = u_train[train_idx]
            y_tr = y_train[train_idx]
            u_te = u_train[test_idx]
            y_te = y_train[test_idx]
            
            # Refit logic is duplicated here. This is inefficient.
            # Alternative: 
            # 1. Identify the type of candidate.
            # 2. Refit only that type on u_tr, y_tr.
            # 3. Predict on u_te.
            
            # This requires restructuring the candidate generation to return a "fitter" function.
            
            # Let's restructure:
            # Instead of pre-fitting, we store the "recipe" for fitting.
            
            # This is a significant rewrite. Let's do it.
            
            # Break out of this loop if we decide to restructure.
            # Actually, let's just restructure the whole fit function to store recipes.
            
            pass # Placeholder
            
    # RESTRUCTURED APPROACH:
    # Define a list of (name, fit_func) where fit_func(X, y) returns predict(X)
    
    def create_polynomial_fitter(deg):
        def fit_func(X, y):
            V = np.vander(X, N=deg + 1, increasing=True)
            lam = 1e-6
            A = V.T @ V + lam * np.eye(deg + 1)
            b = V.T @ y
            coeffs = np.linalg.solve(A, b)
            def predict(X):
                V_test = np.vander(X, N=deg + 1, increasing=True)
                return V_test @ coeffs
            return predict
        return fit_func
        
    def create_fourier_fitter(base_f, max_k):
        def fit_func(X, y):
            cols = [np.ones(len(X))]
            for k in range(1, max_k + 1):
                cols.append(np.cos(k * base_f * X))
                cols.append(np.sin(k * base_f * X))
            V = np.column_stack(cols)
            lam = 1e-6
            A = V.T @ V + lam * np.eye(V.shape[1])
            b = V.T @ y
            coeffs = np.linalg.solve(A, b)
            def predict(X):
                cols = [np.ones(len(X))]
                for k in range(1, max_k + 1):
                    cols.append(np.cos(k * base_f * X))
                    cols.append(np.sin(k * base_f * X))
                V_test = np.column_stack(cols)
                return V_test @ coeffs
            return predict
        return fit_func
        
    def create_exp_fitter(b):
        def fit_func(X, y):
            y_pred_basis = np.exp(b * X)
            denom = np.sum(y_pred_basis ** 2)
            if denom < 1e-10:
                # Fallback to zero
                return lambda X: np.zeros(len(X))
            a = np.sum(y * y_pred_basis) / denom
            def predict(X):
                return a * np.exp(b * X)
            return predict
        return fit_func
        
    def create_pow_fitter(b):
        def fit_func(X, y):
            # Only works if X > 0
            if np.min(X) <= 0:
                return lambda X: np.zeros(len(X))
            y_pred_basis = X ** b
            if not np.all(np.isfinite(y_pred_basis)):
                return lambda X: np.zeros(len(X))
            denom = np.sum(y_pred_basis ** 2)
            if denom < 1e-10:
                return lambda X: np.zeros(len(X))
            a = np.sum(y * y_pred_basis) / denom
            def predict(X):
                return a * (X ** b)
            return predict
        return fit_func
        
    def create_log_fitter(b):
        def fit_func(X, y):
            y_pred_basis = np.log(X + b)
            if not np.all(np.isfinite(y_pred_basis)):
                return lambda X: np.zeros(len(X))
            denom = np.sum(y_pred_basis ** 2)
            if denom < 1e-10:
                return lambda X: np.zeros(len(X))
            a = np.sum(y * y_pred_basis) / denom
            def predict(X):
                return a * np.log(X + b)
            return predict
        return fit_func
        
    def create_rat_fitter(b):
        def fit_func(X, y):
            y_pred_basis = 1.0 / (X + b)
            if not np.all(np.isfinite(y_pred_basis)):
                return lambda X: np.zeros(len(X))
            denom = np.sum(y_pred_basis ** 2)
            if denom < 1e-10:
                return lambda X: np.zeros(len(X))
            a = np.sum(y * y_pred_basis) / denom
            def predict(X):
                return a / (X + b)
            return predict
        return fit_func
        
    def create_spline_fitter(s, k=3):
        def fit_func(X, y):
            try:
                sort_idx = np.argsort(X)
                X_s = X[sort_idx]
                y_s = y[sort_idx]
                spl = UnivariateSpline(X_s, y_s, s=s, k=k)
                def predict(X):
                    return spl(X)
                return predict
            except Exception:
                return lambda X: np.zeros(len(X))
        return fit_func
        
    def create_rbf_fitter(gamma):
        def fit_func(X, y):
            diff = X[:, np.newaxis] - X[np.newaxis, :]
            K = np.exp(-gamma * diff ** 2)
            lam = 1e-6
            A = K + lam * np.eye(len(X))
            try:
                alpha = np.linalg.solve(A, y)
            except Exception:
                alpha = np.linalg.lstsq(A, y, rcond=None)[0]
                
            def predict(X):
                diff = X[:, np.newaxis] - X[np.newaxis, :]
                # Wait, X here is the training X stored in closure? No, we need to pass it.
                # The fit_func returns predict(X). The predict function needs the training X.
                # We can capture X in the closure.
                # But X is an argument to fit_func.
                # So:
                pass
            # Correct closure:
            X_train_local = X
            def predict(X):
                diff = X[:, np.newaxis] - X_train_local[np.newaxis, :]
                K_test = np.exp(-gamma * diff ** 2)
                return K_test @ alpha
            return predict
        return fit_func
        
    # Build list of fitters
    fitters = []
    
    # Polynomials
    for deg in range(1, 9):
        fitters.append(create_polynomial_fitter(deg))
        
    # Fourier
    for base_f in [1.0, 0.5, 2.0, np.pi, 2 * np.pi]:
        for max_k in range(1, 6):
            fitters.append(create_fourier_fitter(base_f, max_k))
            
    # Exponential
    for b in np.linspace(-2, 2, 20): # Reduced grid for speed in CV
        fitters.append(create_exp_fitter(b))
        
    # Power Law (if x > 0)
    if np.min(u_train) > 0:
        for b in np.linspace(-3, 3, 20):
            fitters.append(create_pow_fitter(b))
            
    # Log
    min_x = np.min(u_train)
    for b in np.linspace(min_x + 0.1, min_x + 2.0, 10):
        fitters.append(create_log_fitter(b))
        
    # Rational
    for b in np.linspace(min_x + 0.1, min_x + 2.0, 10):
        fitters.append(create_rat_fitter(b))
        
    # Splines
    for s in [0, 1, 10, 100, 1000]:
        fitters.append(create_spline_fitter(s))
        
    # RBF
    for gamma in [0.1, 0.5, 1.0, 2.0]:
        fitters.append(create_rbf_fitter(gamma))
        
    # Evaluate using CV
    best_mse = np.inf
    best_fit_func = None
    
    for fit_func in fitters:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        fold_mses = []
        for fold_idx in range(5):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(5) if i != fold_idx])
            
            u_tr = u_train[train_idx]
            y_tr = y_train[train_idx]
            u_te = u_train[test_idx]
            y_te = y_train[test_idx]
            
            try:
                predict_func = fit_func(u_tr, y_tr)
                y_pred = predict_func(u_te)
                if not np.all(np.isfinite(y_pred)):
                    mse = np.inf
                else:
                    mse = np.mean((y_pred - y_te) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
                
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_fit_func = fit_func
            
    # Refit best model on all data
    if best_fit_func is not None:
        final_predict = best_fit_func(u_train, y_train)
    else:
        # Fallback
        final_predict = lambda X: np.zeros(len(X))
        
    # Wrap final predict to ensure 2D input handling
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        u = X[:, 0]
        y_pred = final_predict(u)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        return y_pred
        
    return predict

# EVOLVE-BLOCK-END
