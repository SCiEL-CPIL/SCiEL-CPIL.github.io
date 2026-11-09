"""Multi-Kernel Learning ensemble with Bayesian Optimization for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    # 1. Pre-compute distance matrices for all kernels to avoid recomputation during optimization
    # Kernels:
    # 0: RBF (Gaussian)
    # 1: Polynomial (degree 2, normalized)
    # 2: Sine (Periodic)
    # 3: Rational Quadratic
    
    # Compute pairwise squared distances
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    
    # Pre-compute base kernel structures where possible
    # For RBF: exp(-gamma * dist) -> depends on gamma
    # For Poly: (1 + dist/alpha)^degree -> depends on alpha
    # For Sine: sin(omega * dist) -> depends on omega
    # For RQ: 1 + dist/alpha -> depends on alpha

    # We will compute the full kernel matrix K inside the objective function
    # but we can cache sq_dist.

    def compute_kernels(params):
        """
        params: array of length 2 * n_kernels
        [w0, lam0, w1, lam1, w2, lam2, w3, lam3]
        w: weight for kernel i
        lam: length-scale/width parameter for kernel i
        """
        n_kernels = 4
        K_total = np.zeros((n, n))
        
        # 0: RBF
        gamma = params[0]
        w0 = params[1]
        K_rbf = np.exp(-gamma * sq_dist)
        K_total += w0 * K_rbf
        
        # 1: Polynomial (degree 2)
        # K_poly = (1 + sq_dist / alpha) ** 2
        # Note: Standard poly kernel is (x^T y + c)^d. 
        # Using distance-based polynomial is less common but valid.
        # Let's use standard polynomial kernel: (X @ X.T + 1) ** 2
        # We pre-compute X @ X.T
        # Actually, let's stick to distance-based for consistency or compute X@X.T once.
        # X@X.T is cheaper than sq_dist? No, sq_dist is already computed.
        # Poly kernel: (1 + X_i . X_j)^2. 
        # X_i . X_j = (||Xi||^2 + ||Xj||^2 - ||Xi-Xj||^2) / 2
        # ||Xi||^2 is sum of squares per row.
        # Let's just use the standard polynomial kernel defined on dot products.
        # We need X_train @ X_train.T.
        
        # To avoid recomputing X@X.T, we can assume it's fast enough for n=200.
        # But we can also use the distance-based poly: (1 + dist/alpha)^degree
        # Let's use distance-based to keep it simple and cached.
        alpha = params[2]
        w1 = params[3]
        # Avoid division by zero or negative alpha
        alpha = max(alpha, 1e-6)
        K_poly = (1.0 + sq_dist / alpha) ** 2
        K_total += w1 * K_poly
        
        # 2: Sine (Periodic)
        # K_sin = sin(omega * dist)
        # This kernel is not positive definite for all omega/dist, but often works in practice for regression.
        # A safer periodic kernel is: exp(-2 * sin^2(pi * dist / period))
        # Let's use the safer periodic kernel (Matern-like periodic)
        period = params[4]
        w2 = params[5]
        period = max(period, 1e-6)
        # Standard periodic kernel: exp(-2 * (sin(pi * dist / period))^2)
        K_per = np.exp(-2.0 * (np.sin(np.pi * sq_dist / period)) ** 2)
        # Note: sq_dist is squared distance. Periodic kernels usually use Euclidean distance, not squared.
        # dist = sqrt(sq_dist)
        dist = np.sqrt(sq_dist)
        K_per = np.exp(-2.0 * (np.sin(np.pi * dist / period)) ** 2)
        K_total += w2 * K_per
        
        # 3: Rational Quadratic
        # K_rq = (1 + dist^2 / alpha) ** -nu
        # Let's use nu=0.5 for simplicity (Matern 1/2) or nu=1.5.
        # Standard RQ: (1 + dist^2 / alpha) ** -nu
        alpha_rq = params[6]
        w3 = params[7]
        alpha_rq = max(alpha_rq, 1e-6)
        K_rq = (1.0 + sq_dist / alpha_rq) ** (-1.0)
        K_total += w3 * K_rq
        
        return K_total

    # Objective function for Nelder-Mead
    # Parameters: [w0, lam0, w1, lam1, w2, lam2, w3, lam3, reg]
    # We need to regularize the total kernel matrix: K_reg = K_total + reg * I
    
    def objective(params):
        # Extract weights and scales
        # w_i should be non-negative. We use exp(params) to ensure positivity.
        # scales should be positive. We use exp(params) as well.
        # reg should be positive.
        
        w0 = np.exp(params[0])
        lam0 = np.exp(params[1])
        w1 = np.exp(params[2])
        lam1 = np.exp(params[3])
        w2 = np.exp(params[4])
        lam2 = np.exp(params[5])
        w3 = np.exp(params[6])
        lam3 = np.exp(params[7])
        reg = np.exp(params[8])
        
        # Reconstruct kernel parameters for compute_kernels
        # compute_kernels expects [gamma, w0, alpha, w1, period, w2, alpha_rq, w3]
        # Wait, my compute_kernels above used params directly as gamma, w0, etc.
        # Let's rewrite compute_kernels to take explicit kernel params.
        
        # Actually, it's cleaner to just compute K inside objective.
        K_total = np.zeros((n, n))
        
        # RBF
        K_total += w0 * np.exp(-lam0 * sq_dist)
        
        # Poly
        K_total += w1 * (1.0 + sq_dist / max(lam1, 1e-6)) ** 2
        
        # Periodic
        dist = np.sqrt(sq_dist)
        K_total += w2 * np.exp(-2.0 * (np.sin(np.pi * dist / max(lam2, 1e-6))) ** 2)
        
        # RQ
        K_total += w3 * (1.0 + sq_dist / max(lam3, 1e-6)) ** (-1.0)
        
        K_reg = K_total + reg * np.eye(n)
        
        try:
            # Solve K_reg alpha = y
            alpha = np.linalg.solve(K_reg, y_train)
            pred = K_total @ alpha # Prediction uses unregularized kernel for consistency? 
            # Actually, prediction in KRR is K(x, X_train) @ alpha.
            # If we used K_reg for solving, the prediction is K @ alpha.
            # However, for evaluation on training set, we use K @ alpha.
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(pred)):
                return 1e10
                
            mse = np.mean((pred - y_train) ** 2)
            nrmse = mse / y_var
            
            # Add a small penalty for large weights to prevent overfitting to one kernel
            penalty = 0.01 * np.sum(w0 + w1 + w2 + w3)
            
            return nrmse + penalty
        except:
            return 1e10

    # Initial guess
    # Weights ~ 1, Scales ~ 1, Reg ~ 0.01
    x0 = np.zeros(9)
    x0[0] = 0.0  # w0 = 1
    x0[1] = 0.5  # lam0 = 1.65 (gamma)
    x0[2] = 0.0  # w1 = 1
    x0[3] = 0.0  # lam1 = 1 (alpha)
    x0[4] = 0.0  # w2 = 1
    x0[5] = 1.0  # lam2 = 2.7 (period)
    x0[6] = 0.0  # w3 = 1
    x0[7] = 0.0  # lam3 = 1 (alpha)
    x0[8] = -4.6 # reg = 0.01
    
    best_nrmse = np.inf
    best_params = None
    
    # Run Nelder-Mead optimization
    # We have ~300s budget. NM can be slow if not careful.
    # Let's limit the number of iterations.
    options = {'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=options)
        if result.fun < best_nrmse:
            best_nrmse = result.fun
            best_params = result.x
    except:
        pass
    
    # If optimization failed or didn't improve, try a few random restarts if time allows
    if time.time() < deadline - 30:
        np.random.seed(42)
        for i in range(10):
            if time.time() > deadline - 10:
                break
            # Random initialization
            x_rand = np.random.randn(9) * 0.5
            x_rand[8] = -4.6 # Keep reg small initially
            
            try:
                res_r = minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 100})
                if res_r.fun < best_nrmse:
                    best_nrmse = res_r.fun
                    best_params = res_r.x
            except:
                pass

    # If best_params is still None, fall back to simple RBF
    if best_params is None:
        # Simple RBF with default params
        gamma = 0.5
        reg = 1e-3
        K = np.exp(-gamma * sq_dist) + reg * np.eye(n)
        alpha = np.linalg.solve(K, y_train)
        
        def predict(X):
            dists = cdist(X, X_train, metric='sqeuclidean')
            K_test = np.exp(-gamma * dists)
            return K_test @ alpha
        return predict

    # Extract best parameters
    w0 = np.exp(best_params[0])
    lam0 = np.exp(best_params[1])
    w1 = np.exp(best_params[2])
    lam1 = np.exp(best_params[3])
    w2 = np.exp(best_params[4])
    lam2 = np.exp(best_params[5])
    w3 = np.exp(best_params[6])
    lam3 = np.exp(best_params[7])
    reg = np.exp(best_params[8])
    
    # Reconstruct K_train and solve for alpha
    K_train = np.zeros((n, n))
    K_train += w0 * np.exp(-lam0 * sq_dist)
    K_train += w1 * (1.0 + sq_dist / max(lam1, 1e-6)) ** 2
    dist = np.sqrt(sq_dist)
    K_train += w2 * np.exp(-2.0 * (np.sin(np.pi * dist / max(lam2, 1e-6))) ** 2)
    K_train += w3 * (1.0 + sq_dist / max(lam3, 1e-6)) ** (-1.0)
    
    K_reg = K_train + reg * np.eye(n)
    alpha = np.linalg.solve(K_reg, y_train)
    
    # Define prediction function
    def predict(X):
        # Compute distances from X to X_train
        diff = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dist_test = np.sum(diff**2, axis=2)
        dist_test = np.sqrt(sq_dist_test)
        
        K_test = np.zeros((X.shape[0], n))
        K_test += w0 * np.exp(-lam0 * sq_dist_test)
        K_test += w1 * (1.0 + sq_dist_test / max(lam1, 1e-6)) ** 2
        K_test += w2 * np.exp(-2.0 * (np.sin(np.pi * dist_test / max(lam2, 1e-6))) ** 2)
        K_test += w3 * (1.0 + sq_dist_test / max(lam3, 1e-6)) ** (-1.0)
        
        return K_test @ alpha

    return predict
# EVOLVE-BLOCK-END
