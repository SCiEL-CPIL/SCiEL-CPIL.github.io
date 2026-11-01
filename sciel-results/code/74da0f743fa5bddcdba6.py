import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def compute_d_sq(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return dists_sq[iu]

def soft_log_max(x, tau):
    """Approximation of log(max(x))"""
    m = np.max(x)
    if m > 700:
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def soft_log_min(x, tau):
    """Approximation of log(min(x))"""
    m = np.min(x)
    if m < -700:
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_X = None
    best_score = np.inf
    
    def normalize(X):
        """Center at origin and scale such that sum of squared distances is constant? 
        Or just scale such that min dist is 1? 
        Better: Scale such that RMS distance is 1 to keep numbers well-conditioned."""
        X_centered = X - np.mean(X, axis=0)
        d_sq = compute_d_sq(X_centered)
        if np.sum(d_sq) > 1e-10:
            # Scale so that mean squared distance is 1
            scale = np.sqrt(2.0 * N / (N-1) / np.sum(d_sq)) # approx rms
            X_scaled = X_centered * scale
        else:
            X_scaled = X_centered
        return X_scaled

    def score(X):
        d_sq = compute_d_sq(X)
        if np.min(d_sq) < 1e-10:
            return np.inf
        return (np.max(d_sq) / np.min(d_sq))**2

    # Initializations
    inits = []
    
    # 1. Fibonacci Sphere
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = np.linspace(-1, 1, N)
    z += (np.random.rand(N) - 0.5) / N
    r = np.sqrt(1.0 - z**2)
    t = 2 * np.pi * np.arange(N) / N
    fib = np.column_stack((r * np.cos(t), r * np.sin(t), z))
    inits.append(fib)
    
    # 2. Random
    for _ in range(5):
        inits.append(np.random.randn(N, D))

    # 3. Grid-like (Cubic)
    # 2x2x4 grid
    grid = np.array(list(np.meshgrid(range(2), range(2), range(4), indexing='ij'))).T.reshape(-1, 3)
    inits.append(grid[:14].astype(float))

    for X0 in inits:
        if time.time() > deadline - 20.0:
            break
            
        # Normalize initial guess
        X0 = normalize(X0)
        
        # Objective: Minimize log(max d^2) - log(min d^2)
        # Using soft approximations
        tau = 0.1
        
        def obj(X_flat):
            X = X_flat.reshape(N, D)
            # Re-center and re-scale during optimization to enforce invariance?
            # No, L-BFGS-B works better if we fix the gauge.
            # Let's just optimize the coordinates directly but the objective is scale invariant.
            # To avoid the optimizer wandering in scale/translation, we can penalize deviation from normalized state.
            # Or, simpler: just let it optimize. The log-ratio is scale invariant.
            # However, gradient of scale invariant function is zero in scale direction.
            # This creates a flat valley. L-BFGS-B might struggle.
            # Better: Fix the scale by constraining sum(d^2) = const?
            # SLSQP can handle constraints. L-BFGS-B cannot.
            # Let's use SLSQP for the main optimization. It's slower but handles constraints.
            
            d_sq = compute_d_sq(X)
            d_sq = np.maximum(d_sq, 1e-10)
            
            # Log soft max/min
            log_max = soft_log_max(d_sq, tau)
            log_min = soft_log_min(d_sq, tau)
            
            return log_max - log_min

        # We will use SLSQP with constraints to fix the gauge:
        # 1. Center of mass is 0 (3 constraints)
        # 2. Sum of squared distances is fixed (1 constraint) -> Fixes scale
        
        def constraints_def(X_flat):
            X = X_flat.reshape(N, D)
            # Constraint 1: Centroid is 0
            c1 = np.sum(X, axis=0)
            # Constraint 2: Sum of squared distances is 1 (arbitrary normalization)
            d_sq = compute_d_sq(X)
            c2 = np.sum(d_sq) - 1.0
            return np.append(c1, c2)
            
        # SLSQP requires a vector of constraints.
        # Inequality constraints: d_ij^2 > 0 is handled by the objective/log.
        
        try:
            # Run SLSQP
            res = minimize(
                obj, 
                X0.flatten(), 
                method='SLSQP', 
                constraints=[{'type': 'eq', 'fun': constraints_def}],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success:
                X_opt = res.x.reshape(N, D)
                # Final normalize just in case
                X_opt = normalize(X_opt)
                sc = score(X_opt)
                if sc < best_score:
                    best_score = sc
                    best_X = X_opt.copy()
                    
        except Exception:
            pass

    # Polish with L-BFGS-B on the normalized manifold?
    # Or just return best.
    
    if best_X is None:
        best_X = np.random.randn(N, D)
        
    return best_X
