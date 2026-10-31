import numpy as np
from scipy import optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s limit
    
    # Hermite polynomials (physicists')
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    def H12(x): return 4096*x**12 - 98304*x**10 + 944640*x**8 - 4423680*x**6 + 9031680*x**4 - 6652800*x**2 + 1034800
    
    # Grid for numerical integration
    x = np.linspace(-15, 15, 10000)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2)
    
    H0v = H0(x) * gauss
    H4v = H4(x) * gauss
    H8v = H8(x) * gauss
    H12v = H12(x) * gauss
    
    def compute_objective(coeffs):
        """Compute upper bound on C4 from three coefficients (a0, a4, a8)."""
        a0, a4, a8 = coeffs
        # a12 from constraint g(0) = 0
        a12 = -(a0 + 12*a4 + 1680*a8) / 1034800.0
        
        g = a0*H0v + a4*H4v + a8*H8v + a12*H12v
        
        norm2 = np.sum(g**2) * dx
        if norm2 < 1e-30:
            return 1e10
        
        # x*g for first moment
        xg = x * g
        norm_xg = np.sum(xg**2) * dx
        
        # For Fourier-invariant function, F[g] = g
        # Uncertainty product involves ||g||_2 * ||xg||_2
        # The bound C4 relates to the ratio
        
        # The upper bound on C4 is typically:
        # C4_bound = (||g||_2 * ||xg||_2) / (||g||_2 * ||F_g||_2) 
        # But since F[g] = g, this simplifies.
        
        # Actually, the uncertainty inequality is:
        # ||f||_2^2 <= C * ||f||_2 * ||x f||_2
        # So C >= ||f||_2 / ||x f||_2
        # The "upper bound" we're minimizing is this C value.
        
        C_bound = np.sqrt(norm2) / np.sqrt(norm_xg)
        
        return C_bound
    
    def compute_objective_logsign(params):
        """Log-magnitude and sign parameterization."""
        # params: [log|a0|, sign_param0, log|a4|, sign_param1, log|a8|, sign_param2]
        log_a0, s0, log_a4, s4, log_a8, s8 = params
        a0 = np.exp(log_a0) * np.tanh(s0)
        a4 = np.exp(log_a4) * np.tanh(s4)
        a8 = np.exp(log_a8) * np.tanh(s8)
        return compute_objective([a0, a4, a8])
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_score = compute_objective(seed)
    
    best_params = seed.copy()
    best_score = seed_score
    
    # Convert seed to log-sign parameterization
    def to_logsign(coeffs):
        params = []
        for c in coeffs:
            if abs(c) < 1e-15:
                params.extend([np.log(1e-10), 0.0])
            else:
                params.extend([np.log(abs(c)), np.arctanh(np.tanh(np.sign(c)))])
        return np.array(params)
    
    def from_logsign(params):
        coeffs = []
        for i in range(3):
            log_abs = params[2*i]
            s = params[2*i+1]
            coeffs.append(np.exp(log_abs) * np.tanh(s))
        return np.array(coeffs)
    
    initial_params = to_logsign(seed)
    
    # Try multiple restarts with different initial points
    best_result = None
    n_restarts = 0
    
    # First try from the seed
    try:
        res = optimize.minimize(
            compute_objective_logsign,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-12}
        )
        n_restarts += 1
        if res.fun < best_score:
            best_score = res.fun
            best_params = from_logsign(res.x)
    except:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_params
    
    # Try random restarts
    rng = np.random.default_rng(42)
    n_random = 0
    max_random = 20
    
    for i in range(max_random):
        if time.time() > deadline:
            break
        # Random initial point in log-sign space
        rand_params = np.array([
            rng.uniform(-5, 5), rng.uniform(-3, 3),
            rng.uniform(-5, 5), rng.uniform(-3, 3),
            rng.uniform(-5, 5), rng.uniform(-3, 3)
        ])
        try:
            res = optimize.minimize(
                compute_objective_logsign,
                rand_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-12}
            )
            n_random += 1
            if res.fun < best_score:
                best_score = res.fun
                best_params = from_logsign(res.x)
        except:
            pass
    
    # Polish with higher precision
    if time.time() < deadline - 10:
        try:
            res = optimize.minimize(
                compute_objective_logsign,
                to_logsign(best_params),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-15, 'ftol': 1e-15}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = from_logsign(res.x)
        except:
            pass
    
    return best_params
