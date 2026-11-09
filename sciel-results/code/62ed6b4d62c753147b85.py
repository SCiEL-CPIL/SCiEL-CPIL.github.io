"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_raw_value(c0, c4, c8):
    """Compute the raw uncertainty product A(g)*A(F(g)) for the Hermite-Gaussian test function."""
    # P(0) = 0 constraint determines c12
    # H0(0)=1, H4(0)=-12, H8(0)=2520, H12(0)=-103950
    # c0*1 + c4*(-12) + c8*2520 + c12*(-103950) = 0
    c12 = (c0 - 12*c4 + 2520*c8) / 103950.0
    
    # Build P(y) where y=x^2, P(x) = sum p_i * x^(2i)
    # H0 = 1
    # H4 = 12 - 48y + 16y^2
    # H8 = 2520 - 20160y + 13440y^2 - 3072y^3 + 256y^4
    # H12 = 103950 - 30801600y + 19353600y^2 - 6451200y^3 + 1048320y^4 - 98304y^5 + 4096y^6
    
    p = np.zeros(7)
    p[0] = c0 + 12*c4 + 2520*c8 + 103950*c12
    p[1] = -48*c4 - 20160*c8 - 30801600*c12
    p[2] = 16*c4 + 13440*c8 + 19353600*c12
    p[3] = -3072*c8 - 6451200*c12
    p[4] = 256*c8 + 1048320*c12
    p[5] = -98304*c12
    p[6] = 4096*c12
    
    # I(n) = ∫_{-∞}^{∞} x^(2n) exp(-x^2) dx = sqrt(pi) * (2n-1)!! / 2^n
    def I(n):
        if n == 0:
            return np.sqrt(np.pi)
        # (2n-1)!! = (2n)! / (2^n * n!)
        # Use log-gamma for numerical stability
        from math import lgamma
        log_val = 0.5*np.log(np.pi) + lgamma(n + 0.5) - 0.5*np.log(np.pi)
        # Actually: ∫x^(2n)exp(-x^2)dx = sqrt(pi) * Gamma(n+0.5) / Gamma(0.5) * ... 
        # Gamma(n+0.5)/Gamma(0.5) = (2n)!/(4^n n!) ... let me just compute directly
        result = 1.0
        for k in range(1, 2*n, 2):
            result *= k
        return np.sqrt(np.pi) * result / (2.0 ** n)
    
    # P(x)^2 = (sum_i p_i x^(2i))^2 = sum_{i,j} p_i p_j x^(2(i+j))
    # q_m = sum_{i+j=m} p_i p_j
    q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            if i + j <= 12:
                q[i+j] += p[i] * p[j]
    
    # norm_sq = ∫|g|^2 dx = sum_m q_m I(m)
    norm_sq = 0.0
    for m in range(13):
        norm_sq += q[m] * I(m)
    
    # x2_norm = ∫x^2|g|^2 dx = sum_m q_m I(m+1)
    x2_norm = 0.0
    for m in range(13):
        x2_norm += q[m] * I(m+1)
    
    if norm_sq <= 0:
        return 1e10
    
    # A(g)^2 = x2_norm/norm_sq (since <x>=0)
    # F(g) = sqrt(2*pi) exp(-k^2/2) P(k) for even Hermite polynomials
    # So A(F(g))^2 = x2_norm/norm_sq (same)
    # A(g)*A(F(g)) = x2_norm/norm_sq
    return x2_norm / norm_sq


def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Verify seed gives ~1.0
    seed_val = _compute_raw_value(*seed)
    
    # If our formula doesn't give ~1.0 for the seed, something is wrong
    # In that case, just return the seed
    if abs(seed_val - 1.0) > 0.1:
        return seed
    
    # Local search to minimize the raw value
    best_x = seed.copy()
    best_val = seed_val
    
    # Nelder-Mead local search
    from scipy.optimize import minimize
    
    def objective(x):
        return _compute_raw_value(x[0], x[1], x[2])
    
    try:
        result = minimize(objective, best_x, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_x
    
    # BFGS polish
    try:
        result = minimize(objective, best_x, method='BFGS',
                         options={'maxiter': 2000, 'gtol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
