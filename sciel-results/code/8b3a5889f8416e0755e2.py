import numpy as np
from scipy.optimize import differential_evolution
import time

# Coefficients for H_0, H_4, H_8, H_12 in basis {1, x^2, x^4, x^6, x^8, x^10, x^12}
# Index k corresponds to x^(2k)
H_coeffs = np.zeros((4, 7))
H_coeffs[0, 0] = 1.0 # H0
H_coeffs[1, 0] = 12.0
H_coeffs[1, 1] = -48.0
H_coeffs[1, 2] = 16.0 # H4
H_coeffs[2, 0] = 1680.0
H_coeffs[2, 1] = -13440.0
H_coeffs[2, 2] = 13440.0
H_coeffs[2, 3] = -3584.0
H_coeffs[2, 4] = 256.0 # H8
H_coeffs[3, 0] = 665280.0
H_coeffs[3, 1] = -153676800.0
H_coeffs[3, 2] = 118464000.0
H_coeffs[3, 3] = -39813120.0
H_coeffs[3, 4] = 6291456.0
H_coeffs[3, 5] = -367008.0
H_coeffs[3, 6] = 4096.0 # H12

# Precompute moments M_n(alpha) = int x^n e^{-alpha x^2} dx for even n
# M_{2k}(alpha) = (2k-1)!! / (2^k alpha^k) * sqrt(pi/alpha)
# We need n up to 2*6*4 = 48 for P^4.
max_power = 48
# alpha1 = 2*pi, alpha2 = 4*pi
alpha1 = 2 * np.pi
alpha2 = 4 * np.pi

def compute_moments(alpha, max_k):
    # Returns array M where M[k] = int x^(2k) e^{-alpha x^2} dx
    M = np.zeros(max_k + 1)
    M[0] = np.sqrt(np.pi / alpha)
    for k in range(1, max_k + 1):
        # M[k] = (2k-1) * M[k-1] / (2 * alpha)
        M[k] = (2 * k - 1) * M[k - 1] / (2 * alpha)
    return M

M1 = compute_moments(alpha1, 24) # max power 48 -> k=24
M2 = compute_moments(alpha2, 24)

def _compute_objective_analytic(coeffs):
    a0, a4, a8 = coeffs
    # Calculate a12
    # P(0) = a0*1 + a4*12 + a8*1680 + a12*665280 = 0
    a12 = -(a0 + 12 * a4 + 1680 * a8) / 665280.0
    
    # Power coefficients c for P(x)
    # c = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    c = a0 * H_coeffs[0] + a4 * H_coeffs[1] + a8 * H_coeffs[2] + a12 * H_coeffs[3]
    
    # Power coefficients d for Q(x) (Fourier side)
    # d = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
    d = a0 * H_coeffs[0] + a4 * H_coeffs[1] + a8 * H_coeffs[2] - a12 * H_coeffs[3]
    
    # Compute ||g||_2^2 = sum_{i,j} c_i c_j M1[i+j]
    # M1[k] is moment for x^(2k)
    norm2_sq = 0.0
    for i in range(7):
        if c[i] == 0: continue
        for j in range(7):
            if c[j] == 0: continue
            norm2_sq += c[i] * c[j] * M1[i + j]
            
    # Compute ||g||_4^2 = sum_{i,j,k,l} c_i c_j c_k c_l M2[i+j+k+l]
    norm4_sq = 0.0
    for i in range(7):
        if c[i] == 0: continue
        ci = c[i]
        for j in range(7):
            if c[j] == 0: continue
            cij = ci * c[j]
            for k in range(7):
                if c[k] == 0: continue
                cijk = cij * c[k]
                for l in range(7):
                    if c[l] == 0: continue
                    norm4_sq += cijk * c[l] * M2[i + j + k + l]
                    
    # Same for Q
    norm2_sq_hat = 0.0
    for i in range(7):
        if d[i] == 0: continue
        for j in range(7):
            if d[j] == 0: continue
            norm2_sq_hat += d[i] * d[j] * M1[i + j]
            
    norm4_sq_hat = 0.0
    for i in range(7):
        if d[i] == 0: continue
        di = d[i]
        for j in range(7):
            if d[j] == 0: continue
            dij = di * d[j]
            for k in range(7):
                if d[k] == 0: continue
                dijk = dij * d[k]
                for l in range(7):
                    if d[l] == 0: continue
                    norm4_sq_hat += dijk * d[l] * M2[i + j + k + l]
                    
    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    return obj

def construct():
    start_time = time.time()
    deadline = 190.0 # Safety margin for 210s limit
    
    # Seed from original
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_score = _compute_objective_analytic(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Differential Evolution
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    try:
        de_result = differential_evolution(
            _compute_objective_analytic,
            bounds,
            seed=42,
            maxiter=200, # More iterations due to speed
            tol=1e-12,
            polish=True, # Use L-BFGS-B for final polish
            polish_options={'maxiter': 1000, 'ftol': 1e-15}
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x.copy()
    except Exception as e:
        pass
        
    # Additional local refinement if time permits
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            res = minimize(_compute_objective_analytic, best_result, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
            
    return best_result
