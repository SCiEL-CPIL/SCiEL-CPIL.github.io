# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import differential_evolution

# Constants
L = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature nodes and weights for the sphere
u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# Precompute the Spherical Harmonics Y_m(u, phi) for m = -10..10
# Y_0 = N_0 P_10^0(u)
# Y_m (m>0) = sqrt(2) N_m P_10^m(u) cos(m phi)
# Y_m (m<0) = sqrt(2) N_|m| P_10^{|m|}(u) sin(|m| phi)
# N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)

def fact(n):
    return math.factorial(n)

Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)

for m in range(-L, L + 1):
    abs_m = abs(m)
    # Normalization N_m
    # N_m = sqrt( (2L+1)/(4*pi) * (L-abs_m)! / (L+abs_m)! )
    # L=10, 2L+1 = 21
    N_m = math.sqrt(21.0 / (4.0 * np.pi) * fact(L - abs_m) / fact(L + abs_m))
    
    # P_L^m(u) with Condon-Shortley phase
    # scipy.special.lpmv(m, n, x) includes the Condon-Shortley phase (-1)^m
    # We need P_10^{|m|}(u)
    P_vals = lpmv(abs_m, L, u_nodes) # Shape (N_THETA,)
    
    if m == 0:
        # Y_0 = N_0 P_10^0(u)
        # Broadcast to all phi
        Y_flat = np.tile(P_vals * N_m, (1, N_PHI)).flatten()
    else:
        if m > 0:
            # Y_m = sqrt(2) N_m P_10^m(u) cos(m phi)
            cos_term = np.cos(m * phi_nodes) # (N_PHI,)
            # Y(i, j) = sqrt(2) N_m P_vals[i] cos_term[j]
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, cos_term) # (N_THETA, N_PHI)
            Y_flat = Y_2d.flatten()
        else:
            # Y_m = sqrt(2) N_|m| P_10^{|m|}(u) sin(|m| phi)
            sin_term = np.sin(abs_m * phi_nodes) # (N_PHI,)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, sin_term) # (N_THETA, N_PHI)
            Y_flat = Y_2d.flatten()
    
    Y_matrix[:, m + 10] = Y_flat

# Precompute flattened weights
w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

def compute_ratio(c):
    """
    Computes R = ||f||_4 / ||f||_2
    c: array of 21 coefficients
    """
    # f = Y_matrix @ c
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15:
        return 1e15 # Large penalty for zero function
    
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq)) # L4 norm is (integral |f|^4)^{1/4}
    
    return norm4 / norm2

def objective(c):
    """
    Objective function for minimization.
    We minimize R.
    """
    # Scale invariance: normalize c to unit norm to avoid scale issues in DE
    # If c is zero, return large penalty
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        return 1e15
    c_norm = c / norm
    return compute_ratio(c_norm)

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    # Bounds for Differential Evolution
    # We search in a box [-1, 1]^21. Since the objective is scale-invariant 
    # (we normalize internally), the magnitude doesn't matter, only direction.
    bounds = [(-1.0, 1.0)] * 21
    
    best_c = None
    best_score = 1e15
    
    # Run Differential Evolution
    # maxiter and popsize are tuned to fit within time limit
    # DE is stochastic, so we run it once with sufficient population
    try:
        # DE can be slow. We limit maxiter.
        # 21 dimensions. popsize=15 is reasonable.
        # maxiter=1000 might be too slow. Let's try 200-500.
        # We will monitor time.
        
        # Use a callback to check time and break if needed?
        # scipy.optimize.differential_evolution doesn't support time-based stopping directly in older versions
        # but we can wrap it or just set conservative limits.
        
        # Let's use a wrapper that checks time inside the objective? No, DE calls objective many times.
        # Better: run DE with a fixed budget and hope it fits.
        # 21 dims * 15 pop * 200 iter = 63000 evaluations.
        # Each eval is ~21*9100 ops. ~200k flops.
        # 63000 * 200k = 12.6 Gflops. Should take < 1s in numpy? 
        # Actually, matrix-vector product is fast.
        # Let's be safe: maxiter=100, popsize=20.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        c_opt = res.x
        score = objective(c_opt)
        
        if score < best_score:
            best_score = score
            best_c = c_opt.copy()
            
    except Exception as e:
        # Fallback if DE fails
        pass
    
    if best_c is None:
        # Fallback: return a valid default
        best_c = np.zeros(21)
        best_c[20] = 1.0 # m=10
        
    # Ensure validity
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
