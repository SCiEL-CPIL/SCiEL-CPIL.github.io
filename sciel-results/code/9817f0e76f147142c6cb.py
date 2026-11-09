# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
import scipy.optimize

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
    return compute_ratio(c)

def map_complex_to_real(z):
    """
    z: array of shape (11,) complex.
    z[0] corresponds to c_0 (real part).
    z[k] for k=1..10 corresponds to c_{-k} + i c_k.
    Returns c: array of shape (21,) real.
    """
    c = np.zeros(21)
    c[10] = z[0].real
    
    for k in range(1, 11):
        c[10 - k] = z[k].imag
        c[10 + k] = z[k].real
    return c

def map_real_to_complex(c):
    """
    c: array of shape (21,) real.
    Returns z: array of shape (11,) complex.
    """
    z = np.zeros(11, dtype=np.complex128)
    z[0] = c[10]
    for k in range(1, 11):
        z[k] = c[10 + k] + 1j * c[10 - k]
    return z

def objective_complex(z):
    c = map_complex_to_real(z)
    return objective(c)

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    best_c = None
    best_score = 1e15
    
    # 1. Grid search over the 10-dimensional subspace of complex amplitudes
    # z_m = c_{-m} + i c_m for |m| >= 1 and c_0.
    # The ratio R is invariant under global phase rotation (multiplying all z_m by e^{i\alpha}).
    # This reduces the dimensionality by 1. We can fix the phase of z_{10} to be real and non-negative.
    # However, a full grid search in 10 dimensions is too expensive.
    # Instead, we will use a structured random search in the complex space, which effectively
    # samples the manifold of phase-invariant configurations more efficiently than real-space random search.
    
    # We will also exploit the fact that the optimal solution likely has a specific symmetry.
    # But let's stick to a robust multi-start optimization in the complex domain.
    
    rng = np.random.default_rng(42)
    
    # Generate initial seeds in complex space
    seeds_z = []
    
    # A. Single-m seeds
    for k in range(11):
        z = np.zeros(11, dtype=np.complex128)
        z[k] = 1.0
        seeds_z.append(z)
        
    # B. Random complex seeds
    for _ in range(200):
        # Generate random complex numbers
        z = (rng.standard_normal(11) + 1j * rng.standard_normal(11))
        seeds_z.append(z)
        
    # C. Structured seeds: real and imaginary parts
    for _ in range(100):
        z = np.zeros(11, dtype=np.complex128)
        # Randomly choose which m have real or imaginary parts
        for k in range(11):
            if rng.random() > 0.5:
                z[k] = rng.standard_normal()
            else:
                z[k] = 1j * rng.standard_normal()
        seeds_z.append(z)
        
    rng.shuffle(seeds_z)
    
    # 2. Optimization Loop
    # Use SLSQP or BFGS in the complex domain (treating real and imag as separate variables)
    # Since scipy.optimize.minimize works with real vectors, we will flatten the complex vector
    # to a real vector of length 22 (11 real + 11 imag).
    
    def flatten_complex(z):
        return np.concatenate([z.real, z.imag])
        
    def unflatten_complex(v):
        z = v[:11] + 1j * v[11:]
        return z
        
    def objective_flat(v):
        z = unflatten_complex(v)
        return objective_complex(z)
        
    # We want to minimize R.
    # The problem is scale invariant. We can add a constraint ||z||_2 = 1 to fix the scale.
    # This makes the optimization landscape on the unit sphere in C^11.
    # Constraint: sum(v_i^2) - 1 = 0
    
    constraint = {'type': 'eq', 'fun': lambda v: np.dot(v, v) - 1.0}
    
    for z0 in seeds_z:
        if time.time() > deadline:
            break
            
        v0 = flatten_complex(z0)
        norm_v = np.linalg.norm(v0)
        if norm_v < 1e-6:
            continue
        v0 = v0 / norm_v
        
        try:
            res = scipy.optimize.minimize(
                objective_flat,
                v0,
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            v_opt = res.x
            z_opt = unflatten_complex(v_opt)
            c_opt = map_complex_to_real(z_opt)
            
            score = objective(c_opt)
            
            if score < best_score:
                best_score = score
                best_c = c_opt.copy()
                
        except Exception:
            continue
            
    # If no optimization succeeded, fallback to best seed
    if best_c is None:
        # Return a valid default, e.g., m=10
        best_c = np.zeros(21)
        best_c[20] = 1.0
        
    # Ensure validity: max|c| < 1e-3 or > 1e5 is invalid.
    # Our optimization should keep it reasonable, but let's check.
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        # Scale up
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        # Scale down
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
