import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import time

# Precomputation
# Max degree of P is 24.
# We need factorials up to 24+24+1 = 49.
MAX_DEG = 24
FACTS = np.array([1.0] * (2 * MAX_DEG + 2))
for i in range(1, 2 * MAX_DEG + 2):
    FACTS[i] = FACTS[i-1] * i

# M_{ij} = (i+j+1)! for i,j in 0..24
# i, j are powers of x in P(x) = sum a_k x^k
# Integral of x^i x^j x e^{-x} is (i+j+1)!
M = np.zeros((MAX_DEG + 1, MAX_DEG + 1))
for i in range(MAX_DEG + 1):
    for j in range(MAX_DEG + 1):
        M[i, j] = FACTS[i + j + 1]

def objective(roots):
    # roots: array of 12 positive numbers
    # Construct Q(x) = prod (x - r_i)
    # numpy.poly expects roots and returns coefficients in descending order
    # We want ascending order for the matrix product
    q_coeffs_desc = np.poly(roots)
    q_coeffs_asc = q_coeffs_desc[::-1]
    
    # P(x) = Q(x)^2
    p_coeffs_asc = np.convolve(q_coeffs_asc, q_coeffs_asc)
    
    # Ensure length is 25 (degree 24)
    if len(p_coeffs_asc) < MAX_DEG + 1:
        p_coeffs_asc = np.pad(p_coeffs_asc, (0, MAX_DEG + 1 - len(p_coeffs_asc)))
    else:
        p_coeffs_asc = p_coeffs_asc[:MAX_DEG + 1]
        
    # V = sum_{i,j} p_i p_j M_{ij}
    # V = p^T M p
    Mp = M @ p_coeffs_asc
    val = np.dot(p_coeffs_asc, Mp)
    
    return val

def construct():
    start_time = time.time()
    deadline = 600.0 # 600s limit, keep margin
    
    # Initial guess: Zeros of Laguerre L_12
    # roots_laguerre returns (x, w)
    x_init, _ = sp.roots_laguerre(12)
    # Ensure positive and sorted
    x_init = np.sort(x_init)
    
    # Bounds: x_i > 0. Upper bound can be large, say 1000.
    bounds = [(1e-6, 1000.0) for _ in range(12)]
    
    best_val = objective(x_init)
    best_roots = x_init.copy()
    
    # Try to optimize
    try:
        # Use L-BFGS-B for speed and bounds handling
        res = opt.minimize(objective, x_init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_roots = res.x
    except Exception:
        pass
    
    # If time permits, try a second pass with Nelder-Mead or more iterations?
    # L-BFGS-B is usually sufficient for this smooth-ish landscape.
    # Let's add a local refinement with Nelder-Mead if we have time.
    if time.time() - start_time < deadline - 30:
        try:
            res2 = opt.minimize(objective, best_roots, method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_roots = res2.x
        except Exception:
            pass
            
    # Ensure roots are positive and sorted for output
    best_roots = np.sort(np.abs(best_roots))
    # If any root is 0 or negative due to numerical error, clip to small positive
    best_roots[best_roots < 1e-9] = 1e-9
    
    return list(best_roots)

# EVOLVE-BLOCK-END
