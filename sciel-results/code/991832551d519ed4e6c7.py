"""Rewritten program with fast approximate optimization and exact final check."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, gamma
from scipy.optimize import minimize
from scipy.linalg import solve

# EVOLVE-BLOCK-START

def _precompute_laguerre_coeffs():
    """
    Precompute the coefficients of L_{2k}^{(-1/2)}(x) in the power basis x^j.
    Returns a matrix C of shape (26, 51) where C[k, j] is the coeff of x^j in L_{2k}^{(-1/2)}(x).
    """
    alpha = -0.5
    K = 26
    MaxDeg = 50
    C = np.zeros((K, MaxDeg + 1))
    
    for k in range(K):
        n = 2 * k
        for j in range(n + 1):
            # L_n^(alpha)(x) = sum_{j=0}^n binom(n+alpha, n-j) * (-1)^j / j! * x^j
            # binom(n+alpha, n-j) = Gamma(n+alpha+1) / (Gamma(n-j+1) * Gamma(j+alpha+1))
            try:
                num = gamma(n + alpha + 1)
                den = gamma(n - j + 1) * gamma(j + alpha + 1)
                if den == 0:
                    coeff = 0.0
                else:
                    coeff = num / den
            except Exception:
                coeff = 0.0
            
            # Term: coeff * (-1)^j / j!
            fact_j = 1.0
            for m in range(2, j + 1):
                fact_j *= m
            if j > 0:
                coeff *= ((-1) ** j) / fact_j
            else:
                coeff *= 1.0
                
            C[k, j] = coeff
    return C

# Precompute once
L_COEFFS = _precompute_laguerre_coeffs()

def _fast_score(z_list):
    """
    Fast approximate score calculation using numpy.
    Returns the largest simple real root of q(x), or 0 if none.
    """
    z = np.array(z_list)
    if np.any(z <= 0) or np.any(np.diff(np.sort(z)) < 1e-10):
        return float('inf')
        
    alpha = -0.5
    K = 26
    MaxDeg = 50
    
    # Build A (26x26) and b (26)
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    # L_{2k}^{(-1/2)}(0) = L_{2k}^{(-1/2)}(0)
    # We can use the precomputed coeffs: C[k, 0]
    for k in range(K):
        A[0, k] = L_COEFFS[k, 0]
    b[0] = 0
    
    # Row 1: g'(0) = 1
    # L_{2k}'(0) = -L_{2k-1}^{(1/2)}(0) for k>0, 0 for k=0
    # L_n^(alpha)(0) = binom(n+alpha, n)
    for k in range(K):
        if k == 0:
            A[1, k] = 0.0
        else:
            n = 2 * k - 1
            alpha_p = 0.5
            # L_n^(alpha_p)(0) = Gamma(n+alpha_p+1) / (Gamma(n+1) * Gamma(alpha_p+1))
            try:
                val = gamma(n + alpha_p + 1) / (gamma(n + 1) * gamma(alpha_p + 1))
                A[1, k] = -val
            except Exception:
                A[1, k] = 0.0
    b[1] = 1
    
    # Rows 2..25: g(z_i) = 0, g'(z_i) = 0
    for i in range(12):
        zi = z[i]
        row_g = 2 + 2 * i
        row_gp = 3 + 2 * i
        
        for k in range(K):
            # g(z_i) = sum c_k L_{2k}(z_i)
            A[row_g, k] = eval_genlaguerre(2 * k, alpha, zi)
            
            # g'(z_i) = sum c_k L'_{2k}(z_i) = sum c_k (-L_{2k-1}^{(1/2)}(z_i))
            if k == 0:
                A[row_gp, k] = 0.0
            else:
                A[row_gp, k] = -eval_genlaguerre(2 * k - 1, 0.5, zi)
        
        b[row_g] = 0
        b[row_gp] = 0
    
    try:
        c = solve(A, b)
    except Exception:
        return float('inf')
    
    # Compute g(x) coefficients in power basis
    # G = L_COEFFS @ c
    G = L_COEFFS @ c
    # Trim to MaxDeg
    G = G[:MaxDeg + 1]
    
    # Construct Denominator D(x) = x * prod (x - z_i)^2
    # Roots: 0, z_1, z_1, ..., z_12, z_12
    roots_d = np.concatenate(([0.0], np.repeat(z, 2)))
    D = np.poly(roots_d) # Coeffs in descending order
    # Convert D to ascending order for numpy polynomial division
    D_asc = D[::-1]
    
    # Polynomial Division: Q = G / D
    # numpy.polydiv expects descending order
    G_desc = G[::-1]
    Q_desc, R_desc = np.polydiv(G_desc, D)
    
    # Q is the quotient
    # Find real roots of Q
    Q_roots = np.roots(Q_desc)
    
    max_r = 0.0
    eps = 1e-12 # Approximation for 1e-198 in float
    
    for r in Q_roots:
        if np.isreal(r):
            r_real = np.real(r)
            # Check if it's a simple root by checking sign change
            # q(r - eps) * q(r + eps) < 0
            # We need to evaluate Q at r +/- eps
            # Use Horner's method or np.polyval
            try:
                q_minus = np.polyval(Q_desc, r_real - eps)
                q_plus = np.polyval(Q_desc, r_real + eps)
                if q_minus * q_plus < 0:
                    if r_real > max_r:
                        max_r = r_real
            except Exception:
                pass
                
    return max_r

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Uses fast optimization to find a good configuration, then returns it.
    The evaluator will run the exact check on these z_i.
    """
    deadline = time.time() + 550.0
    
    # Initial guesses
    # 1. Zeros of L_12^(-1/2)(x)
    # We can approximate zeros using numpy roots of the polynomial
    # L_12^(-1/2)(x) coeffs are in L_COEFFS[6, :] (since 2*6=12)
    # But we need 12 zeros. L_12 has 12 zeros.
    try:
        l12_coeffs_desc = L_COEFFS[6, :13][::-1]
        z_l12 = np.real(np.roots(l12_coeffs_desc))
        z_l12 = np.sort(z_l12)
        if np.all(z_l12 > 0):
            candidates = [z_l12]
        else:
            candidates = []
    except Exception:
        candidates = []
        
    # 2. Bessel zeros (fallback)
    if not candidates:
        z_bessel = []
        for i in range(1, 13):
            x = (i - 0.25) * math.pi
            z_bessel.append(x - 1.0/(8.0*x) + 9.0/(128.0*x**3))
        candidates.append(np.array(z_bessel))
        
    # 3. Geometric progression
    z_geo = np.array([2.0 * 1.5**i for i in range(12)])
    candidates.append(z_geo)

    best_z = candidates[0]
    best_score = _fast_score(best_z)
    
    # Optimization
    # Parameterize z_i = exp(u_i) to ensure positivity
    # We need to ensure distinctness. We'll just try and hope, or add a penalty.
    # Let's use Nelder-Mead on the raw z_i with bounds? Nelder-Mead doesn't support bounds well.
    # We'll use the log-transformed space.
    
    def objective(u):
        z = np.exp(u)
        # Check distinctness
        z_sorted = np.sort(z)
        if np.any(np.diff(z_sorted) < 1e-8):
            return 1e6
        return _fast_score(z)
        
    # Initial point in log space
    u0 = np.log(best_z)
    
    # Run optimization
    # Time limit: 300 seconds
    start_time = time.time()
    try:
        # Method: Nelder-Mead
        # We might need to scale the problem.
        res = minimize(objective, u0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 2000})
        
        if res.success or res.fun < best_score:
            z_opt = np.exp(res.x)
            if np.all(z_opt > 0):
                # Validate distinctness
                z_sorted = np.sort(z_opt)
                if np.all(np.diff(z_sorted) > 1e-8):
                    score_opt = res.fun
                    if score_opt < best_score:
                        best_score = score_opt
                        best_z = z_opt
    except Exception:
        pass
        
    # If we have time, try a second round with different initial guesses or finer tolerance
    if time.time() < deadline - 100:
        for cand in candidates:
            if np.array_equal(cand, best_z):
                continue
            u0_c = np.log(cand)
            try:
                res_c = minimize(objective, u0_c, method='Nelder-Mead', 
                                 options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 1000, 'maxfev': 1000})
                if res_c.fun < best_score:
                    z_c = np.exp(res_c.x)
                    if np.all(z_c > 0) and np.all(np.diff(np.sort(z_c)) > 1e-8):
                        best_score = res_c.fun
                        best_z = z_c
            except Exception:
                pass
            if time.time() > deadline - 60:
                break
                
    # Final validation
    best_z = np.array(best_z, dtype=float)
    if len(best_z) != 12 or np.any(best_z <= 0) or not np.all(np.isfinite(best_z)):
        # Fallback to Bessel
        best_z = np.array([2.0 * 1.5**i for i in range(12)])
        
    # Ensure distinctness
    z_sorted = np.sort(best_z)
    if np.any(np.diff(z_sorted) < 1e-6):
        # Perturb
        np.random.seed(42)
        best_z = z_sorted + np.random.uniform(0, 0.1, 12)
        best_z = np.sort(best_z)
        
    return list(best_z)

# EVOLVE-BLOCK-END
