import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import time

def _precompute_hermite_coeffs(max_n=13):
    """
    Pre-compute monomial coefficients for physicists' Hermite polynomials H_0 ... H_max_n.
    coeffs[n][k] is the coefficient of x^k in H_n(x).
    """
    coeffs = [[0.0] * (n + 1) for n in range(max_n + 1)]
    coeffs[0][0] = 1.0
    if max_n >= 1:
        coeffs[1][1] = 2.0
    for n in range(2, max_n + 1):
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        for k in range(n + 1):
            val = 0.0
            if k - 1 >= 0:
                val += 2.0 * coeffs[n-1][k-1]
            if n - 2 >= 0 and k <= n - 2:
                val -= 2.0 * (n - 1) * coeffs[n-2][k]
            coeffs[n][k] = val
    return coeffs

# Pre-compute coefficients at module level to save time in construct()
_H_COEFFS = _precompute_hermite_coeffs(13)

# Pre-compute moments M_n = int x^n exp(-x^2/2) dx for n up to 26
# M_n = 0 if n is odd.
# M_{2k} = (2k)! / k! * sqrt(pi/2)
_MOMENTS = [0.0] * 27
for n in range(27):
    if n % 2 == 1:
        _MOMENTS[n] = 0.0
    else:
        k = n // 2
        _MOMENTS[n] = factorial(2*k) / factorial(k) * np.sqrt(np.pi / 2.0)

def _compute_integrals(c0, c4, c8, c12):
    """
    Compute the upper bound C4 for given coefficients.
    c12 is derived from the constraint g(0)=0, but here we pass it explicitly for flexibility or compute it.
    The objective is (M2 * K) / N^2.
    """
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # p_coeffs[k] is coefficient of x^k in P(x)
    p_coeffs = np.zeros(13)
    
    # H0 is degree 0
    p_coeffs[0] += c0 * _H_COEFFS[0][0]
    
    # H4 is degree 4
    for k in range(5):
        p_coeffs[k] += c4 * _H_COEFFS[4][k]
        
    # H8 is degree 8
    for k in range(9):
        p_coeffs[k] += c8 * _H_COEFFS[8][k]
        
    # H12 is degree 12
    for k in range(13):
        p_coeffs[k] += c12 * _H_COEFFS[12][k]
        
    # Q(x) = P'(x) - x/2 P(x)
    # P'(x) = sum_{k=1}^{12} k p_k x^{k-1}
    # x/2 P(x) = sum_{k=0}^{12} 0.5 p_k x^{k+1}
    # q_coeffs[j] for x^j
    # From P': contributes k * p_k to x^{k-1}. So for j, k=j+1.
    # From -x/2 P: contributes -0.5 p_k to x^{k+1}. So for j, k=j-1.
    
    q_coeffs = np.zeros(14) # Degree can go up to 13?
    # Max degree of P is 12.
    # P' has max degree 11.
    # x P has max degree 13.
    # So Q has max degree 13.
    
    for k in range(1, 13):
        # Term from derivative: k * p_k x^{k-1}
        q_coeffs[k-1] += k * p_coeffs[k]
        
    for k in range(13):
        # Term from -x/2 P: -0.5 p_k x^{k+1}
        q_coeffs[k+1] -= 0.5 * p_coeffs[k]
        
    # Calculate integrals
    # N = int P^2 exp(-x^2/2) dx
    # M2 = int x^2 P^2 exp(-x^2/2) dx
    # K = int Q^2 exp(-x^2/2) dx
    
    def _calc_int(p1, p2, m):
        total = 0.0
        len1 = len(p1)
        len2 = len(p2)
        for i in range(len1):
            if p1[i] == 0.0: continue
            for j in range(len2):
                if p2[j] == 0.0: continue
                n = i + j + m
                if n < 27:
                    total += p1[i] * p2[j] * _MOMENTS[n]
        return total
        
    N = _calc_int(p_coeffs, p_coeffs, 0)
    if N <= 1e-12:
        return float('inf')
        
    M2 = _calc_int(p_coeffs, p_coeffs, 2)
    K = _calc_int(q_coeffs, q_coeffs, 0)
    
    return (M2 * K) / (N * N)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin for 210s limit
    
    # Seed
    c0_seed, c4_seed, c8_seed = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    x0 = np.array([c0_seed, c4_seed, c8_seed])
    
    # Constraint: g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Wait, let's re-verify H12(0))
    # H_n(0) for even n: (-1)^{n/2} (n-1)!! * 2^{n/2} ? No.
    # Let's check coeffs[12][0].
    # Recurrence:
    # H0(0) = 1
    # H2(0) = -2
    # H4(0) = 12
    # H6(0) = -120
    # H8(0) = 1680
    # H10(0) = -30240
    # H12(0) = 665280
    # Let's verify with the precomputed coeffs.
    h12_0 = _H_COEFFS[12][0]
    h8_0 = _H_COEFFS[8][0]
    h4_0 = _H_COEFFS[4][0]
    h0_0 = _H_COEFFS[0][0]
    
    # c0*h0_0 + c4*h4_0 + c8*h8_0 + c12*h12_0 = 0
    # c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
        return _compute_integrals(c0, c4, c8, c12)
        
    # Initial check
    seed_val = objective(x0)
    best_c = x0.copy()
    best_val = seed_val
    
    # Strategy:
    # 1. Try Nelder-Mead from seed (fast, local)
    # 2. Try Differential Evolution (global, robust) if time permits
    # 3. Polish with BFGS/L-BFGS-B
    
    if time.time() > deadline - 20:
        return best_c
        
    try:
        # Phase 1: Local refinement from seed
        # Nelder-Mead is good for small dim smooth problems
        res_nm = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})
        
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_c = res_nm.x
            
        if time.time() > deadline - 40:
            # Polish and return
            pass
        else:
            # Phase 2: Global search with Differential Evolution
            # Bounds: The coefficients can vary. Let's set reasonable bounds.
            # Seed: c0 ~ -1, c4 ~ 0.04, c8 ~ 0.0003
            # Let's try a wider range to ensure we don't miss a better basin.
            bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
            
            # DE is slower, so limit population size and iterations
            # maxiter=100, popsize=15 -> 1500 evaluations.
            # Each eval is fast (<1ms). So 1.5s approx.
            res_de = differential_evolution(objective, bounds, 
                                            maxiter=100, 
                                            popsize=15, 
                                            tol=1e-12, 
                                            seed=42,
                                            polish=False) # We will polish manually
            
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_c = res_de.x
            
            # Phase 3: High-precision polish
            # Use L-BFGS-B or BFGS for final convergence
            res_polish = minimize(objective, best_c, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False})
            
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_c = res_polish.x
                
    except Exception:
        pass
        
    # Final safety check
    if time.time() > deadline - 5:
        return best_c
        
    # One more very tight polish if time remains
    try:
        res_final = minimize(objective, best_c, method='Nelder-Mead', 
                             options={'maxiter': 100, 'xatol': 1e-14, 'fatol': 1e-14, 'disp': False})
        if res_final.fun < best_val:
            best_c = res_final.x
    except Exception:
        pass
        
    return np.array(best_c)
