import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

# Precompute constants for speed
MAX_N = 13
_MAX_N = MAX_N
_coeffs = [[0.0] * (n + 1) for n in range(MAX_N + 1)]
_coeffs[0][0] = 1.0
if MAX_N >= 1:
    _coeffs[1][1] = 2.0
for n in range(2, MAX_N + 1):
    for k in range(n + 1):
        val = 0.0
        if k - 1 >= 0:
            val += 2.0 * _coeffs[n-1][k-1]
        if n - 2 >= 0 and k <= n - 2:
            val -= 2.0 * (n - 1) * _coeffs[n-2][k]
        _coeffs[n][k] = val

# Precompute moments M_n = int x^n exp(-x^2/2) dx
_M_n = [0.0] * (2 * MAX_N + 5) # Max degree of product is 2*12=24, plus m=2 -> 26. Let's go up to 30 to be safe.
for n in range(len(_M_n)):
    if n % 2 == 1:
        _M_n[n] = 0.0
    else:
        k = n // 2
        # int x^{2k} exp(-x^2/2) dx = (2k)! / k! * sqrt(pi/2)
        _M_n[n] = factorial(2*k) / factorial(k) * np.sqrt(np.pi / 2.0)

def _compute_integrals(c0, c4, c8, c12):
    # p_coeffs[k] is coefficient of x^k in P(x)
    p_coeffs = [0.0] * MAX_N
    for k in range(MAX_N):
        p_coeffs[k] = c0 * _coeffs[0][k] + c4 * _coeffs[4][k] + c8 * _coeffs[8][k] + c12 * _coeffs[12][k]
        
    # Q(x) = P'(x) - x/2 P(x)
    # deg(P) = 12, deg(P') = 11, deg(xP) = 13. So Q has degree 13.
    q_coeffs = [0.0] * (MAX_N + 1)
    for k in range(MAX_N):
        # Derivative term: (k+1) p_{k+1} x^k
        if k + 1 < MAX_N:
            q_coeffs[k] += (k + 1) * p_coeffs[k + 1]
        # -x/2 P term: -0.5 p_k x^{k+1}
        q_coeffs[k + 1] -= 0.5 * p_coeffs[k]
        
    def calc_int(p1, p2, m):
        total = 0.0
        len1 = len(p1)
        len2 = len(p2)
        for i in range(len1):
            c1 = p1[i]
            if c1 == 0.0: continue
            for j in range(len2):
                c2 = p2[j]
                if c2 == 0.0: continue
                n = i + j + m
                if n < len(_M_n):
                    total += c1 * c2 * _M_n[n]
        return total
        
    N = calc_int(p_coeffs, p_coeffs, 0)
    M2 = calc_int(p_coeffs, p_coeffs, 2)
    K = calc_int(q_coeffs, q_coeffs, 0)
    
    if N <= 1e-12:
        return float('inf')
        
    return (M2 * K) / (N * N)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin
    
    # Seed
    c0_seed, c4_seed, c8_seed = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    
    # Constraint: g(0) = 0
    # c12 = -(c0 + 12 c4 + 1680 c8) / 665280
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
        return _compute_integrals(c0, c4, c8, c12)
        
    best_c = np.array([c0_seed, c4_seed, c8_seed])
    best_val = objective(best_c)
    
    # Multi-start optimization
    # Define a set of starting points around the seed and some random perturbations
    starts = [
        np.array([c0_seed, c4_seed, c8_seed]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-1.1, 0.045, 0.0003]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.5, 0.02, 0.0001]),
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(5):
        p = np.array([c0_seed, c4_seed, c8_seed])
        p = p + np.random.normal(0, 0.1, 3)
        starts.append(p)
        
    for x0 in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B which is robust for smooth functions
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
                
            # Polish with BFGS
            if time.time() < deadline - 5:
                res2 = minimize(objective, best_c, method='BFGS', 
                               options={'gtol': 1e-10, 'disp': False, 'maxiter': 100})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_c = res2.x
                    
        except Exception:
            continue
            
    return np.array(best_c)
