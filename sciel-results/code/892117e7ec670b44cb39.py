import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Constants for Hermite polynomials H_0, H_4, H_8, H_12
    K = np.array([0, 4, 8, 12])
    
    # Precompute factorials and powers of 2 for norms
    # Norm of H_k is sqrt(pi) * 2^k * k!
    # We can ignore sqrt(pi) as it cancels out in ratios
    norms = 2.0 ** K * np.array([np.math.factorial(k) for k in K], dtype=float)
    
    # Helper to get norm of H_m
    # If m < 0, norm is 0
    # If m not in K, we might need it for cross terms?
    # In S_2, we need <H_{k+2} H_l>. If l is in K, and k+2 == l, we need norm of H_l.
    # In <P'' P>, we need <H_{k-2} H_l>. If k-2 == l, we need norm of H_l.
    # In <x P' P>, we need <H_{m+1} H_l> and <H_{m-1} H_l>.
    # m = k-1. So m+1=k, m-1=k-2.
    # So we need norms for H_k and H_{k-2}.
    # K contains 0, 4, 8, 12.
    # k-2 can be -2, 2, 6, 10.
    # We need norms for H_2, H_6, H_10 as well?
    # Yes, for the cross terms in the sums.
    
    def get_norm(m):
        if m < 0:
            return 0.0
        return (2.0 ** m) * np.math.factorial(m)
    
    def compute_moments(coeffs):
        # coeffs are [a, b, c, d]
        C = coeffs
        S_0 = np.sum(C**2 * norms)
        
        # S_2 = <x^2 P^2>
        S_2 = 0.0
        for i in range(len(K)):
            for j in range(len(K)):
                ki = K[i]
                kj = K[j]
                # x^2 H_ki = 0.25 H_{ki+2} + (ki+0.5) H_{ki} + 0.5*ki*(ki-1) H_{ki-2}
                term = 0.0
                if kj == ki + 2:
                    term += 0.25 * get_norm(kj)
                elif kj == ki:
                    term += (ki + 0.5) * get_norm(kj)
                elif kj == ki - 2:
                    term += 0.5 * ki * (ki - 1) * get_norm(kj)
                S_2 += C[i] * C[j] * term
        
        # S_g'' = <P'' P> - 2<x P' P> + S_2 - S_0
        # <P'' P>
        term_pp = 0.0
        for i in range(len(K)):
            for j in range(len(K)):
                ki = K[i]
                kj = K[j]
                # P'' has H_{ki-2} coeff 4*ki*(ki-1)
                # P has H_{kj}
                if kj == ki - 2:
                    term_pp += C[i] * C[j] * (4.0 * ki * (ki - 1)) * get_norm(kj)
        
        # <x P' P>
        term_xpp = 0.0
        for i in range(len(K)):
            for j in range(len(K)):
                ki = K[i]
                kj = K[j]
                # P' has H_{ki-1} coeff 2*ki
                # x H_{ki-1} = 0.5 H_{ki} + (ki-1) H_{ki-2}
                # So <x H_{ki-1} H_{kj}> is non-zero if kj == ki or kj == ki-2
                m = ki - 1
                coeff_p_prime = 2.0 * ki
                term = 0.0
                if kj == ki: # m+1 = ki
                    term += 0.5 * get_norm(kj)
                elif kj == ki - 2: # m-1 = ki-2
                    term += m * get_norm(kj)
                term_xpp += C[i] * C[j] * coeff_p_prime * term
        
        S_gpp = term_pp - 2.0 * term_xpp + S_2 - S_0
        
        return S_0, S_2, S_gpp

    def objective(vars):
        a, b, c = vars
        # d = -(a + 12b + 1680c) / 665280
        d = -(a + 12.0 * b + 1680.0 * c) / 665280.0
        coeffs = np.array([a, b, c, d])
        
        S_0, S_2, S_gpp = compute_moments(coeffs)
        
        if S_0 <= 1e-10:
            return 1e100
        if S_gpp >= 0:
            # Variance in frequency must be positive, so -S_gpp > 0
            # If S_gpp is positive, it's invalid or numerical error
            # For valid functions, <g'',g> should be negative
            return 1e100
        
        val = (S_2 * (-S_gpp)) / (S_0 * S_0)
        return val

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Optimization
    start_time = time.time()
    best_val = objective(x0)
    best_x = x0
    
    # Try a few random restarts
    np.random.seed(42)
    for _ in range(20):
        x_rand = np.random.normal(0, 1, 3)
        res = minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        if time.time() - start_time > 150:
            break

    # Polish with L-BFGS-B
    res = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-12})
    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x

    a, b, c = best_x
    return np.array([a, b, c])
