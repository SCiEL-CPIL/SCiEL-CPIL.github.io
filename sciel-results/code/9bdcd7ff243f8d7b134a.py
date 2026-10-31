"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _double_factorial(n):
    """Compute n!! for non-negative integer n. (-1)!! = 0!! = 1."""
    if n < 0:
        return 1
    if n == 0 or n == 1:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def _compute_integrals_exact(a, b, c):
    """Compute I0, I2, J0 exactly for g(x) = P(x) e^{-x^2/2}.
    
    P(x) = a*H0 + b*H4 + c*H8 + d*H12 where d is fixed by g(0)=0.
    All Hermite polynomials here are even, so P(x) = sum_{j=0}^{6} p_j x^{2j}.
    
    I0 = ∫ P^2 e^{-x^2} dx
    I2 = ∫ x^2 P^2 e^{-x^2} dx  
    J0 = ∫ (P' - xP)^2 e^{-x^2} dx
    """
    # Build P(x) = sum p_j x^{2j}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 84x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 100800x^2 + 10395
    
    # d from g(0) = 0: a*1 + b*3 + c*105 + d*10395 = 0
    d = -(a + 3*b + 105*c) / 10395.0
    
    # p_j coefficient of x^{2j} in P(x)
    # j=0: constant term
    p = np.zeros(7)
    p[0] = a + 3*b + 105*c + 10395*d  # = 0 by construction
    p[1] = -6*b - 420*c - 100800*d    # x^2
    p[2] = b + 210*c + 55440*d        # x^4
    p[3] = -28*c - 12600*d            # x^6
    p[4] = c + 1260*d                 # x^8
    p[5] = -84*d                      # x^10
    p[6] = d                          # x^12
    
    # Precompute integral values: ∫ x^{2n} e^{-x^2} dx = (2n-1)!! / 2^n * sqrt(pi)
    # For n=0: integral = sqrt(pi)
    # (2n-1)!! for n=0 is (-1)!! = 1
    sqrt_pi = math.sqrt(math.pi)
    # integral_vals[n] = ∫ x^{2n} e^{-x^2} dx
    integral_vals = np.zeros(13)  # need up to x^{2*12} = x^{24}, so n up to 12
    for n in range(13):
        if n == 0:
            integral_vals[n] = sqrt_pi
        else:
            integral_vals[n] = _double_factorial(2*n - 1) / (2.0**n) * sqrt_pi
    
    # I0 = sum_{i,j} p_i p_j * integral_vals[i+j]
    I0 = 0.0
    I2 = 0.0
    J0 = 0.0
    for i in range(7):
        for j in range(7):
            pi_pj = p[i] * p[j]
            I0 += pi_pj * integral_vals[i + j]
            I2 += pi_pj * integral_vals[i + j + 1]
            # For J0: (P' - xP) has coefficient (2k-1)p_k for x^{2k-1}
            # (P'-xP)^2 = sum_{i,j} (2i-1)(2j-1) p_i p_j x^{2(i+j-1)}
            # ∫ x^{2(i+j-1)} e^{-x^2} dx = integral_vals[i+j-1]
            if i >= 1 and j >= 1:
                J0 += (2*i - 1) * (2*j - 1) * pi_pj * integral_vals[i + j - 1]
    
    return I0, I2, J0


def _objective_and_grad(params):
    """Compute objective value and gradient.
    
    Objective: sqrt(I2 * J0) / I0
    """
    a, b, c = params
    I0, I2, J0 = _compute_integrals_exact(a, b, c)
    
    if I0 <= 0 or J0 <= 0 or I2 <= 0:
        return 1e10, np.zeros(3)
    
    # f = sqrt(I2 * J0) / I0
    # Let's compute gradient via chain rule
    # f = (I2 * J0)^{1/2} * I0^{-1}
    # df/da = (1/2)(I2*J0)^{-1/2} * (J0 * dI2/da + I2 * dJ0/da) * I0^{-1} - (I2*J0)^{1/2} * I0^{-2} * dI0/da
    
    # We need dI0/da, dI2/da, dJ0/da and similarly for b, c
    # I0 = sum_{i,j} p_i p_j * integral_vals[i+j]
    # p_k depends on (a,b,c) linearly
    
    # dp_k/da, dp_k/db, dp_k/dc
    # p[0] = a + 3b + 105c + 10395d, d = -(a+3b+105c)/10395
    # So p[0] = 0 always
    # p[1] = -6b - 420c - 100800d = -6b - 420c + 100800(a+3b+105c)/10395
    # p[2] = b + 210c + 55440d = b + 210c - 55440(a+3b+105c)/10395
    # etc.
    
    # Let me compute dp_k/dparam for each parameter
    # p_k = alpha_k * a + beta_k * b + gamma_k * c
    # where the coefficients come from the Hermite polynomials and d's dependence
    
    # d = -(a + 3b + 105c)/10395
    # dp_k/da = (coeff of a in p_k) + (coeff of d in p_k) * (-1/10395)
    # dp_k/db = (coeff of b in p_k) + (coeff of d in p_k) * (-3/10395)
    # dp_k/dc = (coeff of c in p_k) + (coeff of d in p_k) * (-105/10395)
    
    # Coefficients of a, b, c, d in each p_k:
    # p[0]: a:1, b:3, c:105, d:10395
    # p[1]: a:0, b:-6, c:-420, d:-100800
    # p[2]: a:0, b:1, c:210, d:55440
    # p[3]: a:0, b:0, c:-28, d:-12600
    # p[4]: a:0, b:0, c:1, d:1260
    # p[5]: a:0, b:0, c:0, d:-84
    # p[6]: a:0, b:0, c:0, d:1
    
    # d_coeff[k] = coefficient of d in p_k
    d_coeff = np.array([10395, -100800, 55440, -12600, 1260, -84, 1], dtype=float)
    # a_coeff[k] = coefficient of a in p_k (excluding d's contribution)
    a_direct = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
    b_direct = np.array([3, -6, 1, 0, 0, 0, 0], dtype=float)
    c_direct = np.array([105, -420, 210, -28, 1, 0, 0], dtype=float)
    
    # dp_k/da = a_direct[k] + d_coeff[k] * (-1/10395)
    # dp_k/db = b_direct[k] + d_coeff[k] * (-3/10395)
    # dp_k/dc = c_direct[k] + d_coeff[k] * (-105/10395)
    
    dp_da = a_direct + d_coeff * (-1.0 / 10395.0)
    dp_db = b_direct + d_coeff * (-3.0 / 10395.0)
    dp_dc = c_direct + d_coeff * (-105.0 / 10395.0)
    
    # dI0/da = sum_{i,j} (dp_da[i]*p[j] + p[i]*dp_da[j]) * integral_vals[i+j]
    #        = 2 * sum_{i,j} dp_da[i]*p[j] * integral_vals[i+j] (by symmetry)
    # Actually more carefully: dI0/da = sum_{i,j} (dp_i/da * p_j + p_i * dp_j/da) * IV[i+j]
    # By symmetry of the sum, this equals 2 * sum_i dp_i/da * sum_j p_j * IV[i+j]
    
    # Let's compute S_k = sum_j p_j * integral_vals[k+j] for k=0..6
    S = np.zeros(7)
    for k in range(7):
        S[k] = np.sum(p[k:] * integral_vals[k:k+7])  # sum over j from 0 to 6 of p[j]*IV[k+j]
    
    dI0_da = 2.0 * np.sum(dp_da * S)
    dI0_db = 2.0 * np.sum(dp_db * S)
    dI0_dc = 2.0 * np.sum(dp_dc * S)
    
    # Similarly for I2: IV2[i+j] = integral_vals[i+j+1]
    S2 = np.zeros(7)
    for k in range(7):
        S2[k] = np.sum(p[k:] * integral_vals[k+1:k+8])  # sum over j of p[j]*IV[k+1+j]
    
    dI2_da = 2.0 * np.sum(dp_da * S2)
    dI2_db = 2.0 * np.sum(dp_db * S2)
    dI2_dc = 2.0 * np.sum(dp_dc * S2)
    
    # For J0: J0 = sum_{i,j>=1} (2i-1)(2j-1) p_i p_j * IV[i+j-1]
    # dJ0/da = sum_{i,j>=1} (2i-1)(2j-1) (dp_i/da * p_j + p_i * dp_j/da) * IV[i+j-1]
    # By symmetry = 2 * sum_{i,j>=1} (2i-1)(2j-1) dp_i/da * p_j * IV[i+j-1]
    
    # Let T_k = sum_{j>=1} (2j-1) p_j * IV[k+j-1] for k>=1
    T = np.zeros(7)
    for k in range(1, 7):
        T[k] = np.sum((2*np.arange(1,7) - 1) * p[1:] * integral_vals[k-1:k+6])
    
    dJ0_da = 2.0 * np.sum((2*np.arange(1,7) - 1) * dp_da[1:] * T[1:])
    dJ0_db = 2.0 * np.sum((2*np.arange(1,7) - 1) * dp_db[1:] * T[1:])
    dJ0_dc = 2.0 * np.sum((2*np.arange(1,7) - 1) * dp_dc[1:] * T[1:])
    
    # Now compute gradient of f = sqrt(I2*J0)/I0
    prod = I2 * J0
    sqrt_prod = np.sqrt(prod)
    f = sqrt_prod / I0
    
    # df/dparam = (1/(2*sqrt(prod))) * (J0*dI2/dparam + I2*dJ0/dparam) / I0 - sqrt_prod * dI0/dparam / I0^2
    #           = (J0*dI2/dparam + I2*dJ0/dparam) / (2*sqrt(prod)*I0) - sqrt_prod * dI0/dparam / I0^2
    #           = (J0*dI2/dparam + I2*dJ0/dparam) / (2*sqrt(prod)*I0) - f * dI0/dparam / I0
    
    coeff1 = 1.0 / (2.0 * sqrt_prod * I0)
    coeff2 = f / I0
    
    grad_a = coeff1 * (J0 * dI2_da + I2 * dJ0_da) - coeff2 * dI0_da
    grad_b = coeff1 * (J0 * dI2_db + I2 * dJ0_db) - coeff2 * dI0_db
    grad_c = coeff1 * (J0 * dI2_dc + I2 * dJ0_dc) - coeff2 * dI0_dc
    
    return f, np.array([grad_a, grad_b, grad_c])


def construct():
    start_time = time.time()
    deadline = 170.0  # seconds, safety margin
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    start_points = [best_params.copy()]
    
    # Perturbed versions of the seed
    for i in range(20):
        perturbation = rng.randn(3) * 0.05
        start_points.append(seed_result + perturbation)
    
    # Random points
    for i in range(20):
        start_points.append(rng.randn(3) * 0.3)
    
    # Structured grid
    for a_val in [-1.5, -1.0, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
        for b_val in [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3]:
            for c_val in [-0.02, -0.005, 0.0, 0.005, 0.02]:
                start_points.append(np.array([a_val, b_val, c_val]))
    
    # Optimize from each starting point
    for sp in start_points:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            result = minimize(_objective_and_grad, sp, method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Refine with tighter tolerances
    if time.time() - start_time < deadline - 20:
        try:
            result = minimize(_objective_and_grad, best_params, method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead for fine-tuning
    if time.time() - start_time < deadline - 10:
        try:
            result = minimize(lambda x: _objective_and_grad(x)[0], best_params, 
                            method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Make sure we're at least as good as seed
    seed_score, _ = _objective_and_grad(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()
    
    return best_params
# EVOLVE-BLOCK-END
