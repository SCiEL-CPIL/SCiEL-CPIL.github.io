"""
Optimized uncertainty bound using Differential Evolution for global search 
and L-BFGS-B for local polishing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Safety margin: evaluator kills at 210s. We aim to finish by 195s.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 195.0

def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    Represented as vector p = [p0, p1, p2, p3, p4, p5, p6] for x^0, x^2, ..., x^12.
    """
    H = np.zeros((4, 7))
    # H_0 = 1
    H[0, 0] = 1.0
    # H_4 = x^4 - 6x^2 + 3
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    # H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    # H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H

def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..12.
    I[n] = (2n-1)!! * sqrt(pi) / 2^n
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I

_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _calc_bound_and_grad(params):
    """
    Compute the uncertainty bound A(f)*A(F) and its gradient.
    params = [a0, a4, a8]
    a12 is determined by P(0)=0 constraint: a0 + 3a4 + 105a8 + 945a12 = 0
    """
    a0, a4, a8 = params[0], params[1], params[2]
    # Constraint: sum c_i H_i(0) = 0 => a0*1 + a4*3 + a8*105 + a12*945 = 0
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    # Moments for position space: m_k = int x^{2k} |P|^2 e^{-x^2} dx
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0: continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
            
    # Moments for frequency space: n_k = 2*pi * int xi^{2k} |Q|^2 e^{-xi^2} dx
    # Q(x) = i^6 P(i x) ? No, Fourier transform of P(x)e^{-x^2/2} relates to P(-i k) or similar.
    # The seed code uses q = p * (-1)^k. This corresponds to the fact that 
    # FT of x^n is proportional to i^n. For even n=2k, i^{2k} = (-1)^k.
    # So the polynomial in frequency domain has coefficients p_k * (-1)^k.
    q = p * _SIGN
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0: continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
            
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient calculation
    # d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2 * m0 - m2 * dm0) / m0^2
    
    # Derivatives of a12 w.r.t a0, a4, a8
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for c in range(3):
        # dp/da_c = H[c] + da12[c]*H[3]
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        # dm0 = 2 * sum_i dp_i (sum_j p_j I[i+j])
        # dm2 = 2 * sum_i dp_i (sum_j p_j I[1+i+j])
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0: continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += p[j] * _I[i+j]
                s2 += p[j] * _I[1+i+j]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
            
        dq = dp * _SIGN
        
        # dn0 = 4*pi * sum_i dq_i (sum_j q_j I[i+j])
        # dn2 = 4*pi * sum_i dq_i (sum_j q_j I[1+i+j])
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0: continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += q[j] * _I[i+j]
                s2 += q[j] * _I[1+i+j]
            dn0 += 4.0 * np.pi * dqi * s0
            dn2 += 4.0 * np.pi * dqi * s2
            
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _calc_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _calc_bound_and_grad(params)
    return val, grad

def construct():
    """
    1. Global search using Differential Evolution.
    2. Local polish using L-BFGS-B with analytic gradient.
    3. Final tight polish.
    """
    # Define bounds for a0, a4, a8. 
    # Based on seed [-1.13, 0.04, 0.0003], we can use wider bounds for global search.
    # The function grows as |a| increases, so a reasonable bound is [-5, 5] for a0, 
    # [-1, 1] for a4, [-0.05, 0.05] for a8.
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.05, 0.05)]
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 1: Differential Evolution
    # DE is stochastic and global. 
    # We allocate about 100 seconds for this.
    # maxiter: 100 generations is usually enough for 3D.
    # popsize: 15 (15*3=45 individuals).
    if time.time() < _DEADLINE - 110:
        try:
            # Seed the RNG for reproducibility in this block, but DE is robust
            res_de = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=150, 
                tol=1e-10, 
                popsize=20, 
                init='latinhypercube'
            )
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass

    # Phase 2: Local Optimization from DE result (or seed if DE failed)
    # L-BFGS-B is very fast with analytic gradients.
    start_point = best_params
    if time.time() < _DEADLINE - 20:
        try:
            res_lbfgs = minimize(
                _objective_grad, 
                start_point, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_lbfgs.fun < best_val and np.isfinite(res_lbfgs.fun):
                best_val = res_lbfgs.fun
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass

    # Phase 3: Polish with Nelder-Mead (derivative-free) to escape potential flat spots
    if time.time() < _DEADLINE - 10:
        try:
            res_nm = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish from the best point found so far
    if time.time() < _DEADLINE - 2:
        try:
            res_final = minimize(
                _objective_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res_final.fun < best_val and np.isfinite(res_final.fun):
                best_val = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
