"""Uncertainty inequality optimization with analytic gradient and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, basin_hopping

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill

def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I

_I = _precompute_I()

# Precompute the matrix D where p_k = sum_c D[k,c] a_c
# p[0] = 0 (constraint P(0)=0 is enforced by a12, so p[0] is always 0 in the optimized space? 
# No, p[0] is the constant term of the polynomial P(x). 
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12.
# P(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0 => a12 = -(a0+3a4+105a8)/945.
# The polynomial coefficients p_k are the coefficients of x^(2k) in P(x).
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients for x^(2k) in H_m:
# H0: [1]
# H4: [3, -6, 1] (k=0,1,2)
# H8: [105, -420, 210, -28, 1] (k=0..4)
# H12: [945, -7560, 13860, -6930, 1155, -66, 1] (k=0..6)

# D[k, c] is coefficient of a_c in p_k.
# a = [a0, a4, a8]
# a12 = -(1/945)*a0 - (3/945)*a4 - (105/945)*a8

# p_k = c_k(a0)*a0 + c_k(a4)*a4 + c_k(a8)*a8
# c_k(a0) = coeff of a0 in p_k = coeff in H0 + coeff in H4*a4? No.
# p_k = (H0_k)*a0 + (H4_k)*a4 + (H8_k)*a8 + (H12_k)*a12
# a12 = -1/945 * a0 - 1/315 * a4 - 1/9 * a8

# D[k, 0] (coeff of a0) = H0_k + H12_k * (-1/945)
# D[k, 1] (coeff of a4) = H4_k + H12_k * (-1/315)
# D[k, 2] (coeff of a8) = H8_k + H12_k * (-1/9)

H0 = np.zeros(7); H0[0] = 1.0
H4 = np.zeros(7); H4[0] = 3.0; H4[1] = -6.0; H4[2] = 1.0
H8 = np.zeros(7); H8[0] = 105.0; H8[1] = -420.0; H8[2] = 210.0; H8[3] = -28.0; H8[4] = 1.0
H12 = np.zeros(7); H12[0] = 945.0; H12[1] = -7560.0; H12[2] = 13860.0; H12[3] = -6930.0; H12[4] = 1155.0; H12[5] = -66.0; H12[6] = 1.0

D = np.zeros((7, 3))
D[:, 0] = H0 + H12 * (-1.0 / 945.0)
D[:, 1] = H4 + H12 * (-3.0 / 945.0)
D[:, 2] = H8 + H12 * (-105.0 / 945.0)

def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(vf * vF) and its analytic gradient.
    """
    a = np.array([a0, a4, a8])
    p = D @ a
    
    # q is related to p by Fourier transform properties.
    # For Hermite polynomials, the Fourier transform of H_n(x) e^{-x^2/2} is i^n H_n(x) e^{-x^2/2}.
    # Here we have P(x) e^{-x^2/2}. 
    # The task says "Fourier-invariant". 
    # The seed program uses q = p * [1, -1, 1, -1, 1, -1, 1].
    # This corresponds to i^{2k} = (-1)^k.
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Compute m_k and n_k for k=0, 2
    # m_k = sum_{i,j} p_i p_j I_{k+i+j}
    # n_k = 2*pi * sum_{i,j} q_i q_j I_{k+i+j}
    
    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0
    
    # Precompute sums for efficiency?
    # m_k = p^T M_k p where M_k[i,j] = I_{k+i+j}
    # But p is small (7 elements). Direct sum is fine.
    
    for i in range(7):
        if p[i] == 0: continue
        for j in range(7):
            if p[j] == 0: continue
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            
            idx0 = i + j
            if idx0 < 25:
                m0 += pij * _I[idx0]
                n0 += qij * _I[idx0]
            
            idx2 = 2 + i + j
            if idx2 < 25:
                m2 += pij * _I[idx2]
                n2 += qij * _I[idx2]
                
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Gradient
    # J = sqrt(m2*n2 / (m0*n0))
    # ln J = 0.5 * (ln m2 + ln n2 - ln m0 - ln n0)
    # dJ/da = 0.5 * J * (dm2/m2 - dm0/m0 + dn2/n2 - dn0/n0)
    
    # Compute dm_k/da_c
    # dm_k/da_c = sum_{i,j} (D[i,c] p[j] + p[i] D[j,c]) I_{k+i+j}
    # = 2 * sum_i D[i,c] (sum_j p[j] I_{k+i+j})
    
    dm0_da = np.zeros(3)
    dm2_da = np.zeros(3)
    dn0_da = np.zeros(3)
    dn2_da = np.zeros(3)
    
    for c in range(3):
        # Sum_j p[j] I_{k+i+j} for each i
        # Let S_k[i] = sum_j p[j] I_{k+i+j}
        S0 = np.zeros(7)
        S2 = np.zeros(7)
        for i in range(7):
            for j in range(7):
                idx0 = i + j
                if idx0 < 25:
                    S0[i] += p[j] * _I[idx0]
                idx2 = 2 + i + j
                if idx2 < 25:
                    S2[i] += p[j] * _I[idx2]
        
        dm0_da[c] = 2.0 * np.dot(D[:, c], S0)
        dm2_da[c] = 2.0 * np.dot(D[:, c], S2)
        
        # Same for n, but with q
        T0 = np.zeros(7)
        T2 = np.zeros(7)
        for i in range(7):
            for j in range(7):
                idx0 = i + j
                if idx0 < 25:
                    T0[i] += q[j] * _I[idx0]
                idx2 = 2 + i + j
                if idx2 < 25:
                    T2[i] += q[j] * _I[idx2]
        
        # dn_k/da_c = 2*pi * 2 * sum_i D[i,c] T_k[i]
        dn0_da[c] = 4.0 * np.pi * np.dot(D[:, c], T0)
        dn2_da[c] = 4.0 * np.pi * np.dot(D[:, c], T2)
        
    grad = np.zeros(3)
    for c in range(3):
        grad[c] = 0.5 * bound * (dm2_da[c]/m2 - dm0_da[c]/m0 + dn2_da[c]/n2 - dn0_da[c]/n0)
        
    return bound, grad

def _objective(params):
    bound, _ = _compute_bound_and_grad(params[0], params[1], params[2])
    return bound

def _objective_grad(params):
    return _compute_bound_and_grad(params[0], params[1], params[2])

def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start L-BFGS-B with diverse initial points
    # Seed point
    starts = [np.array([-1.13, 0.04, 0.000308642])]
    
    # Random starts in a reasonable range
    for _ in range(50):
        s = np.array([
            rng.uniform(-3.0, 2.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.01, 0.01)
        ])
        starts.append(s)
        
    # Grid starts
    for a0 in np.linspace(-2.5, 1.5, 10):
        for a4 in np.linspace(-0.3, 0.3, 5):
            for a8 in np.linspace(-0.005, 0.005, 3):
                starts.append(np.array([a0, a4, a8]))
                
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Basin Hopping from the best point found
    if time.time() < _DEADLINE - 30:
        try:
            def step(x):
                return x + rng.normal(0, 0.1, 3)
            
            bh_res = basin_hopping(_objective_grad, best_params, niter=100, 
                                   stepfun=step, T=1.0, minstep=1e-4, 
                                   take_step=True, interval=10, 
                                   disp=False, 
                                   minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 
                                                      'options': {'maxiter': 1000, 'ftol': 1e-15}})
            if bh_res.fun < best_val and np.isfinite(bh_res.fun):
                best_val = bh_res.fun
                best_params = bh_res.x.copy()
        except Exception:
            pass
            
    # Phase 3: High-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            # Use a very tight tolerance
            res = minimize(_objective_grad, best_params, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
                
            # Nelder-Mead polish for final touch
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
            
    return np.array(best_params)
# EVOLVE-BLOCK-END
