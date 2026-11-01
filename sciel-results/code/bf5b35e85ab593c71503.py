import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

START_TIME = time.time()
TIME_LIMIT = 205.0

# Hermite polynomial coefficients in power basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]
_H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8 = np.array([105.0, -420.0, 210.0, 0.0, 1.0, 0.0, 0.0])
_H12 = np.array([945.0, -226800.0, 151200.0, -30240.0, 2520.0, -84.0, 1.0])

# Precompute integral table: I[n] = integral x^{2n} exp(-x^2) dx from -inf to inf
_I_table = np.zeros(30)
_I_table[0] = np.sqrt(np.pi)
for _n in range(1, 30):
    _I_table[_n] = _I_table[_n-1] * (2*_n - 1) / 2.0

# Precompute dP/dc vectors (constant)
_dP_dc0 = _H0 - (1.0/945.0)*_H12
_dP_dc4 = _H4 - (3.0/945.0)*_H12
_dP_dc8 = _H8 - (105.0/945.0)*_H12
_dP_dc = np.stack([_dP_dc0, _dP_dc4, _dP_dc8])  # shape (3, 7)

# Precompute R derivative structure: dR[i]/dP[j]
# R[k] = 2*(k+1)*P[k+1] - P[k] for k<6, R[6] = -P[6]
# So dR[i]/dP[j]:
#   if i < 6: nonzero when j == i (coeff -1) or j == i+1 (coeff 2*(i+1))
#   if i == 6: nonzero when j == 6 (coeff -1)
_dR_dP = np.zeros((7, 7))  # dR_dP[i, j] = dR[i]/dP[j]
for i in range(6):
    _dR_dP[i, i] = -1.0
    _dR_dP[i, i+1] = 2.0*(i+1)
_dR_dP[6, 6] = -1.0

def _get_P(c0, c4, c8):
    """Get P coefficients given c0, c4, c8 with c12 derived from P(0)=0."""
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    P = c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12
    return P

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty bound sqrt(||xg||^2 * ||g'||^2 / ||g||^4)."""
    P = _get_P(c0, c4, c8)
    
    # Q[m] = coefficient of x^{2m} in P(x)^2
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    # norm_sq = sum Q[m] * I[m]
    norm_sq = np.dot(Q, _I_table[:13])
    
    # x_norm_sq = sum Q[m] * I[m+1]
    x_norm_sq = np.dot(Q, _I_table[1:14])
    
    # R[k] = coefficient of x^{2k+1} in P'(x) - x*P(x)
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    # S[m] = coefficient of x^{2m} in R(x)^2
    S = np.zeros(14)
    for i in range(7):
        for j in range(7):
            S[i+j] += R[i] * R[j]
    
    # deriv_norm_sq = sum S[m] * I[m]
    deriv_norm_sq = np.dot(S, _I_table[:14])
    
    if norm_sq <= 0:
        return 0.0
    
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    return bound

def _compute_bound_grad(c0, c4, c8):
    """Compute bound and its gradient using vectorized operations."""
    P = _get_P(c0, c4, c8)
    
    # Q[m] = sum_{i+j=m} P[i]*P[j]
    Q = np.zeros(13)
    for i in range(7):
        Q[i:i+7] += P[i] * P[:7-i]
    
    norm_sq = np.dot(Q, _I_table[:13])
    x_norm_sq = np.dot(Q, _I_table[1:14])
    
    # R[k] = 2*(k+1)*P[k+1] - P[k] for k<6, R[6] = -P[6]
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    # S[m] = sum_{i+j=m} R[i]*R[j]
    S = np.zeros(14)
    for i in range(7):
        S[i:i+7] += R[i] * R[:7-i]
    
    deriv_norm_sq = np.dot(S, _I_table[:14])
    
    if norm_sq <= 0:
        return 0.0, np.zeros(3)
    
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    if bound <= 0:
        return bound, np.zeros(3)
    
    # d(norm_sq)/dP[j] = 2 * sum_{m>=j} P[m-j] * I[m]
    # This is 2 * (P * I_conv) where I_conv[m] = I[m]
    d_norm_dP = np.zeros(7)
    for j in range(7):
        d_norm_dP[j] = 2.0 * np.dot(P[7-j:13-j+1] if 7-j <= 13-j else P[7-j:13], _I_table[7-j:13-j+7])
    # Simpler: d_norm_dP[j] = 2 * sum_{i=0}^{6} P[i] * I[i+j]
    d_norm_dP = np.array([2.0 * np.dot(P, _I_table[j:j+7]) for j in range(7)])
    
    # d(x_norm_sq)/dP[j] = 2 * sum_{i=0}^{6} P[i] * I[i+j+1]
    d_xnorm_dP = np.array([2.0 * np.dot(P, _I_table[j+1:j+8]) for j in range(7)])
    
    # d(deriv_norm_sq)/dP[j]:
    # deriv_norm_sq = sum_m S[m] * I[m] = sum_{i,j} R[i]*R[j]*I[i+j]
    # d/dP[j] = sum_{i,j} (dR[i]/dP[j]*R[j] + R[i]*dR[j]/dP[j]) * I[i+j]
    # = 2 * sum_{i,j} R[i] * dR[j]/dP[j] * I[i+j]  (by symmetry, since I[i+j]=I[j+i])
    # = 2 * sum_i R[i] * (sum_j dR[j]/dP[j] * I[i+j])
    # Let D_j = dR[j]/dP[j] (the j-th column of _dR_dP)
    # Then d(deriv_norm_sq)/dP[j] = 2 * sum_i R[i] * D_i * I[i+j]
    # where D_i = dR[i]/dP[j]
    
    # For each free parameter alpha in {c0, c4, c8}:
    # d(deriv_norm_sq)/dalpha = sum_j (dP[j]/dalpha) * d(deriv_norm_sq)/dP[j]
    # d(deriv_norm_sq)/dP[j] = 2 * sum_i R[i] * _dR_dP[i,j] * I[i+j]
    
    # Precompute: for each j, d_deriv_dP[j] = 2 * sum_i R[i] * _dR_dP[i,j] * I[i+j]
    d_deriv_dP = np.zeros(7)
    for j in range(7):
        s = 0.0
        for i in range(7):
            s += R[i] * _dR_dP[i, j] * _I_table[i+j]
        d_deriv_dP[j] = 2.0 * s
    
    # Chain rule for each free parameter
    # dA/dalpha = d_xnorm_dP @ dP_dc_alpha
    # dB/dalpha = d_deriv_dP @ dP_dc_alpha
    # dC/dalpha = d_norm_dP @ dP_dc_alpha
    
    A = x_norm_sq
    B = deriv_norm_sq
    C = norm_sq
    
    # dP_dc shape is (3, 7)
    dA_dc = np.array([np.dot(d_xnorm_dP, _dP_dc[k]) for k in range(3)])
    dB_dc = np.array([np.dot(d_deriv_dP, _dP_dc[k]) for k in range(3)])
    dC_dc = np.array([np.dot(d_norm_dP, _dP_dc[k]) for k in range(3)])
    
    # bound^2 = A*B/C^2
    # d(bound^2)/dalpha = (B*dA + A*dB - 2*A*B*dC/C)/C^2
    bound_sq = A * B / (C * C)
    if bound_sq <= 0:
        return bound, np.zeros(3)
    
    factor = 0.5 / bound
    AB_over_C2 = A * B / (C * C)
    d_bound_sq_dc = (B*dA_dc + A*dB_dc - 2.0*AB_over_C2*dC_dc) / C
    
    grad = factor * d_bound_sq_dc
    
    return bound, grad

def _objective(coeffs):
    return _compute_bound(coeffs[0], coeffs[1], coeffs[2])

def _objective_grad(coeffs):
    return _compute_bound_grad(coeffs[0], coeffs[1], coeffs[2])

def construct():
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    np.random.seed(42)
    bounds = [(-10.0, 10.0), (-5.0, 5.0), (-2.0, 2.0)]
    
    # Phase 1: Multi-start L-BFGS-B with diverse initializations
    num_starts = 120
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 30:
            break
        
        if i == 0:
            x0 = seed.copy()
        elif i == 1:
            x0 = np.array([0.0, 0.0, 0.0])
        elif i == 2:
            x0 = np.array([1.0, -0.1, 0.001])
        elif i == 3:
            x0 = np.array([-1.0, 0.1, -0.001])
        elif i == 4:
            x0 = np.array([0.5, 0.0, 0.0])
        elif i == 5:
            x0 = np.array([-0.5, 0.0, 0.0])
        elif i == 6:
            x0 = np.array([0.0, 0.5, 0.0])
        elif i == 7:
            x0 = np.array([0.0, -0.5, 0.0])
        elif i == 8:
            x0 = np.array([2.0, -0.2, 0.002])
        elif i == 9:
            x0 = np.array([-2.0, 0.2, -0.002])
        else:
            # Mix of uniform and Gaussian perturbations around seed
            if i % 3 == 0:
                x0 = np.array([
                    np.random.uniform(bounds[0][0], bounds[0][1]),
                    np.random.uniform(bounds[1][0], bounds[1][1]),
                    np.random.uniform(bounds[2][0], bounds[2][1])
                ])
            elif i % 3 == 1:
                x0 = seed + np.random.normal(0, 0.5, size=3)
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            else:
                x0 = seed + np.random.normal(0, 1.0, size=3)
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement around best solution with decreasing step
    for i in range(40):
        if time.time() > TIME_LIMIT - 15:
            break
        
        sigma = 0.2 * (1.0 - i/40.0)**2 + 0.0005
        x0 = best_coeffs + np.random.normal(0, sigma, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: 1D line searches along each coordinate for final polish
    if time.time() < TIME_LIMIT - 10:
        for coord in range(3):
            if time.time() > TIME_LIMIT - 5:
                break
            def line_obj(t, coord=coord):
                x = best_coeffs.copy()
                x[coord] = t
                return _objective(x)
            
            try:
                # Wider initial search
                lo = max(bounds[coord][0], best_coeffs[coord] - 1.0)
                hi = min(bounds[coord][1], best_coeffs[coord] + 1.0)
                res = minimize_scalar(line_obj, bounds=(lo, hi),
                                      method='bounded', options={'xatol': 1e-14})
                if res.fun < best_score:
                    new_x = best_coeffs.copy()
                    new_x[coord] = res.x
                    best_coeffs = new_x
                    best_score = _objective(best_coeffs)
            except Exception:
                pass
    
    # Phase 4: Final high-precision L-BFGS-B
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: One more line search pass after final polish
    if time.time() < TIME_LIMIT - 3:
        for coord in range(3):
            if time.time() > TIME_LIMIT - 2:
                break
            def line_obj2(t, coord=coord):
                x = best_coeffs.copy()
                x[coord] = t
                return _objective(x)
            
            try:
                lo = max(bounds[coord][0], best_coeffs[coord] - 0.1)
                hi = min(bounds[coord][1], best_coeffs[coord] + 0.1)
                res = minimize_scalar(line_obj2, bounds=(lo, hi),
                                      method='bounded', options={'xatol': 1e-15})
                if res.fun < best_score:
                    new_x = best_coeffs.copy()
                    new_x[coord] = res.x
                    best_coeffs = new_x
                    best_score = _objective(best_coeffs)
            except Exception:
                pass
    
    return best_coeffs
