"""Direct analytic approach: build Hermite-Gaussian test function to minimize uncertainty product."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195  # safety margin before 210s kill


def _hermite_coeffs():
    """
    Precompute the polynomial coefficients of H_0, H_4, H_8, H_12.
    H_n are physicists' Hermite polynomials.
    H_0 = 1
    H_4 = x^4 - 6x^2 + 3
    H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    We represent P(x) = sum_{k=0}^{6} p_k * x^{2k}
    H_0: p = [1, 0, 0, 0, 0, 0, 0]
    H_4: p = [3, -6, 1, 0, 0, 0, 0]
    H_8: p = [105, -420, 210, -28, 1, 0, 0]
    H_12: p = [945, -7560, 13860, -6930, 1155, -66, 1]
    """
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    """
    Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
    for n = 0, 1, ..., 12 (need up to x^{24} since max degree is 12, |P|^2 has degree 24)
    """
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        if n == 0:
            I[n] = sp
        else:
            # Recurrence: I[n] = I[n-1] * (2n-1) / (4*n)
            I[n] = I[n-1] * (2.0*n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()


def _bound(params):
    """
    Compute A(f)*A(F) for g(x) = P(x) * exp(-x^2/2)
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 = -(a0+3a4+105a8)/945
    
    A(f)^2 = <x^2>_<1> = m2/m0 where m_k = int x^k |g|^2 dx
    A(F)^2 = n2/n0 where n_k = int xi^k |G|^2 dx
    
    G(xi) = Fourier of g. For g(x) = P(x)e^{-x^2/2}:
    G(xi) = sqrt(2*pi) * P(-i*xi) * e^{-xi^2/2}
    |G|^2 = 2*pi * |P(-i*xi)|^2 * e^{-xi^2}
    P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k} = Q(xi)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # P coefficients: p_k = sum_{m=0}^{3} a_m * H[m, k]
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H  # shape (7,)
    
    # m_k = int x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    # Q coefficients: q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # n_k = 2*pi * int xi^{2k} |Q|^2 e^{-xi^2} dx
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i+j]
            n2 += 2.0 * np.pi * qij * _I[1+i+j]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_grad(params):
    """Compute bound and its gradient analytically."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    # Compute m0, m2
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i+j]
            m2 += pij * _I[1+i+j]
    
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
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
    
    # Gradient: d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    # da12/da_c for c = a0, a4, a8:
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c = H[c,k] + da12[c] * H[3,k]
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        # dm0/da_c = 2 * sum_i dp_i * sum_j p_j * I[i+j]
        # dm2/da_c = 2 * sum_i dp_i * sum_j p_j * I[1+i+j]
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                s0 += p[j] * _I[i+j]
                s2 += p[j] * _I[1+i+j]
            dm0 += 2.0 * dpi * s0
            dm2 += 2.0 * dpi * s2
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
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
    return _bound(params)


def _objective_grad(params):
    return _bound_grad(params)


def construct():
    """
    Hybrid approach: variational quantum-inspired ansatz + SPSA + L-BFGS-B polish.
    
    Strategy:
    1. Map Hermite coefficients to a 2-parameter quantum circuit (rotation gates)
    2. Use SPSA to navigate non-convex landscape without explicit gradients
    3. Polish best SPSA results with L-BFGS-B using exact gradients
    4. Combine with direct multi-start L-BFGS-B as fallback
    """
    from scipy.optimize import minimize
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # --- Phase 1: Variational Quantum-Inspired Ansatz with SPSA ---
    # Map (a0, a4, a8) to 2 quantum parameters (theta, phi) via rotation gates
    # a0 = A * cos(theta) * cos(phi)
    # a4 = B * sin(theta) * cos(phi)  
    # a8 = C * sin(phi)
    # This creates a non-linear parameterization that may find different optima
    
    def quantum_to_hermite(theta, phi):
        """Map 2 quantum parameters to 3 Hermite coefficients."""
        A, B, C = 1.5, 0.15, 0.005
        a0 = A * np.cos(theta) * np.cos(phi)
        a4 = B * np.sin(theta) * np.cos(phi)
        a8 = C * np.sin(phi)
        return np.array([a0, a4, a8])
    
    def spsa_optimize(theta0, phi0, n_iter=5000, c=0.05, a=0.1, gamma=0.6, alpha=0.6):
        """SPSA optimization on the quantum parameter space."""
        rng = np.random.default_rng(12345)
        theta, phi = theta0, phi0
        best_t, best_p = theta, phi
        best_f = _bound(quantum_to_hermite(theta, phi))
        
        for k in range(n_iter):
            if time.time() > _DEADLINE - 60:
                break
            # SPSA step size
            c_k = c / (k + 1) ** gamma
            a_k = a / (k + 1) ** alpha
            
            # Random perturbation
            delta = rng.choice([-1, 1], size=2)
            
            # Evaluate at x +/- c*delta
            t_plus = np.array([theta + c_k * delta[0], phi + c_k * delta[1]])
            t_minus = np.array([theta - c_k * delta[0], phi - c_k * delta[1]])
            
            f_plus = _bound(quantum_to_hermite(t_plus[0], t_plus[1]))
            f_minus = _bound(quantum_to_hermite(t_minus[0], t_minus[1]))
            
            # Gradient estimate
            grad_est = (f_plus - f_minus) / (2.0 * c_k) * delta
            
            # Update
            theta -= a_k * grad_est[0]
            phi -= a_k * grad_est[1]
            
            # Track best
            f_curr = _bound(quantum_to_hermite(theta, phi))
            if f_curr < best_f:
                best_f = f_curr
                best_t, best_p = theta, phi
        
        return best_t, best_p, best_f
    
    # Run SPSA from multiple initial quantum parameters
    if time.time() < _DEADLINE - 80:
        quantum_starts = [
            (0.5, 0.3), (1.0, 0.5), (1.5, 0.8), (2.0, 1.0),
            (0.3, 0.2), (0.8, 0.4), (1.2, 0.6), (1.8, 0.9),
            (0.1, 0.1), (2.5, 1.2), (0.7, 0.7), (1.3, 0.3),
        ]
        
        for qt, qp in quantum_starts:
            if time.time() > _DEADLINE - 70:
                break
            try:
                bt, bp, bf = spsa_optimize(qt, qp, n_iter=3000)
                if bf < best_val and np.isfinite(bf):
                    best_val = bf
                    best_params = quantum_to_hermite(bt, bp)
            except Exception:
                pass
    
    # --- Phase 2: Direct multi-start L-BFGS-B (fallback + complement) ---
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
    ]
    
    # Add random starts
    for seed in [42, 123, 456, 789]:
        rng = np.random.default_rng(seed)
        for _ in range(5):
            starts.append(rng.normal([0, 0.04, 0.0005], [1.0, 0.1, 0.002]))
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # --- Phase 3: Coarse grid refinement ---
    if time.time() < _DEADLINE - 30:
        center = best_params.copy()
        scales = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        for scale in scales:
            if time.time() > _DEADLINE - 15:
                break
            for axis in range(3):
                for direction in [-1, 1]:
                    if time.time() > _DEADLINE - 15:
                        break
                    trial = center.copy()
                    trial[axis] += direction * scale
                    try:
                        val = _bound(trial)
                        if val < best_val:
                            best_val = val
                            best_params = trial.copy()
                            res = minimize(_objective_grad, trial, method='L-BFGS-B',
                                          jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
                            if res.fun < best_val and np.isfinite(res.fun):
                                best_val = res.fun
                                best_params = res.x.copy()
                    except Exception:
                        pass
    
    # --- Phase 4: High-precision polish ---
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-22, 'gtol': 1e-20, 'maxls': 500, 'factr': 1e10, 'pgtol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-20, 'fatol': 1e-22, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final ultra-precise L-BFGS-B
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100000, 'ftol': 1e-24, 'gtol': 1e-22, 'maxls': 1000})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
