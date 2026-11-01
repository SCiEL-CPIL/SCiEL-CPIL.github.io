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
    Stochastic local search on normalized coefficient space.
    
    Reparameterize using c = (c0, c4, c8) with |c| = 1, then map to
    (a0, a4, a8) via a scale factor. The vanishing condition P(0)=0 fixes
    a12 = -(a0 + 3*a4 + 105*a8)/945.
    
    Strategy:
    1. Multi-start on the unit sphere in (c0, c4, c8) space
    2. Adaptive step-size stochastic local search
    3. Final high-precision polish with gradient-based refinement
    """
    from scipy.optimize import minimize
    
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(12345)
    
    def _normalize_and_scale(c, scale):
        """Convert normalized direction c to actual params (a0, a4, a8)."""
        nrm = np.linalg.norm(c)
        if nrm < 1e-15:
            return np.zeros(3)
        return scale * c / nrm
    
    def _random_direction():
        """Generate random unit vector in 3D."""
        v = rng.normal(size=3)
        nrm = np.linalg.norm(v)
        if nrm < 1e-15:
            return np.array([1.0, 0.0, 0.0])
        return v / nrm
    
    def _spherical_step(c, step_size):
        """Take a small step on the sphere from direction c."""
        # Generate random tangent direction
        rand_v = rng.normal(size=3)
        # Project onto tangent space (perpendicular to c)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return rand_v / np.linalg.norm(rand_v)
        c_unit = c / c_norm
        tangent = rand_v - np.dot(rand_v, c_unit) * c_unit
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-15:
            # Pick a perpendicular vector
            if abs(c_unit[0]) < 0.9:
                perp = np.cross(c_unit, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(c_unit, np.array([0.0, 1.0, 0.0]))
            perp = perp / np.linalg.norm(perp)
            tangent = perp
            t_norm = 1.0
        tangent = tangent / t_norm
        # New direction: rotate c toward tangent by step_size
        new_c = c * np.cos(step_size) + tangent * np.sin(step_size)
        new_c = new_c / np.linalg.norm(new_c)
        return new_c
    
    # Phase 1: Coarse multi-start on sphere with various scales
    # The seed has |a| ≈ sqrt(1.13^2 + 0.04^2 + 0.000308642^2) ≈ 1.13
    seed_scale = np.linalg.norm(np.array([-1.13, 0.04, 0.000308642]))
    seed_dir = np.array([-1.13, 0.04, 0.000308642]) / seed_scale
    
    # Try multiple scales
    scales_to_try = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    
    for scale in scales_to_try:
        if time.time() > _DEADLINE - 30:
            break
        # Multiple random directions for this scale
        n_starts = 30 if scale in [0.8, 1.0, 1.2] else 15
        for _ in range(n_starts):
            if time.time() > _DEADLINE - 30:
                break
            c = _random_direction()
            # Bias toward seed direction with some probability
            if rng.random() < 0.3:
                c = seed_dir * np.cos(0.3) + _random_direction() * np.sin(0.3)
                c = c / np.linalg.norm(c)
            
            params = _normalize_and_scale(c, scale)
            try:
                val = _bound(params)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = params.copy()
            except Exception:
                pass
    
    # Phase 2: Stochastic local search with adaptive steps
    # Work in (direction, scale) space
    if time.time() < _DEADLINE - 40:
        # Convert best params back to (direction, scale)
        best_scale = np.linalg.norm(best_params)
        if best_scale < 1e-15:
            best_scale = 1.0
            best_dir = np.array([1.0, 0.0, 0.0])
        else:
            best_dir = best_params / best_scale
            best_dir = best_dir / np.linalg.norm(best_dir)
        
        current_dir = best_dir.copy()
        current_scale = best_scale
        current_val = best_val
        
        # Adaptive step sizes
        dir_step = 0.1  # radians on sphere
        scale_step = 0.05  # relative change in scale
        
        # Local search iterations
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > _DEADLINE - 10:
                break
            
            # Try multiple perturbations
            for trial in range(4):
                if time.time() > _DEADLINE - 10:
                    break
                
                # Perturb direction
                new_dir = _spherical_step(current_dir, dir_step * (0.5 + rng.random()))
                # Perturb scale
                new_scale = current_scale * (1.0 + rng.normal(0, scale_step))
                if new_scale < 0.01:
                    new_scale = 0.01
                
                new_params = _normalize_and_scale(new_dir, new_scale)
                try:
                    new_val = _bound(new_params)
                    if new_val < current_val and np.isfinite(new_val):
                        current_dir = new_dir
                        current_scale = new_scale
                        current_val = new_val
                        # Adaptive: increase step on success
                        dir_step = min(dir_step * 1.1, 0.5)
                        scale_step = min(scale_step * 1.1, 0.2)
                        if current_val < best_val:
                            best_val = current_val
                            best_params = new_params.copy()
                        break
                    else:
                        # Decrease step on failure
                        dir_step = max(dir_step * 0.8, 0.001)
                        scale_step = max(scale_step * 0.8, 0.005)
                except Exception:
                    pass
            
            # Occasional restart from best with larger step
            if it % 100 == 0 and it > 0:
                if current_val > best_val * 1.01:
                    current_dir = best_dir.copy()
                    current_scale = best_scale
                    current_val = best_val
                    dir_step = 0.2
                    scale_step = 0.1
    
    # Phase 3: Final high-precision polish with gradient-based methods
    if time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Additional local search from polished point
    if time.time() < _DEADLINE - 5:
        polished_scale = np.linalg.norm(best_params)
        if polished_scale > 1e-15:
            polished_dir = best_params / polished_scale
            polished_dir = polished_dir / np.linalg.norm(polished_dir)
        else:
            polished_dir = np.array([1.0, 0.0, 0.0])
            polished_scale = 1.0
        
        d_step = 0.01
        s_step = 0.01
        for it in range(2000):
            if time.time() > _DEADLINE - 3:
                break
            for trial in range(3):
                if time.time() > _DEADLINE - 3:
                    break
                new_dir = _spherical_step(polished_dir, d_step * (0.5 + rng.random()))
                new_scale = polished_scale * (1.0 + rng.normal(0, s_step))
                if new_scale < 0.01:
                    new_scale = 0.01
                new_params = _normalize_and_scale(new_dir, new_scale)
                try:
                    new_val = _bound(new_params)
                    if new_val < best_val and np.isfinite(new_val):
                        best_val = new_val
                        best_params = new_params.copy()
                        polished_dir = new_params / np.linalg.norm(new_params)
                        polished_scale = np.linalg.norm(new_params)
                        d_step = min(d_step * 1.05, 0.1)
                        s_step = min(s_step * 1.05, 0.1)
                        break
                    else:
                        d_step = max(d_step * 0.9, 0.0001)
                        s_step = max(s_step * 0.9, 0.001)
                except Exception:
                    pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
