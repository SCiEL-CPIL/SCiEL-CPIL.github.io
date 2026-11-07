"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _hermite_coeffs(n, max_degree=16):
    """Return coefficients of physicists' Hermite polynomial H_n as a list
    where index i is the coefficient of x^i (ascending order)."""
    # H_0 = 1
    # H_{n+1} = 2x H_n - 2n H_{n-1}
    # We work in ascending order: coeffs[i] multiplies x^i
    H_prev = [1.0] + [0.0] * (max_degree)  # H_0
    if n == 0:
        return H_prev[:n+1]
    H_curr = [0.0] * (max_degree + 1)
    # H_1 = 2x
    H_curr[1] = 2.0
    if n == 1:
        return H_curr[:n+1]
    for k in range(1, n):
        H_next = [0.0] * (max_degree + 1)
        # 2x * H_curr
        for i in range(max_degree):
            H_next[i+1] += 2.0 * H_curr[i]
        # -2k * H_prev
        for i in range(max_degree + 1):
            H_next[i] -= 2.0 * k * H_prev[i]
        H_prev = H_curr
        H_curr = H_next
    return H_curr[:n+1]


def _q_coeffs(c0, c1, c2):
    """
    Build the polynomial q(x) = g(x)/x^2 where
    g = c0*H0 + c1*H4 + c2*H8 + c3*H12, with c3 chosen so g(0)=0.
    Since H0(0)=1, H4(0)=-12, H8(0)=1008, H12(0)=-1032192 (need to verify).
    Actually let's compute H_n(0) from our coeff arrays.
    """
    H0 = _hermite_coeffs(0, 16)
    H4 = _hermite_coeffs(4, 16)
    H8 = _hermite_coeffs(8, 16)
    H12 = _hermite_coeffs(12, 16)
    
    # H_n(0) = H_n[0]
    H0_0 = H0[0]
    H4_0 = H4[0]
    H8_0 = H8[0]
    H12_0 = H12[0]
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = np.zeros(17)
    g += c0 * np.array(H0 + [0.0]*(17-len(H0)))
    g += c1 * np.array(H4 + [0.0]*(17-len(H4)))
    g += c2 * np.array(H8 + [0.0]*(17-len(H8)))
    g += c3 * np.array(H12 + [0.0]*(17-len(H12)))
    
    # q(x) = g(x) / x^2, so q_coeffs[i] = g[i+2]
    q = np.zeros(15)  # degree up to 12-2=10, but H12 has degree 12, so q has degree 10
    for i in range(2, 17):
        if i-2 < 15:
            q[i-2] = g[i]
    return q


def _largest_sign_changing_root(q_coeffs):
    """
    Find the largest real root of q at which q changes sign.
    q_coeffs is in ascending order.
    """
    # Convert to numpy polynomial (ascending order)
    # Find all real roots
    try:
        roots = np.roots(q_coeffs[::-1])  # np.roots expects descending order
    except Exception:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 0:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # For each positive real root, check if q changes sign there
    # A root changes sign if it has odd multiplicity
    # We can check by evaluating q just before and after the root
    best_root = None
    for r in real_roots:
        # Check sign change: evaluate q at r-eps and r+eps
        eps = 1e-8 * max(1.0, abs(r))
        q_before = np.polyval(q_coeffs[::-1], r - eps)
        q_after = np.polyval(q_coeffs[::-1], r + eps)
        if q_before * q_after < 0:
            if best_root is None or r > best_root:
                best_root = r
    
    return best_root


def _value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for the given coefficients."""
    q = _q_coeffs(c0, c1, c2)
    # Remove leading zeros
    while len(q) > 1 and abs(q[-1]) < 1e-15:
        q = q[:-1]
    
    r = _largest_sign_changing_root(q)
    if r is None or r <= 0:
        return np.inf
    
    # Check if g tends to negative limit at infinity
    # The leading term of g is c3 * H12's leading term
    # H12 leading coeff is 2^12 = 4096
    # So g ~ c3 * 4096 * x^12 as x->inf
    # If c3 < 0, g -> -inf, so we'd flip g to -g, which flips c3 sign
    # Actually, let's just compute c3 and check
    H0 = _hermite_coeffs(0, 16)
    H4 = _hermite_coeffs(4, 16)
    H8 = _hermite_coeffs(8, 16)
    H12 = _hermite_coeffs(12, 16)
    H0_0 = H0[0]
    H4_0 = H4[0]
    H8_0 = H8[0]
    H12_0 = H12[0]
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Leading coeff of H12 is 2^12
    if c3 * (2**12) < 0:
        # g -> -inf, so replace g by -g, which means r stays the same (roots are same)
        # Actually the sign change condition: if we flip g, q flips, so sign changes flip
        # But the largest sign-changing root should be the same set of roots
        # The value is r^2/(2*pi) regardless
        pass
    
    return r**2 / (2.0 * np.pi)


def construct():
    """
    Search for coefficients [c0, c1, c2] that minimize the value.
    Use differential evolution on a 2D space (fix scale via c0=1) then polish.
    """
    start_time = time.time()
    deadline = start_time + 200  # 200s out of 210s limit
    
    seed_val = _value(-113/100, 1/25, 1/3240)
    best_c = np.array([-113/100, 1/25, 1/3240])
    best_val = seed_val
    
    def objective(params):
        """Minimize value with c0 fixed to 1 (scale invariance)."""
        c1, c2 = params
        v = _value(1.0, c1, c2)
        if not np.isfinite(v):
            return 1e10
        return v
    
    # Differential evolution in 2D
    bounds = [(-10, 10), (-10, 10)]
    
    try:
        de_result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15,
            seed=42, tol=1e-8, polish=False,
            callback=None
        )
        if time.time() > deadline - 60:
            # Check if DE result is better
            v_de = objective(de_result.x)
            if np.isfinite(v_de) and v_de < best_val:
                best_val = v_de
                best_c = np.array([1.0, de_result.x[0], de_result.x[1]])
            return best_c
    except Exception:
        pass
    
    # Nelder-Mead polish from DE result
    if time.time() < deadline - 30:
        try:
            nm_result = minimize(
                objective, de_result.x, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            v_nm = objective(nm_result.x)
            if np.isfinite(v_nm) and v_nm < best_val:
                best_val = v_nm
                best_c = np.array([1.0, nm_result.x[0], nm_result.x[1]])
        except Exception:
            pass
    
    # BFGS polish
    if time.time() < deadline - 10:
        try:
            bfgs_result = minimize(
                objective, best_c[1:], method='BFGS',
                options={'gtol': 1e-12, 'maxiter': 200}
            )
            v_bfgs = objective(bfgs_result.x)
            if np.isfinite(v_bfgs) and v_bfgs < best_val:
                best_val = v_bfgs
                best_c = np.array([1.0, bfgs_result.x[0], bfgs_result.x[1]])
        except Exception:
            pass
    
    # Multi-start local optimization
    if time.time() < deadline - 5:
        n_starts = 20
        rng = np.random.RandomState(123)
        for i in range(n_starts):
            if time.time() > deadline - 2:
                break
            init = rng.uniform(-5, 5, size=2)
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
                v = objective(res.x)
                if np.isfinite(v) and v < best_val:
                    best_val = v
                    best_c = np.array([1.0, res.x[0], res.x[1]])
            except Exception:
                pass
    
    return best_c
# EVOLVE-BLOCK-END
