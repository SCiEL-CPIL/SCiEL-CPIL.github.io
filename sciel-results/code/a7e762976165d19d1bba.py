"""Root-structure parameterization: directly target small sign-changing roots."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, brentq

# Precompute exact Hermite polynomial coefficients
# H_n(x) for n = 0, 4, 8, 12
# We'll use numpy polynomial coefficients (highest degree first)

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n, highest degree first."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    # Recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}
    h_prev = _hermite_coeffs(n - 1)
    h_curr = np.array([2.0 * 1.0, 0.0])  # H_1
    if n == 2:
        # H_2 = 2x*H_1 - 2*H_0 = 2x*(2x) - 2*1 = 4x^2 - 2
        return np.array([4.0, 0.0, -2.0])
    # General recurrence
    h0 = np.array([1.0])  # H_0
    h1 = np.array([2.0, 0.0])  # H_1
    for k in range(2, n + 1):
        # H_k = 2x*H_{k-1} - 2(k-1)*H_{k-2}
        # 2x*H_{k-1}: shift left by 1 and multiply by 2
        term1 = np.zeros(len(h1) + 1)
        term1[:-1] = 2.0 * h1
        # 2(k-1)*H_{k-2}
        term2 = np.zeros(len(h0) + 1)
        term2[:-1] = 2.0 * (k - 1) * h0
        # Pad to same length
        max_len = max(len(term1), len(term2))
        term1_padded = np.zeros(max_len)
        term2_padded = np.zeros(max_len)
        term1_padded[:len(term1)] = term1
        term2_padded[:len(term2)] = term2
        h_new = term1_padded - term2_padded
        h0, h1 = h1, h_new
    return h1

# Precompute
H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)

# Values at 0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 2032128.0

def build_q_coeffs(c0, c1, c2):
    """Build q(x) = g(x)/x^2 coefficients (highest degree first)."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g is degree 12, q is degree 10
    # Build g coefficients (length 13, degrees 12..0)
    deg_max = 12
    g = np.zeros(deg_max + 1)
    for c, hc in [(c0, H0_c), (c1, H4_c), (c2, H8_c), (c3, H12_c)]:
        pad = np.zeros(deg_max + 1 - len(hc))
        hc_pad = np.concatenate([pad, hc])
        g += c * hc_pad
    
    # q = g / x^2: remove last 2 coefficients (x^0 and x^1 terms)
    # g[0] is x^12, ..., g[12] is x^0
    # q[0] is x^10, ..., q[10] is x^0
    q = g[:-2]
    return q

def find_largest_sign_change_root(q_coeffs):
    """Find the largest real root of q where q changes sign."""
    roots = np.roots(q_coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    valid = []
    for r in real_roots:
        eps = 1e-7 * max(1.0, abs(r))
        ql = np.polyval(q_coeffs, r - eps)
        qr = np.polyval(q_coeffs, r + eps)
        if ql * qr < 0:
            valid.append(r)
    
    if not valid:
        return None
    return max(valid)

def compute_value(c0, c1, c2):
    """Compute r^2 / (2*pi) where r is largest sign-changing root of q."""
    q = build_q_coeffs(c0, c1, c2)
    r = find_largest_sign_change_root(q)
    if r is None:
        return float('inf')
    return r * r / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s
    
    seed = [-113.0/100.0, 1.0/25.0, 1.0/3240.0]
    best_val = compute_value(*seed)
    best_params = np.array(seed)
    
    # Strategy: For a target root r*, force q(r*) = 0.
    # This is one linear constraint on (c0, c1, c2) [since c3 is determined].
    # So we have a 2-DOF plane. On this plane, we search to minimize r*
    # while ensuring r* is the largest sign-changing root.
    
    # q(r*) = 0 is linear in (c0, c1, c2, c3) and c3 is linear in (c0,c1,c2),
    # so it's linear in (c0, c1, c2).
    # Let's set up: for each (c0, c1, c2), q(r*) = sum of coefficients * (r*)^power
    # We can express q(r*) as a linear function of (c0, c1, c2).
    
    # q(r*) = [c0*H0(r*) + c1*H4(r*) + c2*H8(r*) + c3*H12(r*)] / (r*)^2
    # With c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0)/H12_0
    # So q(r*) = [c0*(H0(r*) - H0_0*H12(r*)/H12_0) + c1*(H4(r*) - H4_0*H12(r*)/H12_0) + c2*(H8(r*) - H8_0*H12(r*)/H12_0)] / (r*)^2
    # For q(r*) = 0, we need the numerator = 0 (assuming r* != 0).
    # Let A(r*) = H0(r*) - H0_0*H12(r*)/H12_0
    #     B(r*) = H4(r*) - H4_0*H12(r*)/H12_0
    #     C(r*) = H8(r*) - H8_0*H12(r*)/H12_0
    # Then A*c0 + B*c1 + C*c2 = 0.
    
    def hermite_eval(n, x):
        """Evaluate H_n(x) using precomputed coefficients."""
        coeffs = {0: H0_c, 4: H4_c, 8: H8_c, 12: H12_c}[n]
        return np.polyval(coeffs, x)
    
    def constraint_coeffs(r_star):
        """Return (A, B, C) such that A*c0 + B*c1 + C*c2 = 0 forces q(r_star) = 0."""
        if abs(r_star) < 1e-12:
            return None
        H0_r = hermite_eval(0, r_star)
        H4_r = hermite_eval(4, r_star)
        H8_r = hermite_eval(8, r_star)
        H12_r = hermite_eval(12, r_star)
        A = H0_r - H0_0 * H12_r / H12_0
        B = H4_r - H4_0 * H12_r / H12_0
        C = H8_r - H8_0 * H12_r / H12_0
        return (A, B, C)
    
    # For a given r*, the plane A*c0 + B*c1 + C*c2 = 0 is 2-DOF.
    # We parameterize this plane and search for the point that minimizes
    # the largest sign-changing root (which should be r* if we're in the right region).
    
    # Multi-start: try many target r* values, and for each, do local optimization
    # on the 2-DOF plane to minimize the actual value.
    
    np.random.seed(12345)
    
    # Candidate r* values: small positive values
    r_targets = np.concatenate([
        np.linspace(0.05, 1.0, 30),
        np.linspace(1.0, 3.0, 20),
        np.linspace(3.0, 6.0, 15),
        np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.5, 2.0, 4.0, 5.0])
    ])
    
    for r_star in r_targets:
        if time.time() > deadline - 15:
            break
        
        cc = constraint_coeffs(r_star)
        if cc is None:
            continue
        A, B, C = cc
        
        # Parameterize the plane: find two basis vectors for the null space of [A, B, C]
        normal = np.array([A, B, C])
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-12:
            continue
        normal = normal / norm_n
        
        # Find two orthonormal vectors perpendicular to normal
        # Pick an arbitrary vector not parallel to normal
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(normal, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        v1 = ref - np.dot(normal, ref) * normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        # Any point on the plane: p = t1*v1 + t2*v2 (passing through origin)
        # But we also want to explore points offset from origin.
        # Actually the plane passes through origin (it's homogeneous), so all points are t1*v1 + t2*v2.
        
        # Objective: compute_value(t1*v1 + t2*v2)
        def obj_2d(t):
            c0, c1, c2 = t[0]*v1[0] + t[1]*v2[0], t[0]*v1[1] + t[1]*v2[1], t[0]*v1[2] + t[1]*v2[2]
            return compute_value(c0, c1, c2)
        
        # Multi-start on this 2D plane
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            t0 = np.random.normal(0, 1.0, 2)
            try:
                res = minimize(obj_2d, t0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    c0, c1, c2 = res.x[0]*v1[0] + res.x[1]*v2[0], res.x[0]*v1[1] + res.x[1]*v2[1], res.x[0]*v1[2] + res.x[1]*v2[2]
                    best_params = np.array([c0, c1, c2])
            except Exception:
                pass
    
    # Also try direct multi-start Nelder-Mead in 3D from various points
    if time.time() < deadline - 20:
        for _ in range(30):
            if time.time() > deadline - 20:
                break
            x0 = np.random.normal(0, 0.5, 3)
            try:
                res = minimize(lambda p: compute_value(*p), x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Polish the best result with high precision
    if time.time() < deadline - 10 and best_val < compute_value(*seed):
        try:
            res = minimize(lambda p: compute_value(*p), best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
