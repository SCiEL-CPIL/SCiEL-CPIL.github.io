"""Structural search for coefficients c0, c1, c2 minimizing r^2/(2*pi)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 175.0
_t_start = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t_start)


def _get_q_matrix():
    """
    Returns a 6x3 matrix M such that q_coeffs = M @ [c0, c1, c2].
    q_coeffs = [b1, b2, b3, b4, b5, b6]
    """
    # b1 = g1 = -48 c1 - 40320 c2 - 5109120 c3
    # b2 = g2 = 16 c1 + 26880 c2 + 9072000 c3
    # b3 = g3 = -4608 c2 - 6005760 c3
    # b4 = g4 = 256 c2 + 1424640 c3
    # b5 = g5 = -142464 c3
    # b6 = g6 = 4096 c3
    # c3 = -(c0 + 12 c1 + 168 c2) / 18480
    
    # Let's compute M entries directly.
    # c3 = -1/18480 c0 - 12/18480 c1 - 168/18480 c2
    # c3 = -1/18480 c0 - 1/1540 c1 - 1/110 c2
    
    # b1 = -48 c1 - 40320 c2 - 5109120 (-1/18480 c0 - 1/1540 c1 - 1/110 c2)
    # b1 = 5109120/18480 c0 + (-48 + 5109120/1540) c1 + (-40320 + 5109120/110) c2
    # 5109120/18480 = 276.467... -> 276 + 4/15? Let's use fractions or high precision floats.
    # 5109120 / 18480 = 276.467... 
    # Actually, let's just construct the matrix numerically.
    
    # Basis vectors for c0, c1, c2
    # v0: c0=1, c1=0, c2=0 -> c3 = -1/18480
    # v1: c0=0, c1=1, c2=0 -> c3 = -12/18480 = -1/1540
    # v2: c0=0, c1=0, c2=1 -> c3 = -168/18480 = -1/110
    
    def calc_b(c0, c1, c2):
        c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
        b1 = -48.0 * c1 - 40320.0 * c2 - 5109120.0 * c3
        b2 = 16.0 * c1 + 26880.0 * c2 + 9072000.0 * c3
        b3 = -4608.0 * c2 - 6005760.0 * c3
        b4 = 256.0 * c2 + 1424640.0 * c3
        b5 = -142464.0 * c3
        b6 = 4096.0 * c3
        return np.array([b1, b2, b3, b4, b5, b6])
        
    M = np.zeros((6, 3))
    M[:, 0] = calc_b(1.0, 0.0, 0.0)
    M[:, 1] = calc_b(0.0, 1.0, 0.0)
    M[:, 2] = calc_b(0.0, 0.0, 1.0)
    return M

_M = _get_q_matrix()

def _q_coefficients(c0, c1, c2):
    c = np.array([c0, c1, c2])
    return _M @ c

def _find_largest_sign_change_root(coeffs):
    """
    Find the largest real positive root of q(y) where q changes sign.
    coeffs = [b1, b2, ..., b6] for polynomial b1 + b2*y + ... + b6*y^5.
    Returns r^2 = y_root (the largest such y), or None if no valid root.
    """
    poly_coeffs = np.array(coeffs[::-1])
    roots = np.roots(poly_coeffs)
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 1e-12:
            real_roots.append(r.real)
            
    if not real_roots:
        return None
        
    real_roots.sort()
    
    def q_eval(y):
        result = 0.0
        power = 1.0
        for b in coeffs:
            result += b * power
            power *= y
        return result
        
    best_y = None
    for y in real_roots:
        eps = max(1e-8, abs(y) * 1e-8)
        q_below = q_eval(y - eps)
        q_above = q_eval(y + eps)
        if q_below * q_above < 0:
            if best_y is None or y > best_y:
                best_y = y
                
    return best_y

def _evaluate_value(c0, c1, c2):
    coeffs = _q_coefficients(c0, c1, c2)
    y_root = _find_largest_sign_change_root(coeffs)
    if y_root is None:
        return np.inf
    return y_root / (2.0 * np.pi)

def _solve_double_root(y_target):
    """
    Find coefficients c such that q(y_target) = 0 and q'(y_target) = 0.
    Returns None if no non-trivial solution exists or if y_target is not the largest sign-changing root.
    """
    # q(y) = b @ [1, y, y^2, y^3, y^4, y^5] = 0
    # q'(y) = b @ [0, 1, 2y, 3y^2, 4y^3, 5y^4] = 0
    # b = M @ c
    # A @ c = 0 where A = [[v0, v1], [v2, v3]]
    # v0 = [1, y, y^2, y^3, y^4, y^5] @ M
    # v1 = [0, 1, 2y, 3y^2, 4y^3, 5y^4] @ M
    
    v0 = np.array([1, y_target, y_target**2, y_target**3, y_target**4, y_target**5]) @ _M
    v1 = np.array([0, 1, 2*y_target, 3*y_target**2, 4*y_target**3, 5*y_target**4]) @ _M
    
    A = np.vstack([v0, v1])
    
    # Find null space of A (2x3 matrix)
    # Use SVD
    U, S, Vt = np.linalg.svd(A)
    # The last row of Vt is the null space vector
    c_vec = Vt[-1]
    
    # Normalize to avoid trivial solution or huge values
    norm = np.linalg.norm(c_vec)
    if norm < 1e-10:
        return None
    c_vec = c_vec / norm
    
    c0, c1, c2 = c_vec
    
    # Verify: Is y_target the largest sign-changing root?
    coeffs = _q_coefficients(c0, c1, c2)
    y_max = _find_largest_sign_change_root(coeffs)
    
    # Check if y_max is close to y_target
    if y_max is None:
        return None
    if abs(y_max - y_target) < 1e-6 * max(1.0, y_target):
        return c_vec
    else:
        return None

def construct():
    global _t_start
    _t_start = time.time()
    
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_value = _evaluate_value(best_params[0], best_params[1], best_params[2])
    
    # Phase 1: Scan y values to find potential double roots
    # The optimal y is likely in a range. Let's scan [0.1, 100] with logarithmic steps.
    y_candidates = np.logspace(-1, 2, 1000)
    
    for y in y_candidates:
        if _remaining() < 5:
            break
        c_vec = _solve_double_root(y)
        if c_vec is not None:
            val = _evaluate_value(c_vec[0], c_vec[1], c_vec[2])
            if np.isfinite(val) and val < best_value:
                best_value = val
                best_params = c_vec.copy()
                
    # Phase 2: Local refinement around best y found
    # We can use the structure: for a given c, y is determined.
    # But we can also do a local search on c using the double root constraint relaxation.
    # Or simply, scan a finer range around the best y found.
    
    if best_value < np.inf:
        y_best = best_value * 2.0 * np.pi
        # Fine scan around y_best
        y_fine = np.linspace(y_best * 0.9, y_best * 1.1, 1000)
        for y in y_fine:
            if _remaining() < 2:
                break
            c_vec = _solve_double_root(y)
            if c_vec is not None:
                val = _evaluate_value(c_vec[0], c_vec[1], c_vec[2])
                if np.isfinite(val) and val < best_value:
                    best_value = val
                    best_params = c_vec.copy()

    return np.array(best_params)
# EVOLVE-BLOCK-END
