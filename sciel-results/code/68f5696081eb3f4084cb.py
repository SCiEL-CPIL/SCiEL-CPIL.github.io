import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as a list [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n."""
    # Use sympy for exact coefficients
    try:
        from sympy.polys.orthopolys import hermite_poly
        from sympy import symbols, Poly
        x = symbols('x')
        Hn = hermite_poly(n, x)
        p = Poly(Hn, x)
        return p.all_coeffs()  # returns [leading, ..., constant]
    except:
        # Fallback: compute recursively
        # H0 = 1, H1 = 2x, H_{n+1} = 2x*Hn - 2n*H_{n-1}
        H0 = [1]
        if n == 0:
            return H0
        H1 = [2, 1]  # 2x + 1*0? No, [2, 1] means 2x^1 + 1*x^0? 
        # Let's use a different representation: list where index i is coefficient of x^i
        H0 = [1.0]
        H1 = [0.0, 2.0]  # 2x
        if n == 1:
            return H1
        for k in range(2, n+1):
            Hk = [0.0] * (k+1)
            # Hk = 2x * H_{k-1} - 2(k-1) * H_{k-2}
            for i in range(k-1):
                Hk[i+1] += 2.0 * H1[i]  # This is wrong, let me redo
            # Let me redo this properly
            break
        # Better approach
        H = [[1.0], [0.0, 2.0]]
        for k in range(2, n+1):
            prev1 = H[k-1]  # H_{k-1}
            prev2 = H[k-2]  # H_{k-2}
            curr = [0.0] * (k+1)
            # 2x * H_{k-1}: shift prev1 by 1 and multiply by 2
            for i in range(k):
                curr[i+1] += 2.0 * prev1[i]
            # -2(k-1) * H_{k-2}
            for i in range(k-1):
                curr[i] += -2.0*(k-1) * prev2[i]
            H.append(curr)
        return H[n]

def _poly_divide_by_x2(coeffs_asc):
    """Divide polynomial (coeffs ascending: [c0, c1, ..., cn]) by x^2.
    Returns coefficients of result, or None if not divisible."""
    if len(coeffs_asc) < 3:
        return None
    if abs(coeffs_asc[0]) > 1e-15 or abs(coeffs_asc[1]) > 1e-15:
        return None
    return coeffs_asc[2:]

def _find_largest_sign_changing_root(coeffs_asc):
    """Find the largest positive real root where the polynomial changes sign.
    coeffs_asc: list of coefficients [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n"""
    # Convert to numpy polynomial (descending order)
    if len(coeffs_asc) == 0:
        return None
    coeffs_desc = list(reversed(coeffs_asc))
    roots = np.roots(coeffs_desc)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    # For each positive real root, check if sign changes
    # Evaluate polynomial at points near the root
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in real_roots:
        if r <= 0:
            continue
        # Check sign change: evaluate at r - eps and r + eps
        eps = 1e-8 * max(1.0, abs(r))
        val_left = poly(r - eps)
        val_right = poly(r + eps)
        if val_left * val_right < 0:
            # Sign change at r
            if largest is None or r > largest:
                largest = r
    
    return largest

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients."""
    H0 = _get_hermite_coeffs(0)
    H4 = _get_hermite_coeffs(4)
    H8 = _get_hermite_coeffs(8)
    H12 = _get_hermite_coeffs(12)
    
    # Build g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # All are even polynomials, so only even powers
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for coeff, H in [(c0, H0), (c1, H4), (c2, H8)]:
        for i in range(len(H)):
            g_coeffs[i] += coeff * H[i]
    
    # Compute c3 so that g(0) = 0
    # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0)
    H0_0 = H0[0]
    H4_0 = H4[0]
    H8_0 = H8[0]
    H12_0 = H12[0]
    
    if abs(H12_0) < 1e-15:
        return float('inf')
    
    c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    
    for i in range(len(H12)):
        g_coeffs[i] += c3 * H12[i]
    
    # Check if g tends to negative at infinity
    # Leading coefficient of g
    lead_coeff = g_coeffs[12]
    if lead_coeff < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    # q = g / x^2
    q_coeffs = _poly_divide_by_x2(g_coeffs)
    if q_coeffs is None:
        return float('inf')
    
    # Find largest sign-changing root
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None:
        return float('inf')
    
    return r**2 / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s
    
    # Seed
    best_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    # Use Nelder-Mead optimization
    def objective(x):
        try:
            return _compute_value(x[0], x[1], x[2])
        except:
            return 1e10
    
    # Try multiple starting points
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline:
            break
        
        if i == 0:
            x0 = best_c.copy()
        else:
            x0 = np.random.uniform(-2, 2, size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    return best_c
