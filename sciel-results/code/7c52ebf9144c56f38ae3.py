import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients (ascending: index i = coeff of x^i)
def _precompute_hermite():
    H = {}
    # H0 = 1
    H[0] = [1.0]
    # H1 = 2x
    H[1] = [0.0, 2.0]
    for n in range(2, 13):
        prev1 = H[n-1]
        prev2 = H[n-2]
        curr = [0.0] * (n+1)
        # Hn = 2x * H_{n-1} - 2(n-1) * H_{n-2}
        for i in range(n):
            curr[i+1] += 2.0 * prev1[i]
        for i in range(n-1):
            curr[i] += -2.0*(n-1) * prev2[i]
        H[n] = curr
    return H

_HERMITE = _precompute_hermite()
_H0 = _HERMITE[0]
_H4 = _HERMITE[4]
_H8 = _HERMITE[8]
_H12 = _HERMITE[12]

# Values at x=0
_H0_0 = _H0[0]       # 1
_H4_0 = _H4[0]       # 12
_H8_0 = _H8[0]       # 1008
_H12_0 = _H12[0]     # 103950

# Build the linear system: g = c0*H0 + c1*H4 + c2*H8 + c3*H12
# where c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
# So g = c0*(H0 - H0_0/H12_0 * H12) + c1*(H4 - H4_0/H12_0 * H12) + c2*(H8 - H8_0/H12_0 * H12)
# Let's define the "effective" basis polynomials
_B0 = [a - (H0_0/H12_0)*b for a, b in zip(_H0 + [0.0]*8, _H12)]
_B1 = [a - (H4_0/H12_0)*b for a, b in zip(_H4 + [0.0]*4, _H12)]
_B2 = [a - (H8_0/H12_0)*b for a, b in zip(_H8 + [0.0]*0, _H12)]

# Pad to length 13
_B0 = _B0 + [0.0]*(13-len(_B0))
_B1 = _B1 + [0.0]*(13-len(_B1))
_B2 = _B2 + [0.0]*(13-len(_B2))

# g(x) = c0*B0 + c1*B1 + c2*B2
# g is even, so only even powers. g(x)/x^2 = q(x) of degree 10
# q(x) = sum_{k=0}^{5} q_k * x^(2k)
# The coefficients of q are linear in (c0, c1, c2)
# Build matrix M where M[i][j] gives coefficient of x^(2i) in q for basis j
# q has powers x^0, x^2, x^4, x^6, x^8, x^10 (6 coefficients)
# g has powers x^0, x^2, ..., x^12 (7 coefficients)
# q_k = g_{2k+2} for k=0..5

def _build_q_matrix():
    """Build 6x3 matrix M such that q_coeffs = M @ [c0, c1, c2]"""
    M = np.zeros((6, 3))
    for k in range(6):
        power = 2*(k+1)  # x^(2k+2) in g corresponds to x^(2k) in q
        M[k, 0] = _B0[power]
        M[k, 1] = _B1[power]
        M[k, 2] = _B2[power]
    return M

_M = _build_q_matrix()

def _compute_value_fast(c0, c1, c2):
    """Fast computation of value using NumPy."""
    try:
        # q coefficients (ascending: q[0] + q[1]*x^2 + ... + q[5]*x^10)
        q_asc = _M @ np.array([c0, c1, c2])
        
        # Check leading coefficient sign
        # g's leading coeff is g[12] = M[5] @ [c0,c1,c2]
        lead = q_asc[5]
        if abs(lead) < 1e-15:
            return float('inf')
        
        # If g tends to negative at infinity, flip
        if lead < 0:
            q_asc = -q_asc
            lead = -lead
        
        # q as polynomial in y = x^2: q(y) = q_asc[0] + q_asc[1]*y + ... + q_asc[5]*y^5
        # But we need roots in x, so q(x) = sum q_asc[k] * x^(2k)
        # Convert to standard polynomial in x: coeffs descending
        # q(x) = q_asc[5]*x^10 + q_asc[4]*x^8 + ... + q_asc[0]
        # Only even powers, so in terms of x:
        q_coeffs_desc = np.zeros(11)
        for k in range(6):
            q_coeffs_desc[10 - 2*k] = q_asc[k]
        
        # Find roots
        roots = np.roots(q_coeffs_desc)
        
        # Filter real positive roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_roots.append(r.real)
        
        if not real_roots:
            return float('inf')
        
        # Evaluate q at points to check sign change
        # Use np.poly1d for evaluation
        poly = np.poly1d(q_coeffs_desc)
        
        largest = None
        for r in real_roots:
            if r <= 1e-12:
                continue
            eps = 1e-8 * max(1.0, abs(r))
            val_left = poly(r - eps)
            val_right = poly(r + eps)
            if val_left * val_right < 0:
                if largest is None or r > largest:
                    largest = r
        
        if largest is None:
            return float('inf')
        
        return largest**2 / (2.0 * np.pi)
    
    except Exception:
        return float('inf')

def _verify_exact(c0, c1, c2):
    """Verify with exact Sympy arithmetic."""
    try:
        from sympy import Rational, symbols, Poly, hermite_poly, pi, S
        from sympy.polys.polytools import Poly as SymPyPoly
        
        x = symbols('x')
        H0 = hermite_poly(0, x)
        H4 = hermite_poly(4, x)
        H8 = hermite_poly(8, x)
        H12 = hermite_poly(12, x)
        
        H0_0 = H0.subs(x, 0)
        H4_0 = H4.subs(x, 0)
        H8_0 = H8.subs(x, 0)
        H12_0 = H12.subs(x, 0)
        
        c0_r = Rational(c0).limit_denominator(10**15)
        c1_r = Rational(c1).limit_denominator(10**15)
        c2_r = Rational(c2).limit_denominator(10**15)
        
        c3_r = -(c0_r*H0_0 + c1_r*H4_0 + c2_r*H8_0) / H12_0
        
        g = c0_r*H0 + c1_r*H4 + c2_r*H8 + c3_r*H12
        
        # Check leading coefficient
        g_poly = SymPyPoly(g, x)
        lead = g_poly.LC()
        if lead < 0:
            g = -g
            g_poly = SymPyPoly(g, x)
        
        # q = g / x^2
        coeff_dict = g_poly.as_dict()
        q_terms = []
        for power in sorted(coeff_dict.keys(), reverse=True):
            if power < 2 or power % 2 != 0:
                continue
            q_power = power - 2
            q_coeff = coeff_dict[(power,)]
            q_terms.append((q_power, q_coeff))
        
        if not q_terms:
            return float('inf')
        
        q_expr = sum(c * x**p for p, c in q_terms)
        q_poly = SymPyPoly(q_expr, x)
        
        # Find real roots
        roots = q_poly.nroots(method='subresultant', n=15)
        
        real_roots = []
        for root in roots:
            re, im = root.as_real_imag()
            if abs(im) < 1e-10:
                real_roots.append(float(re))
        
        if not real_roots:
            return float('inf')
        
        # Check sign change
        valid_roots = []
        for r in real_roots:
            if r <= 1e-12:
                continue
            eps = S(1)/10**8
            left_val = q_expr.subs(x, r - eps)
            right_val = q_expr.subs(x, r + eps)
            if float(left_val) * float(right_val) < 0:
                valid_roots.append(r)
        
        if not valid_roots:
            return float('inf')
        
        r = max(valid_roots)
        return r**2 / (2 * float(pi))
    
    except Exception:
        return float('inf')

def _root_based_search(deadline, best_c, best_val):
    """Search using root parameterization of q(x)."""
    # q(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0 where y = x^2
    # We want q to have a large positive root r where it changes sign
    # Parameterize: q(y) = a5 * prod_{i=1}^{5} (y - r_i)
    # where r_1 is the largest root (in x-space, so r_1 = R^2)
    # For sign change at R, we need q(R) = 0 and q changes sign
    # If R is a simple root of q in x-space, it's a simple root in y-space too (since y=R^2)
    
    # Let's parameterize by the roots of q in y-space
    # q(y) = a * (y - R^2) * (y - s2) * (y - s3) * (y - s4) * (y - s5)
    # For q to be in our 3D subspace, the coefficients must satisfy:
    # q_asc = M @ [c0, c1, c2]
    # So we need: a * (coefficients of expanded polynomial) = M @ [c0, c1, c2]
    # This is an overdetermined system (6 equations, 3 unknowns + a and 4 other roots = 9 params)
    # So we can't freely choose all roots. We need to find roots such that the resulting
    # q is in the column space of M.
    
    # Alternative: just do a direct search in (c0, c1, c2) space but use
    # root information to guide the search direction.
    
    # For now, do a focused coordinate descent
    scales = [1.0, 0.1, 0.01, 0.001, 0.0001]
    current = best_c.copy()
    current_val = best_val
    
    for scale in scales:
        if time.time() > deadline - 30:
            break
        
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(3):
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[i] += direction * scale * [1.0, 0.1, 0.01][i]
                    val = _compute_value_fast(trial[0], trial[1], trial[2])
                    if val < current_val - 1e-12:
                        current = trial
                        current_val = val
                        improved = True
        
        if current_val < best_val:
            best_val = current_val
            best_c = current.copy()
    
    return best_c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s
    
    # Seed
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_value_fast(seed[0], seed[1], seed[2])
    best_c = seed.copy()
    
    # Phase 1: Nelder-Mead from multiple starting points
    def objective(x):
        try:
            return _compute_value_fast(x[0], x[1], x[2])
        except:
            return 1e10
    
    np.random.seed(42)
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 60 and n_restarts < max_restarts:
        if n_restarts == 0:
            x0 = best_c.copy()
        elif n_restarts < 5:
            # Near seed
            x0 = best_c + np.random.normal(0, 0.1, size=3)
        else:
            # Broader
            x0 = np.random.uniform([-2.0, -0.5, -0.1], [0.5, 0.5, 0.1], size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            if result.fun < best_val - 1e-12:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
        
        n_restarts += 1
    
    # Phase 2: Root-based local refinement
    if time.time() < deadline - 30:
        best_c, best_val = _root_based_search(deadline, best_c, best_val)
    
    # Phase 3: Fine-tuning with very small steps
    if time.time() < deadline - 15:
        for scale in [1e-4, 1e-5, 1e-6, 1e-7]:
            if time.time() > deadline - 15:
                break
            for _ in range(50):
                if time.time() > deadline - 15:
                    break
                trial = best_c + np.random.normal(0, scale, size=3)
                val = _compute_value_fast(trial[0], trial[1], trial[2])
                if val < best_val - 1e-14:
                    best_val = val
                    best_c = trial
    
    # Phase 4: Exact verification and final polish
    if time.time() < deadline - 5:
        exact_val = _verify_exact(best_c[0], best_c[1], best_c[2])
        if exact_val < best_val:
            best_val = exact_val
    
    return best_c
