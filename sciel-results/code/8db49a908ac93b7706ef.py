"""Refined program: fast surrogate search followed by exact verification."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """High-accuracy k-th positive zero of J_0."""
    zeros = [
        2.4048255576957728,
        5.5200781102863106,
        8.6537279129110122,
        11.7915344390143598,
        14.9309177084877878,
        18.0710639679109298,
        21.2116366299713778,
        24.3524715308462127,
        27.4934791320444287,
        30.6346064684319602,
        33.7758202136374444,
        36.9171012943291185
    ]
    if 1 <= k <= 12:
        return zeros[k-1]
    # Fallback
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _get_laguerre_zeros(n, alpha):
    """Get zeros of L_n^(alpha)(x) using sympy (cached or fast)."""
    from sympy import symbols, Rational, Poly
    from sympy.polys.orthopoly import genlaguerre
    x = symbols('x')
    L = genlaguerre(n, Rational(alpha))(x)
    roots = Poly(L, x).nroots()
    return [float(r) for r in roots if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]

def _precompute_laguerre_coeffs():
    """Precompute power basis coefficients for L_{2k}^{(-1/2)}(x), k=0..25."""
    from sympy import symbols, Rational, Poly
    from sympy.polys.orthopoly import genlaguerre
    x = symbols('x')
    alpha = Rational(-1, 2)
    coeffs_list = []
    deriv_coeffs_list = []
    for k in range(26):
        n = 2 * k
        L = genlaguerre(n, alpha)(x)
        p = Poly(L, x)
        coeffs = np.array([float(c) for c in p.all_coeffs()])
        coeffs_list.append(coeffs)
        
        # Derivative
        Ld = L.diff(x)
        pd = Poly(Ld, x)
        if pd.degree() == -1: # zero polynomial
            dcoeffs = np.array([0.0])
        else:
            dcoeffs = np.array([float(c) for c in pd.all_coeffs()])
        deriv_coeffs_list.append(dcoeffs)
    return coeffs_list, deriv_coeffs_list

# Global cache for coeffs
_LAG_COEFFS = None
_LAG_DERIV_COEFFS = None

def _get_lag_coeffs():
    global _LAG_COEFFS, _LAG_DERIV_COEFFS
    if _LAG_COEFFS is None:
        _LAG_COEFFS, _LAG_DERIV_COEFFS = _precompute_laguerre_coeffs()
    return _LAG_COEFFS, _LAG_DERIV_COEFFS

def _eval_poly(coeffs, x):
    """Evaluate polynomial with coeffs [a_n, ..., a_0] at x."""
    return np.polyval(coeffs, x)

def _eval_poly_deriv(coeffs, x):
    """Evaluate derivative of polynomial with coeffs [a_n, ..., a_0] at x."""
    if len(coeffs) <= 1:
        return 0.0
    # Derivative coeffs: [n*a_n, (n-1)*a_{n-1}, ...]
    n = len(coeffs) - 1
    dcoeffs = np.array([coeffs[i] * (n - i) for i in range(len(coeffs) - 1)])
    return np.polyval(dcoeffs, x)

def _fast_score(z_list):
    """
    Compute approximate B/(2*pi) using float arithmetic.
    Returns None if invalid.
    """
    coeffs_list, deriv_coeffs_list = _get_lag_coeffs()
    
    z = np.array(z_list, dtype=float)
    
    # Build A (26x26) and b (26x1)
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0 -> sum c_k L_{2k}(0) = 0
    # Row 1: g'(0) = 1 -> sum c_k L'_{2k}(0) = 1
    for k in range(26):
        A[0, k] = _eval_poly(coeffs_list[k], 0.0)
        A[1, k] = _eval_poly_deriv(coeffs_list[k], 0.0)
        
        # Rows 2..25: g(z_i) = 0, g'(z_i) = 0
        for i in range(12):
            A[2 + 2*i, k] = _eval_poly(coeffs_list[k], z[i])
            A[3 + 2*i, k] = _eval_poly_deriv(coeffs_list[k], z[i])
    
    b[1] = 1.0
    
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    # Build g(x) coeffs
    # g(x) = sum c_k L_{2k}(x)
    # We need to sum polynomials.
    max_deg = 50
    g_coeffs = np.zeros(max_deg + 1)
    
    for k in range(26):
        L_coeffs = coeffs_list[k]
        # Pad L_coeffs to length 51
        len_L = len(L_coeffs)
        # L_coeffs is [a_50, ..., a_0] for degree 50?
        # sympy all_coeffs returns highest degree first.
        # L_{2k} has degree 2k.
        # So L_coeffs has length 2k+1.
        # We need to align them to the right (lowest degree).
        # g_coeffs is length 51.
        # L_coeffs[0] is coeff of x^{2k}.
        # We want to add c_k * L_coeffs to g_coeffs.
        # g_coeffs[50 - 2k : 51] += c_k * L_coeffs
        
        start_idx = 50 - 2 * k
        g_coeffs[start_idx : start_idx + len_L] += c[k] * L_coeffs
        
    # Denominator: x * prod (x - z_i)^2
    # Build denom polynomial coeffs
    # Start with x -> [1, 0]
    denom_coeffs = np.array([1.0, 0.0])
    
    for zi in z:
        # (x - zi)^2 = x^2 - 2 zi x + zi^2 -> [1, -2zi, zi^2]
        sq_coeffs = np.array([1.0, -2.0 * zi, zi**2])
        denom_coeffs = np.convolve(denom_coeffs, sq_coeffs)
        
    # Divide g by denom
    # np.polydiv requires highest degree first
    # g_coeffs is highest degree first.
    # denom_coeffs is highest degree first.
    try:
        q_coeffs, rem_coeffs = np.polydiv(g_coeffs, denom_coeffs)
    except Exception:
        return None
        
    # Check remainder (should be small)
    if np.max(np.abs(rem_coeffs)) > 1e-8:
        # Might be numerical instability, but let's proceed if not huge
        pass
        
    # Find roots of q
    if len(q_coeffs) == 0:
        return 0.0
    if len(q_coeffs) == 1:
        # Constant, no roots
        return 0.0
        
    roots = np.roots(q_coeffs)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            real_roots.append(r.real)
            
    if not real_roots:
        return 0.0
        
    max_r = max(real_roots)
    if max_r < 0:
        return 0.0
        
    return max_r / (2.0 * math.pi)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix, S)
    from sympy.polys.orthopoly import genlaguerre

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = symbols('x')
    alpha = S(-1)/2

    laguerre_pols = []
    for k in range(26):
        n = 2*k
        laguerre_pols.append(genlaguerre(n, alpha)(x))

    A = Matrix(26, 26, lambda i, j: S(0))
    b = Matrix(26, 1, lambda i, j: S(0))

    for k in range(26):
        L = laguerre_pols[k]
        Ld = L.diff(x)
        A[0, k] = L.subs(x, 0)
        A[1, k] = Ld.subs(x, 0)
        for i in range(12):
            zi = z_rationals[i]
            A[2 + 2*i, k] = L.subs(x, zi)
            A[3 + 2*i, k] = Ld.subs(x, zi)
    b[1, 0] = 1

    try:
        c = A.LUsolve(b)
    except Exception:
        return None

    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2

    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    try:
        q_poly, rem = g_poly.div(denom_poly)
        if rem != 0:
            return None
    except Exception:
        return None

    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None

    eps = Rational(1, 10)**198
    max_r = S(0)
    for r in real_roots_list:
        try:
            qm = q_expr.subs(x, r - eps)
            qp = q_expr.subs(x, r + eps)
            if qm * qp < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue

    B = max(S(0), max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def construct():
    """Return 12 double-root locations."""
    deadline = time.time() + 580.0
    
    # 1. Generate Base Sets
    base_sets = []
    names = []
    
    # Set 1: Bessel squares
    bessel_z = [_bessel_zero_J0(i+1)**2 for i in range(12)]
    base_sets.append(bessel_z)
    names.append("BesselSq")
    
    # Set 2-5: Laguerre zeros
    lag_zeros = _get_laguerre_zeros(50, -0.5)
    if len(lag_zeros) >= 12:
        lag_zeros.sort()
        # Smallest 12
        base_sets.append(lag_zeros[:12])
        names.append("LagSmall")
        # Largest 12
        base_sets.append(lag_zeros[-12:])
        names.append("LagLarge")
        # Middle 12
        base_sets.append(lag_zeros[19:31])
        names.append("LagMid")
        
    # Set 6: Linear
    base_sets.append([float(i+1) for i in range(12)])
    names.append("Linear")
    
    # Set 7: Quadratic
    base_sets.append([float(i+1)**2 for i in range(12)])
    names.append("Quad")
    
    # 2. Fast Search
    candidates = []
    
    for idx, base in enumerate(base_sets):
        name = names[idx]
        # Determine scale range
        # For Bessel, natural scale is 1.
        # For Lag, natural scale is 1.
        # Try log spaced scales
        scales = np.logspace(-2, 2, 15)
        
        for s in scales:
            z_candidate = [z * s for z in base]
            # Validate
            valid = True
            for i in range(12):
                if z_candidate[i] <= 0 or not math.isfinite(z_candidate[i]):
                    valid = False
                    break
                for j in range(i+1, 12):
                    if abs(z_candidate[i] - z_candidate[j]) < 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            try:
                score = _fast_score(z_candidate)
                if score is not None:
                    candidates.append((score, name, s, z_candidate))
            except Exception:
                pass
                
    # Sort candidates by fast score
    candidates.sort(key=lambda x: x[0])
    
    # 3. Exact Verification of Top Candidates
    best_score = float('inf')
    best_z = None
    
    # Take top 10 unique candidates
    tested = set()
    count = 0
    for score, name, s, z in candidates:
        # Simple uniqueness check
        key = tuple(round(z_i, 4) for z_i in sorted(z))
        if key in tested:
            continue
        tested.add(key)
        count += 1
        
        if time.time() > deadline - 60:
            break
            
        try:
            exact_score = _compute_score_exact(z)
            if exact_score is not None and exact_score < best_score:
                best_score = exact_score
                best_z = z
        except Exception:
            pass
        
        if count >= 10:
            break
            
    if best_z is None:
        # Fallback to original Bessel squares
        best_z = bessel_z
        # Try to get a valid score for fallback
        try:
            best_score = _compute_score_exact(best_z)
        except:
            pass
            
    return best_z

# EVOLVE-BLOCK-END
