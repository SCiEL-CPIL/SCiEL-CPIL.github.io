"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 630
    safety_margin = 40  # seconds
    best_score = float('inf')
    best_z = None

    def time_remaining():
        return deadline - time.time() - safety_margin

    # Precompute Laguerre zeros for various degrees using numpy
    def laguerre_zeros(n, alpha):
        """Compute zeros of L_n^(alpha)(x) using companion matrix eigenvalues."""
        n = int(n)
        a = alpha
        diag = np.array([2.0*k - 1.0 + a for k in range(1, n+1)])
        subdiag = np.array([np.sqrt(k*(k+a)) for k in range(1, n)])
        J = np.diag(diag)
        for i in range(n-1):
            J[i, i+1] = subdiag[i]
            J[i+1, i] = subdiag[i]
        eigvals = np.linalg.eigvalsh(J)
        return np.sort(eigvals)

    # Bessel J_{1/2} zeros: J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    # Zeros at sqrt(x) = k*pi => x = k^2 * pi^2
    def bessel_j_half_zeros(count):
        """Return the 'count' smallest positive zeros of J_{1/2}(x)."""
        return [float(i**2 * math.pi**2) for i in range(1, count + 1)]

    alpha = -0.5

    # Build candidate sets, prioritizing most promising
    candidate_sets = []

    # Priority 1: Zeros of L_12^(-0.5)(x) - natural nodes for this problem
    try:
        z12 = laguerre_zeros(12, alpha)
        if len(z12) == 12 and all(z > 0 for z in z12) and len(set(np.round(z12, 10))) == 12:
            candidate_sets.append(list(z12))
    except:
        pass

    # Priority 2: Bessel J_{1/2} zeros
    try:
        z_bessel = bessel_j_half_zeros(12)
        if len(z_bessel) == 12 and all(z > 0 for z in z_bessel) and len(set(z_bessel)) == 12:
            candidate_sets.append(z_bessel)
    except:
        pass

    # Priority 3: Zeros of L_13^(-0.5)(x), take first 12
    try:
        z13 = laguerre_zeros(13, alpha)
        if len(z13) >= 12:
            z13_12 = z13[:12]
            if all(z > 0 for z in z13_12) and len(set(np.round(z13_12, 10))) == 12:
                candidate_sets.append(list(z13_12))
    except:
        pass

    # Priority 4: Zeros of L_n^(-0.5) for various n, take 12
    for n in [14, 15, 16, 17, 18, 20, 25]:
        try:
            zn = laguerre_zeros(n, alpha)
            if len(zn) >= 12:
                sub = zn[:12]
                if all(z > 0 for z in sub) and len(set(np.round(sub, 10))) == 12:
                    candidate_sets.append(list(sub))
        except:
            pass

    # Priority 5: Log-spaced
    for lo, hi in [(0.5, 50), (1, 20), (0.1, 10), (1, 10)]:
        z = list(np.logspace(np.log10(lo), np.log10(hi), 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Priority 6: Geometric sequences
    for ratio in [1.15, 1.3, 1.5]:
        for start in [0.5, 1.0, 2.0]:
            z = [start * (ratio ** i) for i in range(12)]
            if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
                candidate_sets.append(z)

    # Deduplicate candidate sets
    unique_sets = []
    seen = set()
    for cs in candidate_sets:
        key = tuple(round(v, 8) for v in sorted(cs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(cs)

    # Numerical pre-screening: estimate B using float arithmetic
    def numerical_prescreen(z_list):
        """
        Quick numerical estimate of B/(2*pi) using float Laguerre polynomials.
        Returns estimated B/(2*pi), or inf if something goes wrong.
        """
        try:
            from scipy.special import eval_genlaguerre
        except ImportError:
            return float('inf')

        alpha_val = -0.5
        z_floats = sorted([float(zi) for zi in z_list])

        def L_2k(x_val, k):
            return eval_genlaguerre(2*k, alpha_val, x_val)

        def dL_2k(x_val, k):
            if 2*k == 0:
                return 0.0
            return -eval_genlaguerre(2*k - 1, alpha_val + 1, x_val)

        A_rows = []
        b_vec = []

        # g(0) = 0
        row = [L_2k(0.0, k) for k in range(26)]
        A_rows.append(row)
        b_vec.append(0.0)

        # g'(0) = 1
        row = [dL_2k(0.0, k) for k in range(26)]
        A_rows.append(row)
        b_vec.append(1.0)

        for zi in z_floats:
            row = [L_2k(zi, k) for k in range(26)]
            A_rows.append(row)
            b_vec.append(0.0)

            row = [dL_2k(zi, k) for k in range(26)]
            A_rows.append(row)
            b_vec.append(0.0)

        A_mat = np.array(A_rows)
        b_mat = np.array(b_vec)

        try:
            cond = np.linalg.cond(A_mat)
            if cond > 1e18:
                return float('inf')
            c_sol = np.linalg.solve(A_mat, b_mat)
        except:
            return float('inf')

        # Build g(x) and find roots numerically
        # g(x) = sum c_k * L_2k(x)
        # We know g has double roots at z_i and g(0)=0, g'(0)=1
        # q(x) = g(x) / (x * prod (x-z_i)^2) is degree 25
        # Find roots of g(x) beyond the known ones

        # Sample g on a fine grid and find sign changes
        max_z = max(z_floats)
        # Search range: from 0 to a few times max_z
        upper = max_z * 3.0 + 10.0
        n_samples = 2000
        x_grid = np.linspace(1e-6, upper, n_samples)

        g_vals = np.zeros(n_samples)
        for k in range(26):
            g_vals += c_sol[k] * eval_genlaguerre(2*k, alpha_val, x_grid)

        # Find sign changes (potential roots of g)
        sign_changes = []
        for i in range(n_samples - 1):
            if g_vals[i] * g_vals[i+1] < 0:
                # Bisection to find root
                a, b = x_grid[i], x_grid[i+1]
                fa, fb = g_vals[i], g_vals[i+1]
                for _ in range(50):
                    m = (a + b) / 2
                    fm = sum(c_sol[k] * eval_genlaguerre(2*k, alpha_val, m) for k in range(26))
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                root = (a + b) / 2
                sign_changes.append(root)

        # Filter out the known roots (z_i and 0)
        extra_roots = []
        for r in sign_changes:
            is_known = False
            for zi in z_floats:
                if abs(r - zi) < 1e-4:
                    is_known = True
                    break
            if not is_known and r > 1e-5:
                extra_roots.append(r)

        if not extra_roots:
            return 0.0  # No extra roots means B=0, best case

        B_est = max(extra_roots)
        return B_est / (2 * math.pi)

    # Exact evaluation using sympy
    def evaluate(z_list):
        """Evaluate B/(2*pi) for given z_list using sympy exact arithmetic."""
        import sympy as sp
        x = sp.Symbol('x')
        alpha_val = sp.Rational(-1, 2)

        # Convert z_i to rational
        z_rats = [sp.Rational(float(zi)) for zi in z_list]
        z_rats_sorted = sorted(z_rats)

        if len(set(z_rats_sorted)) != 12:
            return float('inf')
        if any(z <= 0 for z in z_rats_sorted):
            return float('inf')

        # Check time
        if time_remaining() < 30:
            return float('inf')

        # Build L_2k^(alpha)(x) for k=0..25
        # Use sp.genlaguerre for generalized Laguerre
        L_terms = []
        for k in range(26):
            L = sp.genlaguerre(2*k, alpha_val, x)
            L_terms.append(sp.expand(L))

        if time_remaining() < 25:
            return float('inf')

        # Build the linear system
        A_rows = []
        b_vec = []

        # g(0) = 0
        row = [L_terms[k].subs(x, 0) for k in range(26)]
        A_rows.append(row)
        b_vec.append(0)

        # g'(0) = 1
        row = [sp.diff(L_terms[k], x).subs(x, 0) for k in range(26)]
        A_rows.append(row)
        b_vec.append(1)

        for zi in z_rats_sorted:
            row = [L_terms[k].subs(x, zi) for k in range(26)]
            A_rows.append(row)
            b_vec.append(0)

            row = [sp.diff(L_terms[k], x).subs(x, zi) for k in range(26)]
            A_rows.append(row)
            b_vec.append(0)

        if time_remaining() < 20:
            return float('inf')

        A_mat = sp.Matrix(A_rows)
        b_mat = sp.Matrix(b_vec)

        # Solve using LUsolve
        try:
            c_sol = A_mat.LUsolve(b_mat)
        except:
            return float('inf')

        if time_remaining() < 15:
            return float('inf')

        # Build g(x)
        g_sol = sum(c_sol[k] * L_terms[k] for k in range(26))
        g_sol = sp.expand(g_sol)

        if time_remaining() < 10:
            return float('inf')

        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for zi in z_rats_sorted:
            denom = denom * (x - zi)**2
        denom = sp.expand(denom)

        # q(x) = g(x) / denom, exact polynomial division
        q_poly, remainder = sp.div(g_sol, denom, x)
        if remainder != 0:
            return float('inf')

        if time_remaining() < 5:
            return float('inf')

        # Find real roots of q_poly
        try:
            q_poly_sp = sp.Poly(q_poly, x)
            real_roots = q_poly_sp.real_roots()
        except:
            return float('inf')

        # For each real root r, check sign change at r - 1e-198 and r + 1e-198
        # Use exact rational arithmetic for the sign test
        B = 0
        for r in real_roots:
            if r > 0:
                r_float = float(r)
                # Use a small rational delta for exact sign test
                delta = sp.Rational(1, 10**198)
                try:
                    val_neg = q_poly.subs(x, r - delta)
                    val_pos = q_poly.subs(x, r + delta)
                    # Check if signs are strictly opposite
                    if val_neg != 0 and val_pos != 0:
                        if (val_neg > 0 and val_pos < 0) or (val_neg < 0 and val_pos > 0):
                            if r_float > B:
                                B = r_float
                except:
                    pass

        return B / (2 * math.pi)

    # Evaluate candidates in priority order
    for i, cs in enumerate(unique_sets):
        if time_remaining() < 45:
            break
        try:
            prescreen_score = numerical_prescreen(cs)
            print(f"Prescreen {i}: {prescreen_score:.4f}, time_left={time_remaining():.1f}s")

            # Always evaluate top candidates; evaluate others if prescreen is good
            if i < 3 or (prescreen_score < float('inf') and prescreen_score < 5.0):
                score = evaluate(cs)
                if score < best_score:
                    best_score = score
                    best_z = cs
                    print(f"Candidate {i}: score={score:.6f}, prescreen={prescreen_score:.4f}")
        except Exception as e:
            print(f"Error on candidate {i}: {e}")

    if best_z is not None:
        return best_z
    else:
        # Fallback: Laguerre zeros
        try:
            z12 = laguerre_zeros(12, alpha)
            return list(z12)
        except:
            return bessel_j_half_zeros(12)

# EVOLVE-BLOCK-END
