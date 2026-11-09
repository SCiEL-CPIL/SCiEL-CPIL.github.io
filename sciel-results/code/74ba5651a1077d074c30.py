"""Combined program: Bessel-zero initialization + targeted local optimization + exact sympy verification."""
import time
import numpy as np
from scipy.special import eval_genlaguerre, jn_zeros
from scipy.optimize import brentq, minimize

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 620  # Safety margin before 630s kill
    best_score = float('inf')
    best_z = None

    # --- Fast numerical evaluation (proxy for B/(2*pi)) ---
    def evaluate_numeric(z_list):
        """Fast floating-point evaluation to approximate B/(2*pi)."""
        try:
            alpha = -0.5
            z_arr = np.array(sorted(z_list), dtype=np.float64)
            if np.any(z_arr <= 0) or len(set(np.round(z_arr, 12))) != 12:
                return float('inf')

            # Build system: g(x) = sum c_k L_{2k}^{alpha}(x)
            A_rows = []
            b_vec = []

            # g(0) = 0
            row = np.array([eval_genlaguerre(2*k, alpha, 0.0) for k in range(26)])
            A_rows.append(row)
            b_vec.append(0.0)

            # g'(0) = 1: L_{2k}'(x) = -L_{2k-1}^{alpha+1}(x)
            row = np.zeros(26)
            row[0] = 0.0
            for k in range(1, 26):
                row[k] = -eval_genlaguerre(2*k - 1, alpha + 1, 0.0)
            A_rows.append(row)
            b_vec.append(1.0)

            for zi in z_arr:
                # g(zi) = 0
                row = np.array([eval_genlaguerre(2*k, alpha, zi) for k in range(26)])
                A_rows.append(row)
                b_vec.append(0.0)

                # g'(zi) = 0
                row = np.zeros(26)
                row[0] = 0.0
                for k in range(1, 26):
                    row[k] = -eval_genlaguerre(2*k - 1, alpha + 1, zi)
                A_rows.append(row)
                b_vec.append(0.0)

            A_mat = np.array(A_rows)
            b_mat = np.array(b_vec)

            cond = np.linalg.cond(A_mat)
            if cond > 1e14:
                return float('inf')

            try:
                c_sol = np.linalg.solve(A_mat, b_mat)
            except:
                return float('inf')

            def g_func(x_val):
                return sum(c_sol[k] * eval_genlaguerre(2*k, alpha, x_val) for k in range(26))

            def denom_func(x_val):
                result = x_val
                for zi in z_arr:
                    result *= (x_val - zi) ** 2
                return result

            def q_func(x_val):
                d = denom_func(x_val)
                if abs(d) < 1e-300:
                    return 0.0
                return g_func(x_val) / d

            # Find real roots of q(x) by scanning for sign changes
            x_max = max(z_arr[-1] * 3, 50.0)
            x_min = 1e-8

            n_scan = 4000
            x_scan = np.linspace(x_min, x_max, n_scan)
            q_scan = np.array([q_func(xi) for xi in x_scan])

            signs = np.sign(q_scan)
            sign_changes = []
            for i in range(n_scan - 1):
                if signs[i] * signs[i+1] < 0:
                    try:
                        root = brentq(q_func, x_scan[i], x_scan[i+1], xtol=1e-14, rtol=1e-12)
                        sign_changes.append(root)
                    except:
                        pass

            if len(sign_changes) == 0:
                return 0.0

            B = max(sign_changes)
            return B / (2 * np.pi)

        except Exception:
            return float('inf')

    # --- Exact sympy evaluation (for final verification) ---
    def evaluate_exact(z_list):
        """Exact rational evaluation using sympy."""
        import sympy as sp
        x = sp.Symbol('x')
        alpha_val = sp.Rational(-1, 2)

        z_rats = sorted([sp.Rational(float(zi)) for zi in z_list])
        if len(set(z_rats)) != 12:
            return float('inf')
        if any(z <= 0 for z in z_rats):
            return float('inf')

        # Precompute L_{2k}^{alpha}(x) and derivatives
        L_terms = []
        L_derivs = []
        for k in range(26):
            L = sp.laguerre(2*k, x, alpha_val)
            L_terms.append(L)
            L_derivs.append(sp.diff(L, x))

        # Build system matrix
        A_rows = []
        b_vec = []

        A_rows.append([L_terms[k].subs(x, 0) for k in range(26)])
        b_vec.append(0)

        A_rows.append([L_derivs[k].subs(x, 0) for k in range(26)])
        b_vec.append(1)

        for zi in z_rats:
            A_rows.append([L_terms[k].subs(x, zi) for k in range(26)])
            b_vec.append(0)
            A_rows.append([L_derivs[k].subs(x, zi) for k in range(26)])
            b_vec.append(0)

        A_mat = sp.Matrix(A_rows)
        b_mat = sp.Matrix(b_vec)

        if A_mat.det() == 0:
            return float('inf')

        try:
            c_sol = A_mat.LUsolve(b_mat)
        except:
            return float('inf')

        g_sol = sum(c_sol[k] * L_terms[k] for k in range(26))
        g_sol = sp.expand(g_sol)

        denom = x
        for zi in z_rats:
            denom = denom * (x - zi)**2
        denom = sp.expand(denom)

        q_poly, remainder = sp.div(g_sol, denom, x)

        try:
            q_poly_sp = sp.Poly(q_poly, x)
            real_roots = q_poly_sp.real_roots()
        except:
            return float('inf')

        B = 0
        for r in real_roots:
            if r > 0:
                r_float = float(r)
                delta = 1e-198
                try:
                    val_neg = q_poly.subs(x, r - delta)
                    val_pos = q_poly.subs(x, r + delta)
                    vn = float(val_neg)
                    vp = float(val_pos)
                    if vn * vp < 0:
                        if r_float > B:
                            B = r_float
                except:
                    pass

        return B / (2 * np.pi)

    # --- Local optimization using L-BFGS-B with constraints ---
    def optimize_z(z_init, max_time=45):
        """Use L-BFGS-B to minimize B/(2*pi) locally with positivity constraints."""
        def objective(params):
            if time.time() > deadline - 30:
                return float('inf')
            z_candidate = sorted(params)
            if np.any(z_candidate <= 1e-6):
                return float('inf')
            if len(set(np.round(z_candidate, 10))) != 12:
                return float('inf')
            return evaluate_numeric(z_candidate)

        try:
            bounds = [(1e-6, 100.0)] * 12
            result = minimize(
                objective,
                np.array(sorted(z_init)),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < float('inf'):
                z_opt = sorted(result.x)
                if np.all(z_opt > 1e-6) and len(set(np.round(z_opt, 10))) == 12:
                    return z_opt, result.fun
        except:
            pass
        return None, float('inf')

    # --- Generate initial candidates using Bessel-zero asymptotics ---
    candidates = []

    # Primary: J_0(x/2) zeros scaled to match L_12^{(-1/2)} zero distribution
    # For large n, zeros of L_n^{(alpha)}(x) ~ 4*n * j_{alpha+1/2,k}^2 / n^2 = 4*j_k^2/n
    # More precisely: x_k ~ 4*n * (j_{alpha+1/2,k}/2)^2 / n = j_k^2 for alpha=-1/2
    # Actually: L_n^{(alpha)}(x) ~ (2/sqrt(pi)) * x^{(1-alpha)/2} * J_{alpha+1/2}(x/2) / 2^{(alpha+1)/2} * ...
    # Zeros of J_0(x/2) are at x/2 = j_{0,k}, so x = 2*j_{0,k}
    # But we need to match the scale. Let's use empirical scaling.
    
    # Get first 12 zeros of J_0
    j0_zeros = jn_zeros(0, 12)  # Returns array of 12 zeros
    # Scale: for L_n^{(alpha)}, the k-th zero is approximately 4*n*(j_{alpha+1/2,k}/2)^2 / (n + alpha/2 + 1/4)
    # For alpha = -1/2, n = 12: x_k ~ 4*12*(j_{0,k}/2)^2 / (12 - 1/4) = 48*j_{0,k}^2/4 / 11.75 = 12*j_{0,k}^2/11.75
    scale_factor = 12.0 / 11.75
    z_bessel = [scale_factor * (j0_zeros[i] ** 2) for i in range(12)]
    candidates.append(z_bessel)
    
    # Also try with J_{-1/2} zeros (since alpha = -1/2, alpha+1/2 = 0, but let's try -1/2 too)
    # J_{-1/2}(x) = sqrt(2/(pi*x)) * cos(x)
    # Zeros of cos(x) are at (k+1/2)*pi
    j_minus_half_zeros = [(k + 0.5) * np.pi for k in range(12)]
    z_bessel_alt = [scale_factor * (j_minus_half_zeros[i] ** 2) for i in range(12)]
    candidates.append(z_bessel_alt)
    
    # Laguerre zeros of L_12^(-0.5) using three-term recurrence
    def laguerre_zeros(n, alpha):
        a = alpha
        diag = np.array([2.0*k - 1.0 + a for k in range(1, n+1)])
        subdiag = np.array([np.sqrt(k*(k+a)) for k in range(1, n)])
        J = np.diag(diag)
        for i in range(n-1):
            J[i, i+1] = subdiag[i]
            J[i+1, i] = subdiag[i]
        eigvals = np.linalg.eigvalsh(J)
        return np.sort(eigvals)

    try:
        z12 = laguerre_zeros(12, -0.5)
        if len(z12) == 12 and all(z > 0 for z in z12):
            candidates.append(list(z12))
    except:
        pass

    # Add scaled versions of Bessel-based candidates
    for scale in [0.5, 0.7, 1.5, 2.0]:
        candidates.append([scale * z for z in z_bessel])
        candidates.append([scale * z for z in z_bessel_alt])

    # Log-spaced candidates
    for lo, hi in [(0.5, 30), (1, 50), (0.1, 20), (2, 80)]:
        z = list(np.logspace(np.log10(lo), np.log10(hi), 12))
        candidates.append(z)

    # Linear spaced
    for lo, hi in [(0.5, 30), (1, 50), (2, 80)]:
        z = list(np.linspace(lo, hi, 12))
        candidates.append(z)

    # Deduplicate
    unique_candidates = []
    seen = set()
    for cs in candidates:
        key = tuple(round(v, 4) for v in sorted(cs))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cs)

    # --- Phase 1: Quick numerical screening ---
    screened = []
    for i, cs in enumerate(unique_candidates):
        if time.time() > deadline - 120:
            break
        try:
            score = evaluate_numeric(cs)
            if score < float('inf'):
                screened.append((score, cs))
                if score < best_score:
                    best_score = score
                    best_z = cs
        except:
            pass

    # Sort by score, take top candidates for optimization
    screened.sort(key=lambda x: x[0])
    top_candidates = [cs for _, cs in screened[:15]]

    # --- Phase 2: Local optimization on top candidates ---
    for i, cs in enumerate(top_candidates):
        if time.time() > deadline - 60:
            break
        try:
            z_opt, opt_score = optimize_z(cs, max_time=45)
            if z_opt is not None and opt_score < float('inf'):
                if opt_score < best_score:
                    best_score = opt_score
                    best_z = z_opt
        except:
            pass

    # --- Phase 3: Exact verification of best candidate ---
    if best_z is not None:
        if time.time() < deadline - 30:
            try:
                exact_score = evaluate_exact(best_z)
                if exact_score < float('inf'):
                    best_score = exact_score
            except:
                pass

    if best_z is not None:
        result = sorted([float(z) for z in best_z])
        if len(result) == 12 and all(z > 0 for z in result) and len(set(result)) == 12:
            return result

    # Fallback
    return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
