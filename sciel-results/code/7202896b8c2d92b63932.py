import sympy as sp
import numpy as np
import time
import math

def _gen_laguerre(n, alpha, x_val):
    """Compute L_n^{(alpha)}(x) using explicit sum (for mpmath)."""
    import mpmath as mp
    total = mp.mpf(0)
    for j in range(n + 1):
        total += ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
    return total

def _gen_laguerre_deriv(n, alpha, x_val):
    """d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)."""
    import mpmath as mp
    if n == 0:
        return mp.mpf(0)
    return -_gen_laguerre(n - 1, alpha + 1, x_val)

def evaluate_candidate_exact(z_list):
    """
    Exact sympy evaluation. Returns B (float) or None if invalid.
    """
    try:
        z_rat = [sp.Rational(float(z)) for z in z_list]
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)

        Ls = [sp.Laguerre(2 * k, alpha, x) for k in range(26)]

        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)

        # g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        rhs[0] = 0

        # g'(0) = 1
        for k in range(26):
            M[1, k] = sp.diff(Ls[k], x).subs(x, 0)
        rhs[1] = 1

        idx = 2
        for z in z_rat:
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            for k in range(26):
                M[idx, k] = sp.diff(Ls[k], x).subs(x, z)
            rhs[idx] = 0
            idx += 1

        try:
            sol = M.LUsolve(rhs)
        except Exception:
            return None

        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        g_sol = g.subs(list(zip(cs, sol)))

        denom = x
        for z in z_rat:
            denom *= (x - z) ** 2

        q, rem = sp.div(g_sol, denom)
        if rem != 0:
            return None

        try:
            roots = sp.real_roots(q)
        except Exception:
            return None

        if not roots:
            return 0.0

        max_r = 0.0
        for r in roots:
            try:
                r_float = float(r)
            except Exception:
                continue
            if not np.isfinite(r_float) or r_float <= 0:
                continue
            delta = 1e-198
            try:
                v_minus = float(q.subs(x, r_float - delta))
                v_plus = float(q.subs(x, r_float + delta))
            except Exception:
                continue
            if v_minus * v_plus < 0:
                if r_float > max_r:
                    max_r = r_float
        return float(max_r)
    except Exception:
        return None

def evaluate_candidate_fast(z_list):
    """
    Fast mpmath evaluation. Returns B estimate or large penalty.
    """
    try:
        import mpmath as mp
        z = sorted([float(zz) for zz in z_list])
        if any(zi <= 0 for zi in z):
            return 1e10
        for i in range(len(z) - 1):
            if z[i + 1] - z[i] < 1e-6:
                return 1e10

        mp.mp.dps = 40
        alpha = mp.mpf(-1) / 2

        # Precompute Ls values and derivatives at z's for matrix construction
        M = mp.matrix(26, 26)
        rhs = mp.matrix(26, 1)

        for k in range(26):
            M[0, k] = _gen_laguerre(2 * k, alpha, mp.mpf(0))
        rhs[0] = 0

        for k in range(26):
            M[1, k] = _gen_laguerre_deriv(2 * k, alpha, mp.mpf(0))
        rhs[1] = 1

        idx = 2
        for zz in z:
            zv = mp.mpf(zz)
            for k in range(26):
                M[idx, k] = _gen_laguerre(2 * k, alpha, zv)
            rhs[idx] = 0
            idx += 1
            for k in range(26):
                M[idx, k] = _gen_laguerre_deriv(2 * k, alpha, zv)
            rhs[idx] = 0
            idx += 1

        try:
            sol = mp.lu_solve(M, rhs)
        except Exception:
            return 1e10

        def g_val(xv):
            total = mp.mpf(0)
            for k in range(26):
                total += sol[k] * _gen_laguerre(2 * k, alpha, xv)
            return total

        def denom_val(xv):
            d = xv
            for zz in z:
                d *= (xv - mp.mpf(zz)) ** 2
            return d

        def q_val(xv):
            return g_val(xv) / denom_val(xv)

        max_z = max(z)
        upper_bound = max_z * 1.8 + 80

        # Coarse scan for sign changes
        N_grid = 2000
        xs = np.linspace(1e-5, upper_bound, N_grid)
        prev_q = None
        roots_found = []

        for i, xi in enumerate(xs):
            xv = mp.mpf(xi)
            try:
                qi = q_val(xv)
            except Exception:
                prev_q = None
                continue
            if prev_q is not None and prev_q * qi < 0:
                # Bisect / findroot
                a = mp.mpf(xs[i - 1]) if i > 0 else mp.mpf(1e-5)
                b = mp.mpf(xi)
                try:
                    r = mp.findroot(lambda t: q_val(t), (a, b))
                    r_f = float(r)
                    if 0 < r_f < upper_bound:
                        is_z = any(abs(r_f - zi) < 1e-9 for zi in z)
                        if not is_z:
                            roots_found.append(r_f)
                except Exception:
                    pass
            prev_q = qi

        if not roots_found:
            # Try a broader scan
            prev_q = None
            for xi in np.linspace(1e-5, upper_bound * 2, 500):
                xv = mp.mpf(xi)
                try:
                    qi = q_val(xv)
                except Exception:
                    prev_q = None
                    continue
                if prev_q is not None and prev_q * qi < 0:
                    try:
                        r = mp.findroot(lambda t: q_val(t), (mp.mpf(xi * 0.99), xv))
                        r_f = float(r)
                        if 0 < r_f:
                            is_z = any(abs(r_f - zi) < 1e-9 for zi in z)
                            if not is_z:
                                roots_found.append(r_f)
                    except Exception:
                        pass
                prev_q = qi

        if not roots_found:
            return 1e10
        return max(roots_found)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 570.0

    # Strategy: Use the 12 smallest positive zeros of L_12^(-1/2)(x)
    # This leverages the orthogonality of Laguerre polynomials with respect to x^(-1/2) e^(-x)
    
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    L12 = sp.Laguerre(12, alpha, x)
    
    # Find the roots of L_12^(-1/2)(x)
    try:
        roots = sp.nroots(L12, n=15, maxsteps=100)
        # Filter for real, positive roots
        real_positive_roots = []
        for r in roots:
            r_complex = complex(r)
            if abs(r_complex.imag) < 1e-10 and r_complex.real > 0:
                real_positive_roots.append(float(r_complex.real))
        
        if len(real_positive_roots) < 12:
            # Fallback to Bessel zeros if we don't get 12
            j0_zeros = [
                2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
                14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
                27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
            ]
            bessel_sq = [z * z for z in j0_zeros]
            z_init = sorted(bessel_sq)
        else:
            # Sort and take the 12 smallest
            real_positive_roots.sort()
            z_init = real_positive_roots[:12]
            
    except Exception:
        # Fallback to Bessel zeros
        j0_zeros = [
            2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
            14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
            27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
        ]
        bessel_sq = [z * z for z in j0_zeros]
        z_init = sorted(bessel_sq)

    # Ensure validity
    if any(zi <= 0 for zi in z_init) or len(set(round(zi, 10) for zi in z_init)) != 12:
        # Fallback to Bessel zeros
        j0_zeros = [
            2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
            14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
            27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
        ]
        bessel_sq = [z * z for z in j0_zeros]
        z_init = sorted(bessel_sq)

    # Evaluate the initial candidate
    best_B = float('inf')
    best_z = z_init

    if time.time() < deadline - 60:
        B = evaluate_candidate_exact(z_init)
        if B is not None:
            best_B = B
            best_z = z_init

    # Try a few perturbations to see if we can improve
    # Small random perturbations
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 60:
            break
        
        z_perturbed = [z * (1 + 0.05 * (np.random.rand() - 0.5)) for z in z_init]
        z_perturbed = sorted(z_perturbed)
        
        if any(zi <= 0 for zi in z_perturbed) or len(set(round(zi, 10) for zi in z_perturbed)) != 12:
            continue
            
        B_pert = evaluate_candidate_exact(z_perturbed)
        if B_pert is not None and B_pert < best_B:
            best_B = B_pert
            best_z = z_perturbed

    # If we have time, try Nelder-Mead on the fast evaluator
    if time.time() < deadline - 120:
        from scipy.optimize import minimize
        
        def obj(z):
            return evaluate_candidate_fast(z)
        
        try:
            res = minimize(obj, best_z, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res.fun < best_B:
                z_opt = sorted(res.x)
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 8) for zi in z_opt)) == 12:
                    B_exact = evaluate_candidate_exact(z_opt)
                    if B_exact is not None and B_exact < best_B:
                        best_B = B_exact
                        best_z = z_opt
        except Exception:
            pass

    return best_z
