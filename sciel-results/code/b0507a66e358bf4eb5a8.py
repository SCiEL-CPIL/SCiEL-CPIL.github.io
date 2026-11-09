import sympy as sp
import numpy as np
import time
import math
import mpmath as mp

def _gen_laguerre(n, alpha, x_val):
    """Compute L_n^{(alpha)}(x) using explicit sum (for mpmath)."""
    total = mp.mpf(0)
    for j in range(n + 1):
        total += ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
    return total

def _gen_laguerre_deriv(n, alpha, x_val):
    """d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)."""
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

def evaluate_candidate_smooth(z_list, temp=0.1):
    """
    Smooth surrogate: log-sum-exp of positive real roots of q.
    Returns a smooth approximation to B for optimization.
    """
    try:
        z = sorted([float(zz) for zz in z_list])
        if any(zi <= 0 for zi in z):
            return 1e10
        for i in range(len(z) - 1):
            if z[i + 1] - z[i] < 1e-5:
                return 1e10

        mp.mp.dps = 30
        alpha = mp.mpf(-1) / 2

        # Build matrix and solve
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
        upper_bound = max_z * 1.5 + 50

        # Coarse scan for sign changes
        N_grid = 1500
        xs = np.linspace(1e-4, upper_bound, N_grid)
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
                # Bisect to find root
                a = mp.mpf(xs[i - 1]) if i > 0 else mp.mpf(1e-4)
                b = mp.mpf(xi)
                try:
                    r = mp.findroot(lambda t: q_val(t), (a, b))
                    r_f = float(r)
                    if 0 < r_f < upper_bound:
                        is_z = any(abs(r_f - zi) < 1e-8 for zi in z)
                        if not is_z:
                            roots_found.append(r_f)
                except Exception:
                    pass
            prev_q = qi

        if not roots_found:
            return 1e10

        # Log-sum-exp for smooth max
        roots_arr = np.array(roots_found)
        max_r = np.max(roots_arr)
        if max_r < 1e-10:
            return 0.0
        # Smooth approximation: use log-sum-exp with temperature
        shifted = roots_arr / max_r
        log_sum_exp = np.log(np.sum(np.exp(shifted / temp))) * max_r
        return float(log_sum_exp)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 560.0  # Safety margin

    np.random.seed(42)

    best_B = float('inf')
    best_z = None

    # Multi-start random search
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() > deadline - 120:
            break

        # Log-uniform random initialization
        z_init = np.sort(np.exp(np.random.uniform(np.log(0.1), np.log(100), 12)))
        
        # Ensure distinctness
        valid = True
        for i in range(len(z_init) - 1):
            if z_init[i + 1] - z_init[i] < 0.01:
                valid = False
                break
        if not valid:
            continue

        # Try this starting point with Nelder-Mead on smooth surrogate
        from scipy.optimize import minimize

        def obj(z):
            return evaluate_candidate_smooth(z, temp=0.05)

        try:
            res = minimize(obj, z_init, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            z_opt = sorted(res.x)
            if all(zi > 0 for zi in z_opt) and len(set(round(zi, 6) for zi in z_opt)) == 12:
                B_exact = evaluate_candidate_exact(z_opt)
                if B_exact is not None and B_exact < best_B:
                    best_B = B_exact
                    best_z = z_opt
        except Exception:
            pass

    # Also try a few Bessel-based variants as fallback
    if time.time() < deadline - 100:
        j0_zeros = [
            2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
            14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
            27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
        ]
        bessel_sq = [z * z for z in j0_zeros]
        B_bessel = evaluate_candidate_exact(bessel_sq)
        if B_bessel is not None and B_bessel < best_B:
            best_B = B_bessel
            best_z = bessel_sq

    if best_z is None:
        # Fallback: return Bessel squares
        j0_zeros = [
            2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
            14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
            27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
        ]
        best_z = [z * z for z in j0_zeros]

    return best_z
