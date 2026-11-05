"""Growth-index compact approximation; EVOLVE-BLOCK holds the optimizer + construct."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
_BEST_ERR = np.inf
_A_GRID = None

def _deadline():
    return _DEADLINE - 3.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _DEADLINE + 3.0

def _Om(a, Om0, w0, wa):
    # matter fraction at scale factor a
    # Om(a) = Om0 * a^-3 / (Om0*a^-3 + Ode0 * exp(3*int_1^a (1+w(a'))/a' da'))
    # For CPL: integral = (1+w0+wa)*(1-a)  ... check: (1+w(a'))/a' = (1+w0+wa(1-a'))/a'
    #   = (1+w0+wa)/a' - wa. Integral from 1 to a: (1+w0+wa)*ln(a) - wa*(a-1)
    # Ode(a) = Ode0 * exp(3*((1+w0+wa)*ln(a) - wa*(a-1)))
    #         = Ode0 * a^(3*(1+w0+wa)) * exp(-3*wa*(a-1))
    Ode0 = 1.0 - Om0
    if Ode0 < 1e-12:
        return np.ones_like(a)
    log_den = np.log(np.maximum(a, 1e-30)) * (3.0 * (1.0 + w0 + wa)) - 3.0 * wa * (a - 1.0)
    ode_a = Ode0 * np.exp(log_den)
    mat_a = Om0 / (a ** 3)
    return mat_a / (mat_a + ode_a)

def _exact_f(a, Om0, w0, wa):
    # f = Om^gamma where gamma from the ODE? No - the task says f=Om(a)^gamma with OUR gamma.
    # But the REFERENCE uses the true GR growth. The true f(a) = d ln D / d ln a.
    # We need the true growth rate from the ODE for scoring.
    # f_true = (3/2) Om + (3/10)(1 - w) Om   (for slow-roll / standard approx) is NOT exact.
    # The exact f comes from solving the ODE. We'll compute D via ODE and f = d ln D/d ln a.
    raise RuntimeError("use _exact_D")

def _exact_D(a_grid, Om0, w0, wa):
    # Solve D'' + (3/a + H'/H) D' - (3/2) Om(a)/a^2 D = 0
    # Work in x = ln a. Let F = dD/dx. dF/dx + (3 + H'/H) F = (3/2) Om(a) D
    # H'/H = (d ln H / d ln a) = (3/2) sum Om_i(a) (1 + w_i(a))  ... for CPL:
    #   H'/H = (3/2)[ Om(a)(1+0) + Ode(a)(1+w(a)) ]
    # where Ode(a) = 1 - Om(a), w(a) = w0 + wa(1-a)
    from scipy.integrate import solve_ivp
    a = a_grid
    x = np.log(a)
    Om = _Om(a, Om0, w0, wa)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    def rhs(xv, yv):
        ai = np.exp(xv)
        Omi = _Om(ai, Om0, w0, wa)
        Odei = 1.0 - Omi
        wi = w0 + wa * (1.0 - ai)
        Hlog = 1.5 * (Omi + Odei * (1.0 + wi))
        dD = yv[1]
        dF = -(3.0 + Hlog) * yv[1] + 1.5 * Omi * yv[0]
        return [dD, dF]
    # integrate from a=1 (x=0) backward to a_min
    x0 = 0.0
    y0 = [1.0, 1.0]  # D(1)=1, F(1)=dD/dx at 1 ~ f(1)
    sol = solve_ivp(rhs, [x0, x[0]], y0, t_eval=x, method='RK45', rtol=1e-10, atol=1e-12)
    if not sol.success:
        return None, None
    D = sol.y[0]
    F = sol.y[1]
    f = F / D  # d ln D / d ln a
    return D, f

def _score(coeffs, cosmo_list, a_grid):
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    for (Om0, w0, wa) in cosmo_list:
        a = a_grid
        Om = _Om(a, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        # enforce gamma in [0.1,1.2]
        if gamma.min() < 0.1 or gamma.max() > 1.2:
            return np.inf
        f_app = Om ** gamma
        D, f_true = _exact_D(a, Om0, w0, wa)
        if D is None:
            return np.inf
        # normalized D
        D_app = np.exp(-np.trapz(f_app / a, np.log(a)))
        D_app = D_app / D_app[-1]
        D_true = D / D[-1]
        err_f = np.max(np.abs((f_app - f_true) / f_true))
        err_D = np.max(np.abs((D_app - D_true) / D_true))
        e = max(err_f, err_D)
        if e > max_err:
            max_err = e
    return max_err

def _refine(coeffs0, cosmo_list, a_grid, steps=400, scale=0.05):
    best = list(coeffs0)
    best_err = _score(best, cosmo_list, a_grid)
    for _ in range(steps):
        if time.time() > _deadline():
            break
        improved = False
        for i in range(6):
            for d in (1.0, -1.0):
                cand = list(best)
                cand[i] += d * scale
                if any(c < -10 or c > 10 for c in cand):
                    continue
                e = _score(cand, cosmo_list, a_grid)
                if e < best_err:
                    best = cand
                    best_err = e
                    improved = True
        if not improved:
            scale *= 0.6
            if scale < 1e-5:
                break
    return best, best_err

def construct():
    global _DEADLINE, _BEST, _BEST_ERR, _A_GRID
    if _DEADLINE is None:
        _DEADLINE = time.time() + 87.0  # 3s margin
    # Build validation-like cosmology grid (plausible CPL parameters)
    Om0s = [0.25, 0.30, 0.35]
    w0s = [-1.0, -0.9, -1.1]
    was = [-0.5, 0.0, 0.5]
    cosmo_list = []
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                Ode0 = 1.0 - Om0
                if Ode0 <= 0:
                    continue
                cosmo_list.append((Om0, w0, wa))
    # restrict to keep runtime reasonable
    cosmo_list = cosmo_list[:12]
    a_grid = np.logspace(np.log10(0.05), 0.0, 40)
    _A_GRID = a_grid

    # Precompute exact references once
    ref_cache = {}
    for (Om0, w0, wa) in cosmo_list:
        D, f = _exact_D(a_grid, Om0, w0, wa)
        if D is not None:
            ref_cache[(Om0, w0, wa)] = (D, f)

    # Fast score using cached references
    def fast_score(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        for (Om0, w0, wa) in cosmo_list:
            if (Om0, w0, wa) not in ref_cache:
                continue
            D_true, f_true = ref_cache[(Om0, w0, wa)]
            a = a_grid
            Om = _Om(a, Om0, w0, wa)
            q = 1.0 - Om
            u = 1.0 + w0 + wa * (1.0 - a)
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            if gamma.min() < 0.1 or gamma.max() > 1.2:
                return np.inf
            f_app = Om ** gamma
            D_app = np.exp(-np.trapz(f_app / a, np.log(a)))
            D_app = D_app / D_app[-1]
            D_t = D_true / D_true[-1]
            err_f = np.max(np.abs((f_app - f_true) / f_true))
            err_D = np.max(np.abs((D_app - D_t) / D_t))
            e = max(err_f, err_D)
            if e > max_err:
                max_err = e
        return max_err

    # Coarse grid search over physically-motivated ranges
    # c0 ~ base gamma near matter-domination (q~1, u~1+wa~1): gamma(1,1,0)=c0+c1+c2+c3+c5
    # We want gamma ~ 0.55 when Om~1, w~ -1 (wa=0, u=1, q=0)... wait q=1-Om, at matter dom Om~1 so q~0.
    # At early times Om~1, q~0, u~1+wa(1-a)~1+wa. For wa=0, u~1. gamma~c0+c3.
    # Set c0+c3 ~ 0.55. Let c0=0.5, c3=0.05.
    ranges = {
        0: (0.30, 0.70, 8),   # c0
        1: (-0.30, 0.30, 6),  # c1 (q term)
        2: (-0.20, 0.20, 5),  # c2 (q^2)
        3: (-0.30, 0.30, 6),  # c3 (u)
        4: (-0.30, 0.30, 6),  # c4 (wa*a)
        5: (-0.30, 0.30, 6),  # c5 (u*q)
    }
    best = None
    best_err = np.inf
    # coordinate descent over coarse grid
    cur = [0.5, 0.0, 0.0, 0.05, 0.0, 0.0]
    cur_err = fast_score(cur)
    if cur_err < best_err:
        best = cur[:]; best_err = cur_err

    # coordinate descent refinement
    for it in range(200):
        if time.time() > _deadline():
            break
        improved = False
        for i in range(6):
            lo, hi, n = ranges[i]
            for v in np.linspace(lo, hi, n):
                cand = cur[:]
                cand[i] = float(v)
                e = fast_score(cand)
                if e < cur_err:
                    cur = cand; cur_err = e; improved = True
                    if e < best_err:
                        best = cur[:]; best_err = e
        if not improved:
            break
    if best is not None and best_err < _BEST_ERR:
        _BEST = best
        _BEST_ERR = best_err
    # clamp to [-10,10]
    out = [float(np.clip(c, -10.0, 10.0)) for c in _BEST]
    return out
# EVOLVE-BLOCK-END
