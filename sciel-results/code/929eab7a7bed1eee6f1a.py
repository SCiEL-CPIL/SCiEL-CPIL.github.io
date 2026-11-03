"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Strategy:
  1.  Fast, exact objective via FFT autoconvolution.
  2.  Global search over a compact basis expansion (differential evolution).
  3.  Smooth-surrogate local optimisation (adaptive beta log-sum-exp).
  4.  Direct local search: sign-flips and small perturbations.
  5.  Final high-precision polish.

All phases are deadline-guarded so the program always returns within 150 s.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _autoconv_max_abs(h):
    """max |h*h| (linear autoconvolution) computed via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_smooth(h, beta=80.0):
    """Differentiable surrogate: soft_max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    shifted = abs_ac - m
    soft = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return soft / (s * s)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _make_basis_functions(n, n_basis=32):
    """Return an (n_basis, n) array of smooth basis functions."""
    x = np.arange(n, dtype=float)
    basis = []
    basis.append(np.ones(n))                       # 0: constant
    for k in range(1, 12):                          # 1-11: cosines
        basis.append(np.cos(2 * np.pi * k * x / n))
    for k in range(1, 12):                          # 12-21: sines
        basis.append(np.sin(2 * np.pi * k * x / n))
    # Gaussian bumps
    for frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = frac * n
        sigma = n / 12.0
        basis.append(np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    # Quadratic
    basis.append(((x - n / 2) / (n / 2)) ** 2)
    # Linear ramp
    basis.append(x / n - 0.5)
    # Alternating
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    # Block patterns (4 of them)
    bs = n // 8
    for b in range(4):
        blk = np.zeros(n)
        blk[b * bs:(b + 1) * bs] = 1.0
        blk[(b + 4) * bs:(b + 5) * bs] = -1.0
        basis.append(blk)
    return np.array(basis[:n_basis])


# ---------------------------------------------------------------------------
# Local search routines
# ---------------------------------------------------------------------------

def _local_search_flip(h, max_passes=6, deadline=None):
    """Try flipping each element's sign; keep if it improves the score."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(555)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val


def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.04):
    """Coordinate-descent style: try small perturbations per element."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    current_step = step_size
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old = h[i]
            best_local = old
            best_local_val = best_val
            for delta in (current_step, -current_step, 0.5 * current_step, -0.5 * current_step):
                h[i] = old + delta
                v = _score(h)
                if v < best_local_val:
                    best_local_val = v
                    best_local = h[i]
            if best_local != old:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old
        if not improved:
            break
        current_step *= 0.85
    return h, best_val


def _gaussian_perturbation(h, scale=0.06, deadline=None, max_iters=300):
    """Simulated-annealing style search with Gaussian perturbations."""
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    T_start, T_end = 0.4, 0.003
    restart_interval = 60
    no_improve = 0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, max_iters - 1))
        if rng.random() < 0.06:
            pert = rng.standard_normal(N) * scale * 3.0
        else:
            pert = rng.standard_normal(N) * scale
        h_trial = h_curr + pert
        trial_val = _score(h_trial)
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > restart_interval and it < max_iters - 10:
            h_curr = h_best.copy()
            curr_val = best_val
            no_improve = 0
        scale *= 0.996
    return h_best, best_val


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t_start = time.time()
    deadline = t_start + 145.0   # 5 s safety margin

    best_score = np.inf
    best_h = np.ones(N) / N
    base_score = _score(best_h)
    best_score = base_score

    n_basis = 32
    basis = _make_basis_functions(N, n_basis)

    # ---- Phase 1: differential evolution on basis coefficients ----
    p1_dl = deadline - 50
    if time.time() < p1_dl:
        bounds = [(-2.0, 2.0)] * n_basis

        def de_obj(params):
            if time.time() > p1_dl:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)

        try:
            res = differential_evolution(
                de_obj, bounds, seed=42, maxiter=40, tol=1e-6,
                popsize=14, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            h_de = np.dot(res.x, basis)
            s_de = np.sum(h_de)
            if abs(s_de) > 1e-6:
                h_de /= s_de
                v = _score(h_de)
                if v < best_score:
                    best_score = v
                    best_h = h_de.copy()
        except Exception:
            pass

    # ---- Phase 2: smooth-surrogate local optimisation (adaptive beta) ----
    p2_dl = deadline - 28
    if time.time() < p2_dl:
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr /= s_curr
        params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

        rng = np.random.default_rng(123)
        inits = [params.copy()]
        for _ in range(6):
            inits.append(rng.standard_normal(n_basis) * 0.5)

        for p_init in inits[:7]:
            if time.time() > p2_dl - 4:
                break
            try:
                for beta in [60.0, 120.0, 250.0]:
                    if time.time() > p2_dl - 2:
                        break

                    def sobj(params, b=beta):
                        if time.time() > p2_dl - 1:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, beta=b)

                    res_s = minimize(sobj, p_init, method='BFGS',
                                     options={'maxiter': 200, 'gtol': 1e-8})
                    h_s = np.dot(res_s.x, basis)
                    s_s = np.sum(h_s)
                    if abs(s_s) > 1e-6:
                        h_s /= s_s
                        p_init = np.linalg.lstsq(basis.T, h_s, rcond=None)[0]

                # Final refinement with true objective
                def tobj(params):
                    if time.time() > p2_dl - 0.5:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)

                res_t = minimize(tobj, p_init, method='L-BFGS-B',
                                 options={'maxiter': 300, 'gtol': 1e-10})
                h_t = np.dot(res_t.x, basis)
                s_t = np.sum(h_t)
                if abs(s_t) > 1e-6:
                    h_t /= s_t
                    v = _score(h_t)
                    if v < best_score:
                        best_score = v
                        best_h = h_t.copy()
            except Exception:
                pass

    # ---- Phase 3: peak-suppression targeted search ----
    p3_dl = deadline - 14
    if time.time() < p3_dl:
        h_ps = best_h.copy()
        best_ps = best_score
        rng = np.random.default_rng(888)
        base_step = 0.03
        for _ in range(80):
            if time.time() > p3_dl:
                break
            n = len(h_ps)
            size = 2 * n - 1
            fsize = 1
            while fsize < size:
                fsize *= 2
            H = np.fft.rfft(h_ps, fsize)
            ac = np.fft.irfft(H * H, fsize)[:size]
            abs_ac = np.abs(ac)
            peak_idx = np.argmax(abs_ac)
            sign = np.sign(ac[peak_idx])
            valid = [j for j in range(N) if 0 <= peak_idx - j < N]
            if not valid:
                continue
            n_pert = min(3, len(valid))
            pidx = rng.choice(valid, size=n_pert, replace=False)
            h_trial = h_ps.copy()
            for j in pidx:
                delta = rng.uniform(0.01, base_step) * (-sign)
                h_trial[j] += delta
                kj = peak_idx - j
                if 0 <= kj < N and kj != j:
                    h_trial[kj] += delta * 0.5
            v = _score(h_trial)
            if v < best_ps:
                best_ps = v
                h_ps = h_trial.copy()
                base_step = max(0.01, base_step * 0.95)
        if best_ps < best_score:
            best_score = best_ps
            best_h = h_ps.copy()

    # ---- Phase 4: sign-flip local search ----
    p4_dl = deadline - 8
    if time.time() < p4_dl:
        h_fl, v_fl = _local_search_flip(best_h, max_passes=6, deadline=p4_dl)
        if v_fl < best_score:
            best_score = v_fl
            best_h = h_fl.copy()

    # ---- Phase 5: perturbation local search ----
    p5_dl = deadline - 4
    if time.time() < p5_dl:
        h_pt, v_pt = _local_search_perturb(best_h, max_passes=5,
                                           deadline=p5_dl, step_size=0.03)
        if v_pt < best_score:
            best_score = v_pt
            best_h = h_pt.copy()

    # ---- Phase 6: simulated-annealing polish ----
    p6_dl = deadline - 2
    if time.time() < p6_dl:
        h_sa, v_sa = _gaussian_perturbation(best_h, scale=0.04,
                                            deadline=p6_dl, max_iters=400)
        if v_sa < best_score:
            best_score = v_sa
            best_h = h_sa.copy()

    # ---- Phase 7: final high-precision polish ----
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr /= s_curr
            params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

            def final_obj(params):
                if time.time() > deadline - 0.2:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)

            res_f = minimize(final_obj, params, method='L-BFGS-B',
                             options={'maxiter': 500, 'gtol': 1e-12})
            h_f = np.dot(res_f.x, basis)
            s_f = np.sum(h_f)
            if abs(s_f) > 1e-6:
                h_f /= s_f
                v_f = _score(h_f)
                if v_f < best_score:
                    best_score = v_f
                    best_h = h_f.copy()
        except Exception:
            pass

    # ---- Phase 8: final sign-flip polish ----
    if time.time() < deadline - 0.3:
        try:
            h_ff, v_ff = _local_search_flip(best_h, max_passes=2,
                                            deadline=deadline - 0.1)
            if v_ff < best_score:
                best_score = v_ff
                best_h = h_ff.copy()
        except Exception:
            pass

    # Safety: never worse than baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
