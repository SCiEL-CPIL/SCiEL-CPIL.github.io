"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Multi-phase model-and-optimize:
  1. Fast exact objective via FFT autoconvolution.
  2. Global search over a compact basis expansion (differential evolution).
  3. Smooth-surrogate local optimisation (adaptive beta log-sum-exp).
  4. Basin hopping for escape from local minima.
  5. Peak suppression: targeted counter-perturbation at the max autoconv peak.
  6. Direct local search: sign-flips and small perturbations.
  7. Restart-aware simulated annealing.
  8. Final high-precision polish.

All phases are deadline-guarded so the program always returns within budget.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping


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


def _autoconv_full(h):
    """Full autoconvolution array and its max absolute value."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return ac, np.max(np.abs(ac))


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max|ac| via log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    shifted = abs_ac - m
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return m + log_sum


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
    soft = _soft_max_abs(ac, beta)
    return soft / (s * s)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _make_basis_functions(n, n_basis=36):
    """Return an (n_basis, n) array of smooth structural basis functions."""
    x = np.arange(n, dtype=float)
    basis = []
    basis.append(np.ones(n))                       # 0: constant
    for k in range(1, 12):                          # 1-11: cosines
        basis.append(np.cos(2 * np.pi * k * x / n))
    for k in range(1, 12):                          # 12-21: sines
        basis.append(np.sin(2 * np.pi * k * x / n))
    # Gaussian bumps (wider for more structural flexibility)
    for frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = frac * n
        sigma = n / 10.0
        basis.append(np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    # Quadratic
    basis.append(((x - n / 2) / (n / 2)) ** 2)
    # Linear ramp
    basis.append(x / n - 0.5)
    # Alternating (captures Nyquist-like structure)
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    # Block patterns (disjoint rectangular pulses with alternating sign)
    bs = n // 8
    for b in range(4):
        blk = np.zeros(n)
        blk[b * bs:(b + 1) * bs] = 1.0
        blk[(b + 4) * bs:(b + 5) * bs] = -1.0
        basis.append(blk)
    # Higher-frequency cosines
    for k in [14, 18, 24]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    # Cubic polynomial
    basis.append(((x - n / 2) / (n / 2)) ** 3)
    return np.array(basis[:n_basis])


# ---------------------------------------------------------------------------
# Local search routines
# ---------------------------------------------------------------------------

def _local_search_flip(h, max_passes=5, deadline=None):
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
    no_improve_count = 0
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
                no_improve_count = 0
            else:
                h[i] = old
                no_improve_count += 1
        if not improved:
            break
        if no_improve_count > N // 2:
            current_step *= 0.7
        else:
            current_step = min(current_step * 1.05, step_size)
    return h, best_val


def _gaussian_perturbation(h, scale=0.06, deadline=None, max_iters=400):
    """Restart-aware simulated annealing with Gaussian perturbations."""
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    T_start, T_end = 0.5, 0.003
    restart_interval = 80
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
        scale *= 0.997
    return h_best, best_val


def _peak_suppression_search(h, deadline=None, max_iters=120):
    """Targeted search: reduce the maximum autoconvolution peak."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(888)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        ac, max_abs = _autoconv_full(h)
        if ac is None:
            break
        peak_idx = int(np.argmax(np.abs(ac)))
        k = peak_idx
        valid_j = [j for j in range(N) if 0 <= k - j < N]
        if not valid_j:
            continue
        n_perturb = min(5, len(valid_j))
        perturbed_idx = rng.choice(valid_j, size=n_perturb, replace=False)
        h_trial = h.copy()
        for j in perturbed_idx:
            sign = np.sign(ac[k])
            delta = rng.uniform(0.01, 0.05) * (-sign)
            h_trial[j] += delta
            kj = k - j
            if 0 <= kj < N and kj != j:
                h_trial[kj] += delta * 0.5
        trial_val = _score(h_trial)
        if trial_val < best_val:
            best_val = trial_val
            h = h_trial.copy()
    return h, best_val


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t_start = time.time()
    deadline = t_start + 145.0   # 5 s safety margin before 150 s kill

    best_score = np.inf
    best_h = np.ones(N) / N
    base_score = _score(best_h)
    best_score = base_score

    n_basis = 36
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

    # ---- Phase 2: smooth surrogate optimization (adaptive beta) ----
    p2_dl = deadline - 30
    if time.time() < p2_dl:
        inits = []
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
        inits.append(params_init)

        rng = np.random.default_rng(123)
        for i in range(8):
            params_rand = rng.standard_normal(n_basis) * 0.4
            inits.append(params_rand)

        for amp in [0.2, 0.5, 0.8]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)

        for params_init in inits[:10]:
            if time.time() > p2_dl - 5:
                break
            try:
                for beta in [50.0, 100.0, 200.0]:
                    if time.time() > p2_dl - 3:
                        break

                    def smooth_obj(params, b=beta):
                        if time.time() > p2_dl - 2:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, beta=b)

                    res_smooth = minimize(
                        smooth_obj, params_init, method='BFGS',
                        options={'maxiter': 200, 'disp': False, 'gtol': 1e-7}
                    )
                    h_smooth = np.dot(res_smooth.x, basis)
                    s_smooth = np.sum(h_smooth)
                    if abs(s_smooth) > 1e-6:
                        h_smooth = h_smooth / s_smooth
                        params_init = np.linalg.lstsq(basis.T, h_smooth, rcond=None)[0]

                # Final refinement with true objective
                def lbfgs_obj(params):
                    if time.time() > p2_dl - 1:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)

                res_lbfgs = minimize(
                    lbfgs_obj, params_init, method='L-BFGS-B',
                    options={'maxiter': 200, 'disp': False, 'gtol': 1e-8}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # ---- Phase 3: basin hopping ----
    p3_dl = deadline - 15
    if time.time() < p3_dl:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

            def basin_obj(params):
                if time.time() > p3_dl:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)

            res_bh = basinhopping(
                basin_obj, params_curr,
                niter=25, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B',
                                  'options': {'maxiter': 100, 'gtol': 1e-7}},
                seed=42
            )
            if res_bh.fun < best_score:
                h_bh = np.dot(res_bh.x, basis)
                s_bh = np.sum(h_bh)
                if abs(s_bh) > 1e-6:
                    h_bh = h_bh / s_bh
                    val_bh = _score(h_bh)
                    if val_bh < best_score:
                        best_score = val_bh
                        best_h = h_bh.copy()
        except Exception:
            pass

    # ---- Phase 4: peak suppression ----
    p4_dl = deadline - 10
    if time.time() < p4_dl:
        h_peak, val_peak = _peak_suppression_search(best_h, deadline=p4_dl, max_iters=100)
        if val_peak < best_score:
            best_score = val_peak
            best_h = h_peak.copy()

    # ---- Phase 5: local search (flip + perturb) ----
    p5_dl = deadline - 5
    if time.time() < p5_dl:
        h_flip, val_flip = _local_search_flip(best_h, max_passes=5, deadline=p5_dl - 2)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()

        if time.time() < p5_dl - 1:
            h_pert, val_pert = _local_search_perturb(
                best_h, max_passes=5, deadline=p5_dl - 0.5, step_size=0.03
            )
            if val_pert < best_score:
                best_score = val_pert
                best_h = h_pert.copy()

    # ---- Phase 6: restart-aware simulated annealing ----
    if time.time() < deadline - 2:
        h_sa, val_sa = _gaussian_perturbation(
            best_h, scale=0.05, deadline=deadline - 1.0, max_iters=400
        )
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()

    # ---- Phase 7: final high-precision polish ----
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

            def final_obj(params):
                if time.time() > deadline - 0.2:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)

            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 300, 'disp': False, 'gtol': 1e-10}
            )
            h_final = np.dot(res_final.x, basis)
            s_final = np.sum(h_final)
            if abs(s_final) > 1e-6:
                h_final = h_final / s_final
                val_final = _score(h_final)
                if val_final < best_score:
                    best_score = val_final
                    best_h = h_final.copy()
        except Exception:
            pass

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
