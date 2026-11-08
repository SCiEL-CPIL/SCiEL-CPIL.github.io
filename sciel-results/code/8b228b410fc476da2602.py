"""
Combined model-and-optimize approach for the second autocorrelation inequality.

Strategy:
  1. DCT-based low-rank superposition as the primary representation.
     This enforces global smoothness and reduces dimensionality, preventing
     high-frequency spikes that degrade the L2-to-L1 ratio.
  2. Multi-stage optimization:
     a. Coarse DCT optimization (few modes, L-BFGS-B)
     b. Direct high-dimensional refinement with smoothing
     c. Nelder-Mead polish
  3. Multi-start with diverse parameter initializations.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _neg_value(h):
    """Negative exact value for minimization."""
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _dct_basis(N, n_modes):
    """Return DCT-II basis matrix of shape (N, n_modes)."""
    k = np.arange(N)
    n = np.arange(n_modes)
    # DCT-II: cos(pi * (2k+1) * n / (2N))
    basis = np.cos(np.pi * np.outer(2 * k + 1, n) / (2 * N))
    return basis


def _smooth_value(h, beta=50.0):
    """
    Smoothed objective for direct optimization.
    Uses log-sum-exp to smooth the max operation in norm_inf.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 0.0
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    # Smooth max using log-sum-exp
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if lse <= 0:
        return 0.0
    val = l2 / (norm_1 * lse)
    return -val


def _flat_top_profile(params):
    """
    Flat-top with cosine taper edges.
    params: [A, center, flat_halfwidth, taper_width]
    """
    A, center, flat_hw, taper_w = params
    A = max(0.0, A)
    flat_hw = max(0.5, flat_hw)
    taper_w = max(0.5, taper_w)
    x = np.arange(N, dtype=float)
    dist = np.abs(x - center)
    # Inside flat region
    mask_flat = dist <= flat_hw
    # Taper region
    mask_taper = (dist > flat_hw) & (dist <= flat_hw + taper_w)
    h = np.zeros(N)
    h[mask_flat] = A
    t = (dist[mask_taper] - flat_hw) / taper_w
    h[mask_taper] = A * 0.5 * (1 + np.cos(np.pi * t))
    return h


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 0: Analytical plateau baseline ---
    for k in range(5, N + 1):
        h = np.zeros(N)
        center = N // 2
        start = max(0, center - k // 2)
        end = min(N, start + k)
        h[start:end] = 1.0
        track(h)

    # Evaluate all-ones
    track(np.ones(N))

    # --- Phase 1: Optimize flat-top parameterization ---
    ft_bounds = [
        (0.1, 10.0),       # A
        (N * 0.1, N * 0.9),  # center
        (1.0, N * 0.4),    # flat_halfwidth
        (0.5, N * 0.3),    # taper_width
    ]

    ft_starts = [
        [1.0, N / 2.0, 10.0, 5.0],
        [1.0, N / 2.0, 15.0, 8.0],
        [1.0, N / 2.0, 20.0, 10.0],
        [1.5, N / 2.0, 8.0, 4.0],
        [0.5, N / 2.0, 25.0, 12.0],
    ]

    for s in ft_starts:
        if time.time() > deadline - 20:
            break
        p0 = np.array(s, dtype=float)

        def ft_obj(params):
            if time.time() > deadline:
                return 1e10
            h = _flat_top_profile(params)
            return -compute_value(h)

        try:
            res = opt.minimize(ft_obj, p0, method='L-BFGS-B', bounds=ft_bounds,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            h = _flat_top_profile(res.x)
            track(h)
        except Exception:
            pass

    # --- Phase 2: DCT-based low-dimensional optimization ---
    n_modes_list = [5, 8, 12, 15]

    for n_modes in n_modes_list:
        if time.time() > deadline - 20:
            break

        basis = _dct_basis(N, n_modes)

        # Initial coefficients: project best_h onto DCT basis
        init_coeffs = basis.T @ best_h
        # Add some random starts
        dct_starts = [
            init_coeffs.copy(),
            init_coeffs + RNG.normal(0, 0.1, n_modes),
            init_coeffs * 0.5 + RNG.normal(0, 0.5, n_modes),
            np.zeros(n_modes),
            np.ones(n_modes) * 0.5,
        ]

        for start in dct_starts:
            if time.time() > deadline - 10:
                break

            def dct_obj(coeffs, n_m=n_modes, basis_mat=basis):
                h = basis_mat @ coeffs
                h = np.maximum(0.0, h)
                return -compute_value(h)

            try:
                res = opt.minimize(
                    dct_obj,
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_dct = basis @ res.x
                h_dct = np.maximum(0.0, h_dct)
                track(h_dct)
            except Exception:
                pass

    # --- Phase 3: Direct optimization on h with smoothing ---
    bounds_h = [(0, None)] * N

    for _ in range(15):
        if time.time() > deadline - 15:
            break
        scale = 0.05 + 0.1 * RNG.rand()
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)

        for beta in [25.0, 50.0, 100.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    pert,
                    method='L-BFGS-B',
                    bounds=bounds_h,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                track(res.x)
            except Exception:
                pass

    # --- Phase 4: Nelder-Mead polish on exact objective ---
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        scale = 0.03 * (1.0 + 0.5 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            track(res.x)
        except Exception:
            pass

    # --- Phase 5: Final DCT polish with more modes ---
    n_modes_final = 18
    basis_final = _dct_basis(N, n_modes_final)
    init_coeffs_final = basis_final.T @ best_h

    def dct_obj_final(coeffs):
        h = basis_final @ coeffs
        h = np.maximum(0.0, h)
        return -compute_value(h)

    try:
        res_final = opt.minimize(
            dct_obj_final,
            init_coeffs_final,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        h_final = basis_final @ res_final.x
        h_final = np.maximum(0.0, h_final)
        track(h_final)
    except Exception:
        pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
