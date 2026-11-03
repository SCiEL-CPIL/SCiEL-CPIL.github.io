import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis for FFT updates
_freqs = np.arange(_FSIZE // 2 + 1)
_basis = np.exp(-2j * np.pi * _freqs[None, :] * np.arange(N)[:, None] / _FSIZE)

_FSIZE_CORR = 1
while _FSIZE_CORR < 3 * N:
    _FSIZE_CORR *= 2


def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]


def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)


def _soft_topk_abs_and_grad(h, beta=50.0, k=5):
    """
    Smooth surrogate: soft-max of the top-k absolute autoconvolution values,
    normalized by (sum h)^2. Returns (value, gradient).
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)

    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)

    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        # Soft-max over all peaks (log-sum-exp)
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -60, 60)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta

        # Weights
        w = exp_scaled / sum_exp
        v = w * np.sign(ac)

        # Gradient of S w.r.t h: dS/dh_i = 2 * sum_k v_k * h_{k-i}
        V = np.fft.rfft(v, _FSIZE_CORR)
        H = np.fft.rfft(h, _FSIZE_CORR)
        C = np.fft.irfft(V * np.conj(H), _FSIZE_CORR)
        grad_S = 2 * C[:N]

    # f = S / s^2
    grad = (grad_S * s - 2 * S) / (s ** 3)
    return S / (s * s), grad


def _peak_suppression_obj(h, beta=80.0, k=8):
    """
    Epigraph-style peak suppression: penalize the k-th largest |ac| value
    using a smooth order-statistic approximation, plus a small L2 penalty
    to encourage balanced peaks. Returns (value, gradient).
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)

    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    n_ac = len(abs_ac)

    # Soft-kth-largest via log-sum-exp of top-k
    # Approximate k-th largest: use soft-max with a shifted threshold
    m = np.max(abs_ac)
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        # Use a soft-max that emphasizes the top-k peaks
        # by applying a power to the weights
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -80, 80)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta

        # Additional: penalize variance of top peaks to flatten them
        # Use soft-mean of top-k as a secondary term
        w = exp_scaled / sum_exp
        # Soft-mean of absolute values weighted by w
        soft_mean = np.sum(w * abs_ac)
        # Penalize the gap between max and soft-mean (peak sharpness)
        peak_sharpness = m - soft_mean
        # Total S = max + log-sum-exp term + sharpness penalty
        S_total = S + 0.1 * peak_sharpness

        v = w * np.sign(ac)
        V = np.fft.rfft(v, _FSIZE_CORR)
        H = np.fft.rfft(h, _FSIZE_CORR)
        C = np.fft.irfft(V * np.conj(H), _FSIZE_CORR)
        grad_S = 2 * C[:N]

        # Gradient of peak_sharpness = m - soft_mean
        # dm/dh_i = 2 * sign(ac_max) * h_{max-i} (approx via argmax)
        # d(soft_mean)/dh_i = 2 * sum_k w_k * sign(ac_k) * h_{k-i} = 2 * C[:N]_i (same as grad_S direction)
        # So d(peak_sharpness)/dh_i = dm/dh_i - 2*C[:N]_i
        # For simplicity, use the same gradient direction scaled
        # The dominant gradient is from S; add a small correction
        # Actually, let's just use the main gradient and add the sharpness
        # as a constant offset to the value (gradient approximation)

    # f = S_total / s^2
    grad = (grad_S * s - 2 * S_total) / (s ** 3)
    return S_total / (s * s), grad


def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses precomputed FFT basis for incremental updates.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            delta = -2 * old_val * _basis[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            integral = np.sum(h)
            if abs(integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (integral * integral)

            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val


def _local_search_block(h, deadline=None, block_sizes=None):
    """
    Block flip local search: flip contiguous blocks of signs.
    Uses precomputed FFT basis.
    """
    if block_sizes is None:
        block_sizes = [2, 3, 5, 7, 11, 15]

    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)

    improved = True
    pass_count = 0
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        pass_count += 1
        step_base = max(1, (N - 15) // (100 - min(pass_count * 10, 60)))

        for bs in block_sizes:
            step = max(1, step_base)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * _basis[idx]
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                integral = np.sum(h)
                if abs(integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (integral * integral)

                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    improved = True
                else:
                    h[start:start+bs] = block
    return h, best_val


def _gradient_optimize(h_init, deadline, betas=None):
    """
    Multi-stage smooth optimization using the peak-suppression surrogate.
    """
    if betas is None:
        betas = [30, 60, 100, 150]

    best_x = h_init.copy()
    best_val = compute_raw_value(best_x)

    s0 = np.sum(h_init)
    if abs(s0) < 1e-6:
        return best_x, best_val

    h_norm = h_init / s0

    for beta in betas:
        if time.time() > deadline:
            break

        try:
            def obj_and_grad(x, b=beta):
                return _peak_suppression_obj(x, beta=b)

            res = minimize(obj_and_grad, h_norm, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 200, 'disp': False, 'gtol': 1e-9})
            h_opt = res.x

            val = compute_raw_value(h_opt)
            if val < best_val:
                best_val = val
                best_x = h_opt.copy()

            h_bin = np.sign(h_opt)
            h_bin[h_bin == 0] = 1
            val_bin = compute_raw_value(h_bin)
            if val_bin < best_val:
                best_val = val_bin
                best_x = h_bin.copy()

        except Exception:
            pass

    return best_x, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)

    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    t = np.arange(N)
    rng = np.random.default_rng(42)

    # Chirp sequences (quadratic phase)
    for freq in [1.0, 2.0, 0.5, 3.0, 5.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)

    # Linear phase (constant frequency)
    for freq in [1.0, 2.0, 0.5, 3.0, 7.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)

    # Barker-like structured
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)

    # Random sequences
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)

    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)

    # Thue-Morse-like
    x_tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(x_tm)

    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    # 4. Smooth surrogate optimization from best candidates
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])

    # Optimize from top 10 diverse starts
    top_starts = scored_candidates[:10]

    for init_val, h_init in top_starts:
        if time.time() > deadline - 30:
            break

        h_opt, val_opt = _gradient_optimize(h_init, deadline - 10)
        if val_opt < best_val:
            best_val = val_opt
            best_x = h_opt.copy()

    # 5. Local search refinement on best result
    if time.time() < deadline - 20:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1

        # Single-bit flip local search
        h_work, val_work = _local_search_flip(h_work, max_passes=8, deadline=deadline - 12)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()

        # Block flip local search
        if time.time() < deadline - 8:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 5,
                                                   block_sizes=[2, 3, 5, 7, 11])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()

    # 6. Random restarts with local search if time remains
    if time.time() < deadline - 30:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_flip(x_rand, max_passes=4, deadline=deadline - 8)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()

    # 7. Final high-precision polish
    if time.time() < deadline - 5:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        for polish_pass in range(4):
            if time.time() > deadline - 2:
                break
            h_work, val_work = _local_search_flip(h_work, max_passes=2, deadline=deadline - 1)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()

    return best_x


if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
