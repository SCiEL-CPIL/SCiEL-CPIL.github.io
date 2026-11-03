"""
Signed autocorrelation maximum absolute-value variant.
Structural change: Discrete gradient descent on binary signs using
FFT-accelerated soft-max autoconvolution surrogate gradient.
"""
import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

# Precompute FFT sizes and basis for incremental updates
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# For cross-correlation in gradient computation
_FSIZE_CORR = 1
while _FSIZE_CORR < 3 * N:
    _FSIZE_CORR *= 2


def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]


def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2 * N - 1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)


def _soft_max_abs_and_grad(h, beta=50.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
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
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta

        # Weights for the soft-max
        w = exp_scaled / sum_exp

        # v_k = w_k * sign(ac_k)
        v = w * np.sign(ac)

        # Gradient of S w.r.t h
        # dS/dh_i = 2 * sum_k v_k h_{k-i}
        # This is 2 * cross_correlation(v, h)

        V = np.fft.rfft(v, _FSIZE_CORR)
        H = np.fft.rfft(h, _FSIZE_CORR)
        C = np.fft.irfft(V * np.conj(H), _FSIZE_CORR)

        # C[i] should be sum_n v[n+i] conj(h[n])
        # Since h is real, conj(h[n]) = h[n].
        # So C[i] = sum_n v[n+i] h[n].
        # This is exactly what we want for i=0..N-1.

        grad_S = 2 * C[:N]

    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3

    grad = (grad_S * s - 2 * S) / (s ** 3)

    return S / (s * s), grad


def _discrete_gradient_descent(h, max_iters=1000, deadline=None, beta=50.0):
    """
    Discrete gradient descent on binary signs.
    At each step, compute the soft-max surrogate gradient,
    then flip signs where the gradient suggests improvement.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1

    best_val = compute_raw_value(h)
    H_fft = np.fft.rfft(h, _FSIZE)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Compute soft-max surrogate and its gradient
        S_val, grad = _soft_max_abs_and_grad(h, beta=beta)

        # Discrete update: for each position i, consider flipping h[i].
        # The change in objective from flipping h[i] is approximately
        # grad[i] * (-2 * h[i]) = -2 * h[i] * grad[i]
        # If this is negative, the flip improves the objective.

        # Compute the "improvement" for each flip
        improvements = -2.0 * h * grad

        # Flip signs where improvement is positive
        flip_mask = improvements > 0

        if not np.any(flip_mask):
            break

        # Apply flips
        h[flip_mask] = -h[flip_mask]

        # Update FFT incrementally
        # delta = -2 * old_h[flip_mask] * basis[flip_mask]
        # But since we flipped multiple positions, let's just recompute
        # for simplicity (N=400 is small enough)
        H_fft = np.fft.rfft(h, _FSIZE)

        # Compute actual raw value
        new_val = compute_raw_value(h)

        if new_val < best_val:
            best_val = new_val
        else:
            # Revert if it got worse (shouldn't happen often with good gradient)
            # For now, just continue; the next gradient step will correct
            pass

    return h, best_val


def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT incrementally for speedup.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)

    # Precompute basis for incremental FFT updates
    freqs = np.arange(_FSIZE // 2 + 1)
    basis = np.exp(-2j * np.pi * freqs[None, :] * np.arange(N)[:, None] / _FSIZE)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign: h[i] -> -h[i]
            old_val = h[i]
            h[i] = -old_val
            # Update FFT incrementally
            delta = -2 * old_val * basis[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2 * N - 1]
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
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val


def _generate_initializations(rng):
    """Generate diverse initial sign patterns."""
    inits = []

    # All ones
    inits.append(np.ones(N))

    # Random signs
    inits.append(np.sign(rng.randn(N)))

    # Block patterns
    for block_size in [10, 20, 40, 50, 80, 100]:
        pattern = np.zeros(N)
        for i in range(0, N, 2 * block_size):
            pattern[i:i + block_size] = 1
            if i + 2 * block_size <= N:
                pattern[i + block_size:i + 2 * block_size] = -1
        pattern = pattern[:N]
        inits.append(pattern)

    # Sine wave patterns
    for freq in [0.1, 0.2, 0.3, 0.5]:
        t = np.arange(N)
        inits.append(np.sign(np.sin(2 * np.pi * freq * t)))

    # Alternating blocks
    for block_size in [5, 10, 20]:
        pattern = np.zeros(N)
        for i in range(0, N, block_size):
            val = 1 if (i // block_size) % 2 == 0 else -1
            pattern[i:min(i + block_size, N)] = val
        inits.append(pattern)

    # More random patterns
    for _ in range(5):
        inits.append(np.sign(rng.randn(N)))

    return inits


def construct():
    """
    Construct a sequence of N signed step heights minimizing
    max_k |autoconv_k| / (sum x)^2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_x = None
    best_score = np.inf

    rng = np.random.RandomState(42)

    initializations = _generate_initializations(rng)

    # Phase 1: Discrete gradient descent from each initialization
    for x0 in initializations:
        if time.time() > deadline - 30.0:
            break

        x_opt, score = _discrete_gradient_descent(
            x0, max_iters=500, deadline=deadline - 30.0, beta=50.0
        )

        if score < best_score:
            best_score = score
            best_x = x_opt.copy()

    # Phase 2: Local search refinement on best result
    if best_x is not None and time.time() < deadline - 10.0:
        x_final, score_final = _local_search_flip(
            best_x, max_passes=3, deadline=deadline - 5.0
        )
        if score_final < best_score:
            best_score = score_final
            best_x = x_final

    # Phase 3: Final polish with more local search if time permits
    if best_x is not None and time.time() < deadline - 5.0:
        x_final, score_final = _local_search_flip(
            best_x, max_passes=2, deadline=deadline - 2.0
        )
        if score_final < best_score:
            best_score = score_final
            best_x = x_final

    if best_x is None:
        return np.ones(N)

    # Ensure the result is a proper step function (values are +/-1)
    result = np.sign(best_x)
    # If all zeros, default to ones
    if np.all(result == 0):
        result = np.ones(N)

    return result
