"""Program using model-and-optimize approach for first autocorrelation inequality."""
import time
import numpy as np
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1  # set in construct()

def _fft_convolve(h):
    """FFT-based linear convolution (full mode)."""
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_smooth(h, beta=500.0):
    """
    Smooth surrogate: log-sum-exp based approximation of max(conv) / s^2.
    More differentiable than hard max, helps gradient-based optimizers.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    # log-sum-exp for numerical stability
    m = np.max(conv)
    if m <= 0:
        return 0.0
    lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
    return lse / (s * s)

# EVOLVE-BLOCK-START
def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Set FFT size: next power of 2 >= 2N-1
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    # Coordinates x_i:
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    t = x / 0.25  # Map to [-1, 1]

    # --- Parametric families for initial grid search ---
    def param_h1(sigma, p):
        if sigma <= 0 or p <= 0:
            return np.ones(N) / N
        h = np.exp(-np.abs(x / sigma) ** p)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    def param_h2(a, b):
        if a <= 0 or b <= 0:
            return np.ones(N) / N
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, (1.0 - abs_t ** a) ** b, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    def param_h4(c, a, b):
        if c < 0 or c >= 1 or a <= 0 or b <= 0:
            return np.ones(N) / N
        abs_t = np.abs(t)
        h = np.zeros(N)
        mask_plateau = abs_t < c
        mask_transition = (abs_t >= c) & (abs_t < 1.0)
        h[mask_plateau] = 1.0
        if np.any(mask_transition):
            tc = (abs_t[mask_transition] - c) / (1.0 - c)
            h[mask_transition] = (1.0 - tc ** a) ** b
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # Track best
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    def try_update(h, score):
        nonlocal best_h, best_score
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Quick grid search over parametric families
    grid1 = [(s, p) for s in [0.05, 0.08, 0.1, 0.12, 0.15] for p in [0.5, 1, 2, 3, 4]]
    grid2 = [(a, b) for a in [0.5, 1, 2, 3, 4, 6] for b in [0.5, 1, 2, 3, 4, 6]]
    grid4 = [(c, a, b) for c in [0.0, 0.2, 0.4, 0.6] for a in [1, 2, 4] for b in [1, 2, 4]]

    for s0, p0 in grid1:
        if time.time() > deadline:
            break
        h0 = param_h1(s0, p0)
        sc = compute_score(h0)
        try_update(h0, sc)

    for a0, b0 in grid2:
        if time.time() > deadline:
            break
        h0 = param_h2(a0, b0)
        sc = compute_score(h0)
        try_update(h0, sc)

    for c0, a0, b0 in grid4:
        if time.time() > deadline:
            break
        h0 = param_h4(c0, a0, b0)
        sc = compute_score(h0)
        try_update(h0, sc)

    # --- Neural network parameterization ---
    # Latent vector z (dim D) -> MLP -> 600 outputs -> softplus -> normalize
    D = 24  # latent dimension
    H1 = 64  # hidden layer 1
    H2 = 32  # hidden layer 2

    # Initialize weights with small random values
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 1.0 / np.sqrt(D), (D, H1))
    b1 = rng.normal(0, 0.1, H1)
    W2 = rng.normal(0, 1.0 / np.sqrt(H1), (H1, H2))
    b2 = rng.normal(0, 0.1, H2)
    W3 = rng.normal(0, 1.0 / np.sqrt(H2), (H2, N))
    b3 = rng.normal(0, 0.1, N)

    def mlp_forward(z):
        """Forward pass: z -> h (nonnegative, normalized)."""
        a1 = np.maximum(0, z @ W1 + b1)  # ReLU
        a2 = np.maximum(0, a1 @ W2 + b2)  # ReLU
        out = a2 @ W3 + b3
        # Softplus for nonnegativity, then normalize
        h = np.log1p(np.exp(np.clip(out, -50, 50)))
        h = np.maximum(h, 1e-10)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    def mlp_obj(z, beta=500.0):
        """Objective: smooth surrogate of max(autoconv) / (sum h)^2."""
        h = mlp_forward(z)
        return compute_score_smooth(h, beta)

    # Flattened parameter vector for L-BFGS-B
    def flatten_params(z, W1, b1, W2, b2, W3, b3):
        return np.concatenate([z, W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])

    def unflatten_params(p):
        idx = 0
        z = p[idx:idx+D]; idx += D
        W1 = p[idx:idx+D*H1].reshape(D, H1); idx += D*H1
        b1 = p[idx:idx+H1]; idx += H1
        W2 = p[idx:idx+H1*H2].reshape(H1, H2); idx += H1*H2
        b2 = p[idx:idx+H2]; idx += H2
        W3 = p[idx:idx+H2*N].reshape(H2, N); idx += H2*N
        b3 = p[idx:idx+N]; idx += N
        return z, W1, b1, W2, b2, W3, b3

    def mlp_obj_full(p, beta=500.0):
        z, W1, b1, W2, b2, W3, b3 = unflatten_params(p)
        # Temporarily set global weights
        nonlocal _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3
        _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3 = W1, b1, W2, b2, W3, b3
        h = mlp_forward(z)
        return compute_score_smooth(h, beta)

    # Global storage for MLP weights during optimization
    _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3 = W1, b1, W2, b2, W3, b3

    def mlp_forward_global(z):
        a1 = np.maximum(0, z @ _mlp_W1 + _mlp_b1)
        a2 = np.maximum(0, a1 @ _mlp_W2 + _mlp_b2)
        out = a2 @ _mlp_W3 + _mlp_b3
        h = np.log1p(np.exp(np.clip(out, -50, 50)))
        h = np.maximum(h, 1e-10)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    def mlp_obj_global(p, beta=500.0):
        z, W1, b1, W2, b2, W3, b3 = unflatten_params(p)
        nonlocal _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3
        _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3 = W1, b1, W2, b2, W3, b3
        h = mlp_forward_global(z)
        return compute_score_smooth(h, beta)

    # Multi-start optimization in latent+weight space
    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        # Random initialization
        rng_i = np.random.default_rng(42 + i)
        z0 = rng_i.normal(0, 0.5, D)
        W1_i = rng_i.normal(0, 1.0 / np.sqrt(D), (D, H1))
        b1_i = rng_i.normal(0, 0.1, H1)
        W2_i = rng_i.normal(0, 1.0 / np.sqrt(H1), (H1, H2))
        b2_i = rng_i.normal(0, 0.1, H2)
        W3_i = rng_i.normal(0, 1.0 / np.sqrt(H2), (H2, N))
        b3_i = rng_i.normal(0, 0.1, N)

        p0 = flatten_params(z0, W1_i, b1_i, W2_i, b2_i, W3_i, b3_i)

        # Set global weights before optimization
        _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3 = W1_i, b1_i, W2_i, b2_i, W3_i, b3_i

        try:
            res = optimize.minimize(
                mlp_obj_global, p0, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                z_r, W1_r, b1_r, W2_r, b2_r, W3_r, b3_r = unflatten_params(res.x)
                _mlp_W1, _mlp_b1, _mlp_W2, _mlp_b2, _mlp_W3, _mlp_b3 = W1_r, b1_r, W2_r, b2_r, W3_r, b3_r
                h_cand = mlp_forward_global(z_r)
                sc = compute_score(h_cand)
                try_update(h_cand, sc)
        except Exception:
            pass

    # Final check: ensure best_h is valid (non-negative)
    best_h = np.maximum(0, best_h)

    return best_h.tolist()
# EVOLVE-BLOCK-END
