import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # safety margin under 150s
    h = (0.25) / N  # interval width: total length 0.5 / 600
    h = (1.0 / 2.0) / N

    def autoconv(f):
        # circular-friendly linear autoconvolution via FFT, sampled on the 2N-1 grid
        # We embed f in a length >= 2N-1 FFT to get linear convolution.
        M = 1
        while M < (2 * N - 1):
            M *= 2
        F = np.fft.rfft(f, M)
        conv = np.fft.irfft(F * F.conj(), M)
        # linear autoconvolution lives in conv[0 : 2N-1]
        return conv[:2 * N - 1]

    def l1(f):
        return float(np.sum(f) * h)

    def l2sq(f):
        return float(np.sum(f * f) * h)

    def score_proxy(f):
        # normalized max of autoconvolution; scale-invariant
        conv = autoconv(f)
        peak = float(np.max(conv))
        denom = l1(f) ** 2
        if denom <= 0:
            return float('inf')
        return peak / denom

    # Start from seed: uniform ones
    f = np.ones(N, dtype=float)
    # normalize to l1 = 1 for stability
    f = f / l1(f)
    best_f = f.copy()
    best_val = score_proxy(best_f)

    # Projected gradient descent on g(f) = max(conv) with l1(f)=1, f>=0
    # Use a smoothed max via log-sum-exp for gradient.
    temp = 0.05
    lr = 0.02
    max_iter = 20000

    for it in range(max_iter):
        if time.time() > deadline:
            break
        conv = autoconv(f)
        # log-sum-exp smooth max
        m = np.max(conv)
        w = np.exp((conv - m) / temp)
        w = w / np.sum(w)
        # gradient of smooth max wrt conv
        # conv is linear in f via convolution; gradient wrt f_i = sum_j w_j * (delta of conv_j wrt f_i)
        # conv_j = sum_i f_i * f_{j-i} (discrete, scaled by h). d conv_j / d f_i = f_{j-i} + f_{j-N+1-i}... 
        # Simpler: use finite difference-free analytic gradient via correlation.
        # grad_i = 2 * (w * f)[i] convolved appropriately. Use: grad = 2*h*(w * f) where * is the
        # appropriate cross-correlation. We compute via FFT: grad_i = 2*h*sum_j w_j f_{j-i}.
        # Build kernel w and convolve with f.
        M = 1
        while M < (2 * N - 1):
            M *= 2
        W = np.fft.rfft(w, M)
        F = np.fft.rfft(f, M)
        grad = np.fft.irfft(W * F.conj(), M)[:N]
        grad = 2.0 * h * grad
        # scale-invariant: also subtract l1 constraint term; project handles l1
        f_new = f - lr * grad
        # project: f>=0 and l1 = 1
        f_new = np.maximum(f_new, 0.0)
        s = np.sum(f_new) * h
        if s > 0:
            f_new = f_new / s
        # line search
        val = score_proxy(f_new)
        if val < best_val:
            best_f = f_new.copy()
            best_val = val
        f = f_new
        # anneal temperature
        temp = max(0.005, temp * 0.999)
        lr = max(0.001, lr * 0.9995)

    # Return best (scale does not matter for a scale-invariant score, but keep nonnegative)
    out = np.maximum(best_f, 0.0)
    s = np.sum(out)
    if s > 0:
        out = out / s * 1.0  # keep l1=1; scale-invariant anyway
    return [float(x) for x in out]
# EVOLVE-BLOCK-END
