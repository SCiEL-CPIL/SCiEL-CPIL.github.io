"""Differentiable Gumbel-Softmax optimization for signed step function with FFT-based backprop."""
import numpy as np
import time

N = 400
DT = 0.5 / N


def _autoconv(s):
    """Compute autoconvolution R[0..N-1] of s via FFT. R[k] = sum_j s[j]*s[j+k] * DT."""
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(s, n)
    P = F * np.conj(F)
    R = np.fft.irfft(P, n)
    return R[:N] * DT


def _score(s):
    """Peak absolute sidelobe of autoconvolution."""
    R = _autoconv(s)
    return np.max(np.abs(R[1:]))


def _loss_and_grad(logits, tau=0.05):
    """
    Compute soft-max loss L = tau*log(sum(exp((R_k^2 - m)/tau))) + m  for k=1..N-1,
    and its gradient w.r.t. logits via Gumbel-Sigmoid straight-through.
    Uses FFT-based backprop: dL/ds_i = sum_k dL/dR_k * (s[i-k] + s[i+k]).
    """
    # Sample Gumbel noise
    u = np.random.rand(N)
    u = np.clip(u, 1e-8, 1 - 1e-8)
    g = -np.log(-np.log(u))

    # Soft sigmoid
    z = logits + g
    y = 1.0 / (1.0 + np.exp(-z))
    s = 2.0 * y - 1.0  # in (-1, 1)

    # Compute autoconv
    R = _autoconv(s)
    R_sq = R[1:] ** 2  # sidelobes squared, length N-1

    # Soft-max loss
    m = np.max(R_sq)
    if m < 1e-12:
        return 0.0, np.zeros(N)
    exp_terms = np.exp((R_sq - m) / tau)
    sum_exp = np.sum(exp_terms)
    L = m + tau * np.log(sum_exp)

    # dL/dR_sq_k = exp((R_sq_k - m)/tau) / sum_exp
    dL_dRsq = exp_terms / sum_exp
    # dL/dR_k = 2 * R_k * dL_dRsq_k
    dL_dR = 2.0 * R[1:] * dL_dRsq  # length N-1, indexed by k=1..N-1

    # Backprop through autoconv: dL/ds_i = sum_{k=1}^{N-1} dL_dR_k * (s[i-k] + s[i+k])
    # This is a linear operator. Use FFT-based approach:
    # Let w[k] = dL_dR_{k-1} for k=1..N-1, w[0]=0. Then dL/ds = conv(s, w) + conv(s, w_rev)
    # where w_rev is w reversed. More precisely:
    # dL/ds_i = sum_k w[k] * s[i-k] + sum_k w[k] * s[i+k]
    # = (s * w_rev)[i] + (s * w)[i]  (with appropriate padding)

    # Build weight vector of length N: w[0]=0, w[k]=dL_dR[k-1] for k=1..N-1
    w = np.zeros(N)
    w[1:] = dL_dR

    # Compute s * w (linear convolution) via FFT
    n_conv = 1
    while n_conv < 2 * N:
        n_conv <<= 1
    S = np.fft.rfft(s, n_conv)
    W = np.fft.rfft(w, n_conv)
    conv_sw = np.fft.irfft(S * W, n_conv)
    conv_sw = conv_sw[:N]  # first N elements: (s*w)[i] = sum_j s[j] w[i-j]

    # For sum_k w[k] s[i-k]: this is (s * w_rev)[i] where w_rev[k] = w[N-1-k]
    # Actually, sum_k w[k] s[i-k] = (s * w_flipped)[i] where w_flipped[k] = w[-k]
    # Let's compute sum_k w[k] s[i-k] directly:
    # = sum_j w[i-j] s[j] = (w * s)[i] = (s * w)[i]  (convolution is commutative)
    # Wait, no. sum_k w[k] s[i-k] is the cross-correlation, not convolution.
    # Let's use: sum_k w[k] s[i-k] = sum_j w[i-j] s[j]
    # This is (w * s)[i] where * denotes convolution: (w*s)[i] = sum_j w[j] s[i-j]
    # So sum_k w[k] s[i-k] = (w * s)[i]
    # And sum_k w[k] s[i+k] = sum_j w[j] s[i+j] = sum_j w[i+j] s[j]  (let j = i+k)
    # = sum_j w[j-i] s[j]  ... this is getting confusing.

    # Simpler: just use direct O(N^2) since N=400, it's 160k ops, very fast.
    dL_ds = np.zeros(N)
    for k in range(1, N):
        wk = dL_dR[k - 1]
        if wk == 0:
            continue
        # dL/ds_i gets wk * s[i-k] for i >= k, and wk * s[i+k] for i < N-k
        if k < N:
            dL_ds[k:] += wk * s[:N - k]  # s[i-k] for i=k..N-1
            dL_ds[:N - k] += wk * s[k:]  # s[i+k] for i=0..N-1-k

    # Backprop through Gumbel-Sigmoid
    dy_dz = y * (1.0 - y)
    dL_dlogits = dL_ds * 2.0 * dy_dz

    return L, dL_dlogits


def _local_search(signs, max_time=30.0):
    """Greedy local search: flip bits that reduce peak sidelobe."""
    start = time.time()
    s = signs.copy()
    R = _autoconv(s)
    cur = np.max(np.abs(R[1:]))

    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        # For each i, compute delta R when flipping s[i]
        # Delta R[k] = -2 * s[i] * (s[i-k] + s[i+k]) * DT
        # We want to find i that minimizes max|new_R|
        # Efficient: precompute neighbor sums
        # For each i, the change in R is:
        # dR[k] = -2 * s[i] * (s[i-k] if i-k>=0 else 0 + s[i+k] if i+k<N else 0) * DT
        # New R = R + dR
        # We need max|new R[1:]| < cur

        # Vectorized approach: for each i, compute the delta vector and check max
        # But that's O(N^2) per pass. N=400, so 160k per pass. Fine for a few passes.

        best_gain = 0
        best_i = -1

        for i in range(N):
            si = s[i]
            # Compute delta R for flipping i
            # dR[k] = -2 * si * (s[i-k] + s[i+k]) * DT for k=1..N-1
            # Only k where i-k >= 0 or i+k < N contribute

            # Build delta vector
            delta = np.zeros(N)
            # For k=1..N-1:
            #   term = 0
            #   if i-k >= 0: term += s[i-k]
            #   if i+k < N: term += s[i+k]
            #   delta[k] = -2 * si * term * DT

            # Vectorize:
            ks = np.arange(1, N)
            idx1 = i - ks  # i-k
            idx2 = i + ks  # i+k
            valid1 = idx1 >= 0
            valid2 = idx2 < N
            terms = np.zeros(N - 1)
            terms[valid1] += s[idx1[valid1]]
            terms[valid2] += s[idx2[valid2]]
            delta[1:] = -2.0 * si * terms * DT

            new_R = R + delta
            new_peak = np.max(np.abs(new_R[1:]))
            gain = cur - new_peak
            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_i >= 0:
            s[best_i] *= -1
            R += 0  # recompute
            R = _autoconv(s)
            cur = np.max(np.abs(R[1:]))
            improved = True

    return s


def construct():
    start_time = time.time()
    timeout = 115.0
    np.random.seed(42)

    # Phase 1: Gumbel-Softmax gradient descent
    logits = np.zeros(N)
    best_signs = np.ones(N)
    best_score = _score(best_signs)

    lr = 0.5
    tau = 0.05

    for step in range(300):
        if time.time() - start_time > timeout - 35:
            break

        L, grad = _loss_and_grad(logits, tau)
        logits -= lr * grad

        # Sample hard signs from Gumbel-Sigmoid
        u = np.random.rand(N)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        g = -np.log(-np.log(u))
        y = 1.0 / (1.0 + np.exp(-(logits + g)))
        signs = np.sign(2.0 * y - 1.0)
        signs[signs == 0] = 1.0

        sc = _score(signs)
        if sc < best_score:
            best_score = sc
            best_signs = signs.copy()

        # Anneal learning rate
        lr *= 0.995

    # Phase 2: Local search
    remaining = timeout - (time.time() - start_time) - 5
    if remaining > 1:
        result = _local_search(best_signs, max_time=remaining)
        sc = _score(result)
        if sc < best_score:
            best_score = sc
            best_signs = result

    return best_signs
