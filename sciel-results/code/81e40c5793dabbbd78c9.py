import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize
import time

N = 400

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _exact_value(h):
    s = np.sum(h)
    if s == 0.0:
        return np.inf
    return abs(2.0 * N * np.max(_autoconv(h)) / (s * s))

def construct():
    deadline = time.time() + 130.0
    best_h = np.ones(N)
    best_val = _exact_value(best_h)  # 2.0

    # DCT-II basis: c = dct(h); h = idct(c)/sqrt(N)
    # Set c[0] (DC) to fix sum(h) = S, and optimize the other amplitudes.
    # sum(h) = sqrt(N) * c[0]  (for unnormalized dct/idct with scipy convention)
    # scipy: dct(h, ortho=True) -> forward; idct(x, ortho=True) -> inverse
    # With ortho=True: h = idct(c, ortho=True), c = dct(h, ortho=True)
    # sum(h) = sqrt(N)*c[0]. We want sum(h) = S. So c[0] = S/sqrt(N).
    S = float(N)  # target sum; we can also optimize S but keep fixed for now.

    c0 = S / np.sqrt(N)

    def _h_from_amplitudes(a):
        c = np.zeros(N)
        c[0] = c0
        c[1:] = a
        return idct(c, type=2, norm=None) / np.sqrt(N) * np.sqrt(N)  # ortho
        # Actually with norm=None, idct is unnormalized. Let's use ortho=True.

    # Use ortho=True for clean orthonormal transform.
    def _h_from_amplitudes(a):
        c = np.zeros(N)
        c[0] = c0
        c[1:] = a
        return idct(c, type=2, norm='ortho')

    def _amplitudes_from_h(h):
        c = dct(h, type=2, norm='ortho')
        return c[1:].copy()

    def _soft_max(x, t):
        m = np.max(x)
        return m + t * np.log(np.sum(np.exp((x - m) / t)))

    def _objective(a):
        h = _h_from_amplitudes(a)
        c = _autoconv(h)
        t = 0.5
        sm = _soft_max(c, t)
        # We want to minimize |2N * max(c) / S^2|. Since S fixed, minimize |max(c)|.
        # But max(c) >= c[0] > 0. We want max(c) to be as small as possible in magnitude,
        # i.e., we want max(c) to be close to 0 (negative side). So minimize max(c)
        # (signed), but constrained so that max(c) doesn't exceed c[0]... 
        # Actually we just minimize the signed soft-max of c. If we can make all c <= 0,
        # then max(c) < 0 and |max(c)| = -max(c). But c[0] = sum(h^2) > 0 always!
        # So max(c) >= c[0] > 0. Thus we cannot make max(c) negative.
        # 
        # Wait, this means the minimum is indeed 2.0? But inspirations beat it...
        # Let me re-examine. c[0] = sum h_i^2. This is always positive.
        # max(c) >= c[0]. So max(c) > 0. So |max(c)| = max(c).
        # V = 2N * max(c) / S^2. By Cauchy-Schwarz, S^2 <= N * sum(h^2) = N*c[0].
        # So V >= 2N * c[0] / S^2 >= 2N * c[0] / (N*c[0]) = 2.
        # So V >= 2 always. The seed achieves 2.
        # 
        # This means NO improvement is possible! The inspirations must be wrong or
        # I'm misunderstanding the problem. Let me re-read.
        #
        # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
        # np.convolve(h, h) is full convolution, length 2N-1.
        # For h = ones(400), convolve is [1,2,...,400,...,2,1], max = 400.
        # value = |2*400*400/400^2| = 2. Correct.
        #
        # So the lower bound is 2. The inspirations claiming 1.24 score must have
        # a different interpretation. Score = seed_value / value. If value > 2, score < 1.
        # If value < 2, score > 1. But we proved value >= 2.
        #
        # Unless... np.sum(h) ** 2 can be negative? No, it's squared.
        # Unless np.max picks a negative value and abs makes it positive but...
        # max is the maximum, so it's the largest (most positive) entry.
        #
        # I think the inspirations are actually WORSE (score < 1) and the "1.24" is
        # a misreading, OR the problem is different. Let me just try to find something
        # with value < 2 empirically, in case my proof is wrong.
        
        return sm

    # Let me just try random search and gradient to see if we can beat 2.
    rng = np.random.RandomState(42)
    
    # Try random vectors
    for trial in range(200):
        if time.time() > deadline - 20:
            break
        h = rng.randn(N)
        s = np.sum(h)
        if abs(s) < 1e-6:
            continue
        val = _exact_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Try structured: h = ones + small perturbation
    for amp in [0.01, 0.05, 0.1, 0.2, 0.5]:
        if time.time() > deadline - 15:
            break
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            h = np.ones(N) + amp * rng.randn(N)
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            val = _exact_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()

    # Try alternating signs
    for pattern in ['alt', 'block']:
        if time.time() > deadline - 15:
            break
        for amp in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if pattern == 'alt':
                h = np.ones(N) * (1.0 + amp * np.sign(np.arange(N) % 2 - 0.5))
            else:
                h = np.ones(N)
                h[N//2:] = -amp
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            val = _exact_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()

    # L-BFGS-B on amplitudes
    if time.time() < deadline - 30:
        a0 = np.zeros(N - 1)
        a0[0] = 0.1  # small perturbation

        def _obj(a):
            h = _h_from_amplitudes(a)
            c = _autoconv(h)
            return _soft_max(c, 0.3)

        def _grad(a):
            h = _h_from_amplitudes(a)
            c = _autoconv(h)
            t = 0.3
            m = np.max(c)
            w = np.exp((c - m) / t)
            w = w / np.sum(w)
            # dc/da_j = 2 * (h * h)[j] where h depends on a
            # h = idct(c)/... , dh/da_j = e_j / sqrt(N) (basis vector j)
            # dc_k/da_j = sum_i dh_i/da_j * h_{i+k-N+...} ... complex
            # Just use finite difference approximation
            eps = 1e-6
            g = np.zeros(N - 1)
            for j in range(N - 1):
                a_plus = a.copy()
                a_plus[j] += eps
                h_plus = _h_from_amplitudes(a_plus)
                c_plus = _autoconv(h_plus)
                g[j] = (_soft_max(c_plus, t) - _soft_max(c, t)) / eps
            return g

        # Too slow for full gradient. Use analytical gradient.
        # c_k = sum_i h_i h_{i+k} for appropriate indices.
        # For full conv, c_k = sum_i h_i * h_{i+k-N+1} (with zero padding)
        # Let's use the fact that h = idct(C, ortho), so h_i = sum_j C_j * phi_j(i)
        # where phi_j are DCT basis. dc_k/da_j = 2 * sum_i phi_j(i) * h_{i+k-N+1}
        
        def _grad_analytical(a):
            h = _h_from_amplitudes(a)
            c = _autoconv(h)
            t = 0.3
            m = np.max(c)
            w = np.exp((c - m) / t)
            w = w / np.sum(w)
            # DCT-II basis: phi_j(i) = sqrt(2/N) * cos(pi*(i+0.5)*j/N) for j>=1
            # (with ortho normalization)
            i_idx = np.arange(N)
            g = np.zeros(N - 1)
            for k in range(2 * N - 1):
                if w[k] < 1e-10:
                    continue
                # c_k = sum_{i=0}^{N-1} h_i * h_{i+k} where h extended by 0
                # dc_k/da_j = 2 * sum_i (dh_i/da_j) * h_{i+k}
                # dh_i/da_j = phi_j(i)
                # For k >= 0: valid i where i < N and i+k < N, i.e., i < N-k
                # For k < 0: valid i where i < N and i+k >= 0, i.e., i >= -k
                if k >= 0:
                    i_valid = np.arange(0, N - k)
                else:
                    i_valid = np.arange(-k, N)
                j_idx = np.arange(1, N)
                phi = np.sqrt(2.0 / N) * np.cos(np.pi * (i_valid[:, None] + 0.5) * j_idx[None, :] / N)
                h_part = h[i_valid + k] if k >= 0 else h[i_valid + k]
                # h_{i+k} for i in i_valid
                if k >= 0:
                    h_val = h[i_valid + k]
                else:
                    h_val = h[i_valid + k]
                contrib = 2.0 * w[k] * phi.T @ h_val
                g += contrib
            return g

        # This is still O(N^2) per gradient. Let's just use L-BFGS-B with finite diff
        # but limit iterations.
        try:
            res = minimize(_obj, a0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 500})
            a_opt = res.x
            h_opt = _h_from_amplitudes(a_opt)
            val_opt = _exact_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
