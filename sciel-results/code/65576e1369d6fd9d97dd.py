import numpy as np
import time

N = 400

def construct():
    deadline = time.time() + 110  # safety margin under 120s
    best_x = None
    best_val = np.inf

    # Initialize: random signed sequence, normalized to sum=1
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N)
    # Make it signed (both signs) - random normal is
    x = x / x.sum()  # sum x = 1

    def value(x):
        c = np.convolve(x, x)
        # exclude lag 0 (index N-1 in 'full' convolution of length N with N)
        # full conv length = 2N-1, lag 0 is at index N-1
        sidelobes = np.concatenate([c[:N-1], c[N:]])  # exclude index N-1
        S = x.sum()
        if abs(S) < 1e-12:
            return np.inf
        return np.max(np.abs(sidelobes)) / (S*S)

    def smooth_max_sidelobe(x, beta):
        c = np.convolve(x, x)
        sidelobes = np.concatenate([c[:N-1], c[N:]])
        a = beta * np.abs(sidelobes)
        a = a - a.max()  # for stability
        lse = a.max() + np.log(np.sum(np.exp(a)))
        return lse / beta

    def grad_smooth_max_sidelobe(x, beta):
        c = np.convolve(x, x)
        sidelobes = np.concatenate([c[:N-1], c[N:]])
        a = beta * np.abs(sidelobes)
        a = a - a.max()
        w = np.exp(a) / np.sum(np.exp(a))  # weights on sidelobes
        # gradient of lse/beta w.r.t. sidelobes = w * sign(sidelobes)
        # (since d/d|s| lse = beta * w, and d|s|/ds = sign(s), so d lse/beta / ds = w * sign(s))
        ds = w * np.sign(sidelobes)
        # Now gradient w.r.t. x: ds_k * dc_k/dx, where dc_k/dx_j = x_{k-j}+x_{j-k} for the sidelobe indexing
        # Build full weight vector over all lags (length 2N-1), with ds placed at sidelobe positions
        full_w = np.zeros(2*N-1)
        full_w[:N-1] = ds[:N-1]
        full_w[N:] = ds[N-1:]
        # gradient = convolve(full_w, x) + convolve(full_w[::-1], x)  (the two terms)
        # Actually: d/dx_j sum_k full_w_k c_k = sum_k full_w_k (x_{k-j} + x_{j-k})
        # = conv(full_w, x)[j] (for the x_{j-k} term, with proper alignment) + conv(full_w, x_flipped)...
        # Let me compute via: term1[j] = sum_k full_w[k] * x[j-k]  = (full_w * x)[j] in 'full'
        # term2[j] = sum_k full_w[k] * x[k-j]  = sum_k full_w[k] * x[-(j-k)] 
        # Let x_flip[m] = x[-m] (i.e., x reversed around 0). Then x[k-j] = x_flip[j-k].
        # So term2[j] = sum_k full_w[k] x_flip[j-k] = (full_w * x_flip)[j].
        # But x_flip for a finite array: we need x indexed from -(N-1) to (N-1)? No, x is indexed 0..N-1.
        # This is getting complicated with the finite support. Let me just compute the gradient directly using the autocorrelation structure.
        ...
