"""Rewritten program: spectral init + Nelder-Mead with soft-max surrogate and multi-beta."""
import numpy as np
import time

N = 400
_DEADLINE = 150.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - 15.0 - (time.time() - _T0)

def _objective(h, beta):
    # soft-max surrogate for the signed max of autoconvolution
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    # soft max via log-sum-exp (signed)
    m = np.max(conv)
    if m <= 0:
        smax = m
    else:
        smax = m + np.log(np.sum(np.exp((conv - m) * beta))) / beta
    val = abs(2.0 * N * smax / (s * s))
    return val

def _exact_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _nm_optimize(h0, beta, max_iter=400):
    # Nelder-Mead on the soft-max surrogate
    n = len(h0)
    simplex = [h0.copy()]
    for i in range(n):
        v = h0.copy()
        v[i] += 0.01 * (abs(h0[i]) + 1.0)
        simplex.append(v)
    fvals = [_objective(v, beta) for v in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = _objective(xr, beta)
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = _objective(xe, beta)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = _objective(xc, beta)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = _objective(simplex[i], beta)
                sigma = max(sigma / 2.0, 1e-6)
    best = min(range(n + 1), key=lambda i: fvals[i])
    return simplex[best]

def _make_gaussian_spectral(center, width):
    # h whose FFT magnitude is a Gaussian bump -> smooth, low-peak structure
    k = np.fft.fftfreq(N) * N
    spec = np.exp(-((k - center) ** 2) / (2.0 * width ** 2))
    phase = np.zeros(N)
    h = np.fft.ifft(spec * np.exp(1j * phase)).real
    return h

def _make_chirp(f0, f1, phase0=0.0):
    t = np.arange(N)
    f = f0 + (f1 - f0) * t / N
    h = np.cos(2.0 * np.pi * np.cumsum(f) / N + phase0)
    return h

def _make_random_structured(rng, kind):
    if kind == 0:
        return _make_gaussian_spectral(rng.uniform(-N/4, N/4), rng.uniform(N/20, N/5))
    elif kind == 1:
        return _make_chirp(rng.uniform(0.5, 5.0), rng.uniform(5.0, 30.0), rng.uniform(0, 2*np.pi))
    elif kind == 2:
        # sum of two gaussians
        c1 = rng.uniform(-N/4, N/4); c2 = rng.uniform(-N/4, N/4)
        w1 = rng.uniform(N/20, N/5); w2 = rng.uniform(N/20, N/5)
        return _make_gaussian_spectral(c1, w1) + _make_gaussian_spectral(c2, w2)
    else:
        # random smooth via low-pass noise
        k = np.fft.fftfreq(N) * N
        spec = np.random.randn(N) * np.exp(-k**2 / (2.0 * (N/8)**2))
        return np.fft.ifft(spec * np.exp(1j * np.random.randn(N))).real

def construct():
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _exact_value(best_h)  # 2.0
    best_beta = 1.0

    # candidate structured starts
    starts = []
    for i in range(6):
        starts.append(_make_random_structured(rng, i % 4))
    # also a couple of pure chirps and gaussians
    starts.append(_make_chirp(1.0, 10.0, 0.5))
    starts.append(_make_gaussian_spectral(0.0, N/8))

    betas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    for h0 in starts:
        if _remaining() < 8.0:
            break
        for beta in betas:
            if _remaining() < 8.0:
                break
            h_opt = _nm_optimize(h0.copy(), beta, max_iter=300)
            v = _exact_value(h_opt)
            if v < best_val:
                best_val = v
                best_h = h_opt.copy()
                best_beta = beta
            # try a second polish pass with higher beta from the best found so far
            if best_beta is not None and _remaining() > 10.0:
                h_pol = _nm_optimize(best_h.copy(), min(beta * 2, 50.0), max_iter=150)
                vp = _exact_value(h_pol)
                if vp < best_val:
                    best_val = vp
                    best_h = h_pol.copy()

    # final high-precision polish: local coordinate descent on exact objective
    if _remaining() > 5.0:
        h = best_h.copy()
        for _ in range(50):
            if _remaining() < 2.0:
                break
            improved = False
            for i in range(0, N, 4):
                if _remaining() < 2.0:
                    break
                base = h.copy()
                for step in [0.001, -0.001, 0.005, -0.005]:
                    h[i] = base[i] + step
                    v = _exact_value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                        break
                if not improved:
                    h[i] = base[i]
    return best_h
