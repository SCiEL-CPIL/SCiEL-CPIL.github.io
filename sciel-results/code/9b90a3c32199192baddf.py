import numpy as np
import time

N = 50


def _autoconv(h):
    return np.convolve(h, h, mode='full')


def _peak_sidelobe(h):
    c = _autoconv(h)
    main = c[0]
    if main <= 0:
        return 0.0
    return float(np.max(c[1:])) / main


def _score(h):
    # Lower peak sidelobe => better. Return a "higher is better" objective.
    ps = _peak_sidelobe(h)
    return 1.0 / (1.0 + ps)


def _psd_to_h(S):
    # S is the periodized power spectrum (length N), non-negative.
    # h is the real sequence with FFT power spectrum S (up to a phase).
    # We take the phase to be zero (h real, non-negative-ish) via:
    # h = ifft(sqrt(S) * exp(i*phase)). With phase 0, h may have negatives.
    # Instead, use the fact that for a real h, the PSD is |FFT(h)|^2.
    # We construct h by taking h = ifft(sqrt(S)) and symmetrize to be real.
    S = np.maximum(S, 0.0)
    amp = np.sqrt(S)
    # enforce Hermitian symmetry so ifft is real
    amp = (amp + np.conj(amp[::-1])) / 2.0
    h = np.fft.ifft(amp).real
    # make non-negative
    h = np.maximum(h, 0.0)
    n = np.linalg.norm(h)
    if n > 0:
        h = h / n
    return h


def construct():
    start = time.time()
    deadline = start + 120.0

    best_h = None
    best_ps = np.inf

    def consider(h):
        nonlocal best_h, best_ps
        h = np.maximum(h, 0.0)
        n = np.linalg.norm(h)
        if n <= 0:
            return
        h = h / n
        ps = _peak_sidelobe(h)
        if ps < best_ps:
            best_ps = ps
            best_h = h.copy()

    # ---- Phase 1: random PSD (waterfilling-inspired) restarts ----
    rng = np.random.default_rng(0)
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        # random PSD with a floor to suppress sidelobes
        base = rng.random(N)
        # emphasize the DC (lag-0) component
        base[0] = base[0] + 2.0
        # add a floor to all non-DC bins
        floor = 0.05 * base[0]
        S = base + floor
        h = _psd_to_h(S)
        consider(h)

    # ---- Phase 2: peak-sidelobe minimization via PSD coordinate descent ----
    def psd_cd(S, iters=60):
        S = np.maximum(S, 0.0).copy()
        for _ in range(iters):
            if time.time() > deadline - 15.0:
                break
            # compute current h and its sidelobe spectrum
            amp = np.sqrt(S)
            amp = (amp + np.conj(amp[::-1])) / 2.0
            h = np.fft.ifft(amp).real
            h = np.maximum(h, 0.0)
            n = np.linalg.norm(h)
            if n <= 0:
                break
            h = h / n
            c = _autoconv(h)
            main = c[0]
            if main <= 0:
                break
            # the bin with the largest sidelobe in frequency domain
            # we want to reduce the max |FFT(h)|^2 over k != 0
            F = np.abs(np.fft.fft(h)) ** 2
            idx = np.argmax(np.where(np.arange(N) != 0, F, -1.0))
            # reduce S[idx] (the offending frequency), keep DC fixed
            if idx != 0:
                S[idx] = max(0.0, S[idx] - 0.1 * (S[idx] + 1e-9))
                # redistribute a little to other non-DC bins to keep energy
                others = [j for j in range(1, N)]
                S[others] += 0.1 * S[idx] / len(others)
        amp = np.sqrt(S)
        amp = (amp + np.conj(amp[::-1])) / 2.0
        h = np.fft.ifft(amp).real
        h = np.maximum(h, 0.0)
        n = np.linalg.norm(h)
        if n > 0:
            h = h / n
        return h

    # polish a few of the best PSD restarts
    if best_h is not None:
        # reconstruct a PSD from best_h
        S0 = np.abs(np.fft.fft(best_h)) ** 2
        h_cd = psd_cd(S0, iters=40)
        consider(h_cd)

    # ---- Phase 3: time-domain coordinate descent on peak sidelobe ----
    def time_cd(h, iters=200):
        h = h.copy()
        n = np.linalg.norm(h)
        h = h / n
        for _ in range(iters):
            if time.time() > deadline - 8.0:
                break
            improved = False
            for i in range(N):
                c = _autoconv(h)
                main = c[0]
                if main <= 0:
                    break
                ps = np.max(c[1:]) / main
                # try a few perturbations of h[i]
                for factor in (0.5, 0.8, 1.25, 1.6, 0.0):
                    h2 = h.copy()
                    h2[i] = h[i] * factor
                    n2 = np.linalg.norm(h2)
                    if n2 <= 0:
                        continue
                    h2 = h2 / n2
                    c2 = _autoconv(h2)
                    main2 = c2[0]
                    if main2 <= 0:
                        continue
                    ps2 = np.max(c2[1:]) / main2
                    if ps2 < ps - 1e-12:
                        h = h2
                        ps = ps2
                        improved = True
                        break
            if not improved:
                break
        return h

    # polish the current best
    if best_h is not None:
        h_pol = time_cd(best_h, iters=120)
        consider(h_pol)

    # also polish the trivial all-equal as a fallback
    consider(np.ones(N) / np.sqrt(N))

    if best_h is None:
        best_h = np.ones(N) / np.sqrt(N)

    return list(best_h)
