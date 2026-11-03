import time
import numpy as np

N = 600
X0 = 1.0 / 4.0
W = 2.0 * X0 / N            # interval width = 1/300
T = 2.0 * X0                # total interval length = 1/2
SIGMA = 0.15                # width of the frequency-space bump (heuristic)
DEADLINE = 150.0
T_START = time.time()


def _score(f):
    """Normalized maximum of the autoconvolution (lower is better)."""
    conv = np.convolve(f, f, mode='full') * W   # (f*f) sampled on the grid
    peak = conv.max()
    seed_peak = T                               # constant-1 seed: max(f*f)=T
    return peak / seed_peak


def _project_nonneg(f):
    """Project onto the nonnegative cone."""
    return np.maximum(f, 0.0)


def construct():
    """Return a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4].

    Minimizes the peak of the autoconvolution via frequency-domain projected
    gradient descent: minimize ||Fhat||^2_2 (== peak of f*f up to a constant)
    over a bounded frequency band, projecting the inverse transform onto the
    nonnegative cone at each step.
    """
    # Grid for the signal (time domain) and its Fourier transform (frequency domain)
    x = np.linspace(-X0, X0, N, endpoint=False) + 0.5 * W   # cell centers
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=W)               # angular frequencies

    # Bounded frequency band (keeps f smooth => less negative => better projection)
    KMAX = 2.0 * np.pi * 0.5 / W
    mask = np.abs(k) <= KMAX

    # Initialize Fhat as a smooth Gaussian bump concentrated near k=0
    Fhat = np.exp(-0.5 * (k / SIGMA) ** 2)
    Fhat[~mask] = 0.0

    best_f = None
    best_score = np.inf

    # Track the best solution found so far (deadline guard)
    def _update(f):
        nonlocal best_f, best_score
        s = _score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()

    # Project the initial Fhat to get a valid nonnegative starting point
    f = _project_nonneg(np.fft.ifft(Fhat).real)
    _update(f)

    # Projected gradient descent: minimize ||Fhat||^2_2  s.t.  f = F^-1(Fhat) >= 0
    # Gradient of (1/2)||Fhat||^2_2 w.r.t. Fhat is Fhat; step size 1/N gives a
    # damped update on the frequency amplitudes.
    it = 0
    while True:
        if time.time() - T_START > DEADLINE - 8.0:   # safety margin before kill
            break
        # Gradient step in the frequency domain (shrink the amplitudes)
        Fhat = Fhat - Fhat / N
        # Recover the time-domain signal and project onto the nonnegative cone
        f = _project_nonneg(np.fft.ifft(Fhat).real)
        _update(f)
        it += 1

    # Guard: if nothing valid was ever found, fall back to the constant seed
    if best_f is None:
        return [1.0] * N
    return [float(v) for v in best_f]
