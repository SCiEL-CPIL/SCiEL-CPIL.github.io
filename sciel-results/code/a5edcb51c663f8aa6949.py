"""
Strategy:
1. Start with Zadoff-Chu inspired real-valued sequences (near-constant amplitude,
   low autocorrelation sidelobes).
2. Parameterize the 400-length sequence in a low-dimensional spectral basis
   (DCT/DFT) so gradient methods navigate a smooth manifold.
3. Use smooth-max (log-sum-exp) surrogate for the hard max in the objective.
4. Refine with L-BFGS-B on the reduced parameters.
5. Track elapsed time with explicit deadline guard.
"""
import numpy as np
import scipy.optimize as so
import time

N = 400
TIME_LIMIT = 140  # safety margin before 150s kill

def _value(h):
    """Exact task objective."""
    s = np.sum(h)
    if np.abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _fft_convolve(h):
    """Fast full convolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)
    return conv

def _smooth_max_value(h, beta=50.0):
    """Smooth surrogate for the objective using log-sum-exp max."""
    s = np.sum(h)
    if np.abs(s) < 1e-12:
        return 1e12
    conv = _fft_convolve(h)
    # Stable log-sum-exp for max
    c = np.max(conv)
    lse = c + np.log(np.sum(np.exp(beta * (conv - c)))) / beta
    return abs(2.0 * N * lse / (s * s))

def _zadoff_chu_h(m, n=N):
    """
    Real-valued Zadoff-Chu inspired sequence.
    ZC: x[k] = exp(-j*pi*m*k*(k+1)/n)
    We take real part and add a small DC to ensure nonzero sum.
    """
    k = np.arange(n)
    phase = -np.pi * m * k * (k + 1) / n
    h = np.cos(phase)
    # Ensure sum is not zero
    s = np.sum(h)
    if np.abs(s) < 1.0:
        h = h + 0.5
    return h

def _chirp_h(f1, f2, n=N):
    """Linear chirp: real part of exp(j*pi*(f1*t + f2*t^2))."""
    t = np.linspace(0, 1, n, endpoint=False)
    phase = np.pi * (f1 * t + f2 * t * t)
    h = np.cos(phase)
    s = np.sum(h)
    if np.abs(s) < 1.0:
        h = h + 0.5
    return h

def _dct_basis_matrix(n_basis, n=N):
    """Orthonormal DCT-II basis vectors (first n_basis)."""
    k = np.arange(n_basis)[:, None]
    x = np.arange(n)[None, :]
    # DCT-II: phi_k(x) = sqrt(2/n) * cos(pi*(x+0.5)*k/n) for k>0, sqrt(1/n) for k=0
    basis = np.cos(np.pi * (x + 0.5) * k / n)
    # Normalize
    norms = np.sqrt(np.sum(basis ** 2, axis=1))
    basis = basis / norms[:, None]
    return basis  # shape (n_basis, n)

def _dft_basis_matrix(n_basis, n=N):
    """First n_basis real DFT (cosine) basis vectors."""
    k = np.arange(n_basis)[:, None]
    x = np.arange(n)[None, :]
    basis = np.cos(2.0 * np.pi * k * x / n)
    norms = np.sqrt(np.sum(basis ** 2, axis=1))
    basis = basis / norms[:, None]
    return basis

def _construct():
    start = time.time()
    deadline = start + TIME_LIMIT

    def time_ok():
        return time.time() < deadline - 5

    rng = np.random.default_rng(12345)

    # ---- Phase 1: Generate diverse initial candidates ----
    candidates = []

    # Seed fallback
    candidates.append(np.ones(N))

    # Zadoff-Chu for several m values
    for m in [1, 3, 5, 7, 11, 13, 17, 19]:
        candidates.append(_zadoff_chu_h(m))

    # Chirps
    for f1, f2 in [(0, 1), (1, 2), (0, 3), (2, -1), (1, -2)]:
        candidates.append(_chirp_h(f1, f2))

    # Random with nonzero sum
    for seed in [1, 2, 3]:
        r = np.random.default_rng(seed)
        h = r.standard_normal(N)
        if np.abs(np.sum(h)) < 1.0:
            h += 0.5
        candidates.append(h)

    # Evaluate and pick best
    best_h = candidates[0]
    best_val = _value(best_h)
    for h in candidates[1:]:
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h

    # ---- Phase 2: DCT low-dimensional refinement ----
    # Parameterize h = B @ a, where B is DCT basis of size d
    d = 30  # reduced dimension
    B = _dct_basis_matrix(d)

    def _dct_h(a):
        return B @ a

    def _dct_smooth_obj(a, beta=50.0):
        h = B @ a
        return _smooth_max_value(h, beta)

    # Initialize a from best_h projection
    a0 = B @ best_h

    if time_ok():
        try:
            res = so.minimize(_dct_smooth_obj, a0, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                       'maxls': 20})
            h_opt = B @ res.x
            v_opt = _value(h_opt)
            if v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
        except Exception:
            pass

    # ---- Phase 3: DFT low-dimensional refinement ----
    d2 = 25
    B2 = _dft_basis_matrix(d2)

    def _dft_smooth_obj(a, beta=50.0):
        h = B2 @ a
        return _smooth_max_value(h, beta)

    a0_2 = B2 @ best_h
    if time_ok():
        try:
            res = so.minimize(_dft_smooth_obj, a0_2, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            h_opt = B2 @ res.x
            v_opt = _value(h_opt)
            if v_opt < best_val:
                best_val = v_opt
                best_h = h_opt
        except Exception:
            pass

    # ---- Phase 4: Full-dimension refinement with smooth max ----
    # Now refine in full 400-dim starting from best_h
    def _full_smooth_obj(h, beta=30.0):
        return _smooth_max_value(h, beta)

    if time_ok():
        try:
            res = so.minimize(_full_smooth_obj, best_h, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10,
                                       'maxls': 10})
            v_opt = _value(res.x)
            if v_opt < best_val:
                best_val = v_opt
                best_h = res.x
        except Exception:
            pass

    # ---- Phase 5: Random restarts in DCT space ----
    if time_ok():
        for _ in range(5):
            if not time_ok():
                break
            a_rand = B @ best_h + rng.standard_normal(d) * 0.05
            try:
                res = so.minimize(_dct_smooth_obj, a_rand, method='L-BFGS-B',
                                  options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
                h_opt = B @ res.x
                v_opt = _value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt
            except Exception:
                pass

    # ---- Phase 6: Final polish with higher beta ----
    if time_ok():
        try:
            def _polish_obj(h):
                return _smooth_max_value(h, beta=100.0)
            res = so.minimize(_polish_obj, best_h, method='L-BFGS-B',
                              options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12})
            v_opt = _value(res.x)
            if v_opt < best_val:
                best_val = v_opt
                best_h = res.x
        except Exception:
            pass

    # Validate final result
    s = np.sum(best_h)
    if not np.all(np.isfinite(best_h)) or np.abs(s) < 1e-12:
        return np.ones(N)

    return best_h

def construct():
    return _construct()
