"""Erdos minimum overlap - step heights h on [0,2] (N=100) in [0,1], sum=N/2,
minimize max circular overlap of h with shifted 1-h.

Max overlap at shift s = N/2 - C(s), C = cyclic autocorrelation = ifft(|fft(h)|^2).
So minimize max overlap <=> maximize min C(s).

Strategy (stronger than the seed):
  * Reparameterize h_i = softplus(alpha*x_i)/alpha  =>  0 < h_i < 1 automatically.
  * Work in the DFT domain: the objective depends only on |a_k|^2, which is a
    smooth quadratic in the free DFT coefficients.  Free coefficients are the
    independent set {1..N/2-1} (complex) plus a_{N/2} (real).
  * Objective = soft-max of the overlaps (dual surrogate) + a quadratic penalty
    for any h_i outside [0,1] (enforces the box via the penalty, no clipping that
    corrupts the DFT).
  * Multi-start L-BFGS-B with analytic gradients; final soft-min (high beta)
    polish with a sum-matching refinement that stays inside [0,1].
"""
import time
import numpy as np

N = 100
DEADLINE = 150.0
START = time.time()
HALF = N / 2.0
NFREE = N // 2 - 1          # free complex coefficients: indices 1..N/2-1
DIM = 2 * NFREE + 1         # +1 real coefficient a_{N/2}


def _time_left():
    return DEADLINE - (time.time() - START)


def _build_a(flat):
    """flat = [a1r,a1i,...,a_{N/2-1}r,a_{N/2-1}i, a_{N/2}]  -> full DFT a."""
    a = np.empty(N, dtype=complex)
    a[0] = HALF
    a[N // 2] = flat[-1]
    if NFREE:
        a[1:N // 2] = flat[:2 * NFREE].reshape(NFREE, 2)
        a[N // 2 + 1:] = np.conj(a[1:N // 2][::-1])
    return a


def _h_from_a(a, alpha=1.0):
    x = np.real(np.fft.ifft(a))
    # h = softplus(alpha*x)/alpha = (1/alpha)*log(1+exp(alpha*x))
    z = alpha * x
    return np.logaddexp(0.0, z) / alpha


def _softplus(z):
    return np.logaddexp(0.0, z)


def _objective(flat, beta=20.0, pen=300.0, alpha=1.0):
    a = _build_a(flat)
    power = np.abs(a) ** 2
    C = np.real(np.fft.ifft(power))          # cyclic autocorrelation
    overlap = HALF - C                        # overlap at each shift
    m = np.max(overlap)
    # soft-max of overlap (smooth max surrogate)
    softmax = m + np.log(np.exp(beta * (overlap - m)).sum()) / beta
    # box penalty on h
    x = np.real(np.fft.ifft(a))
    h = np.logaddexp(0.0, alpha * x) / alpha
    over = np.maximum(0.0, h - 1.0)
    under = np.maximum(0.0, -h)
    penval = pen * (np.sum(over * over) + np.sum(under * under))
    return softmax + penval


def _objective_grad(flat, beta=20.0, pen=300.0, alpha=1.0):
    a = _build_a(flat)
    power = np.abs(a) ** 2
    C = np.real(np.fft.ifft(power))
    overlap = HALF - C
    m = np.max(overlap)
    e = np.exp(beta * (overlap - m))
    w = e / e.sum()                           # softmax weights, sum=1
    # d(softmax)/dC[s] = -beta*w[s];  dC/d power = ifft basis -> use fft
    # C = ifft(power) => dC/d power = (1/N) e^{2pi i s k/N} = fft^{-1}... 
    # We need dC/d flat.  C = ifft(power).  dC_s/d power_k = (1/N) e^{2pi i s k/N}.
    # power_k = |a_k|^2.  For k in 1..N/2-1: d|a_k|^2/d a_k = 2 a_k (complex).
    # We'll build dC/d a (complex, length N) then map to flat.
    # dC/d power = ifft-adjoint:  dC/d power_k = conj( e^{2pi i k s/N} )? 
    # Simpler: dC_s = sum_k (1/N) e^{2pi i s k/N} d power_k.
    # d power_k / d flat -> build dC/da_k.
    dC_dpower = np.fft.ifft(np.conj(np.ones(N)))  # placeholder; compute directly below
    # Direct: dC/d power_k = (1/N) e^{2pi i k s / N} for each s.
    # Represent as matrix-free: dC = ifft( d power ).  So dC/d power = ifft basis.
    # We want g_flat = d(soft+pen)/d flat.
    # d(soft)/d power_k = sum_s (d soft/d C_s) dC_s/d power_k
    #                    = sum_s (-beta w_s) (1/N) e^{2pi i s k/N}
    # = (1/N) sum_s (-beta w_s) e^{2pi i s k/N} = (1/N) * fft(-beta*w)_k
    dsoft_dpower = np.fft.fft(-beta * w) / N
    # dsoft/d a_k = dsoft_dpower_k * 2 conj(a_k)  (since power=|a|^2)
    dsoft_da = dsoft_dpower * (2.0 * np.conj(a))
    # penalty gradient: d(pen)/d h_i * d h_i / d x_i * d x_i / d a
    x = np.real(np.fft.ifft(a))
    h = np.logaddexp(0.0, alpha * x) / alpha
    over = np.maximum(0.0, h - 1.0)
    under = np.maximum(0.0, -h)
    dpen_dh = 2.0 * pen * (over - under)
    dpen_dx = dpen_dh * alpha * (1.0 - np.exp(-alpha * x))  # sigmoid(alpha x)
    # x = real(ifft(a));  d x_i / d a_k = real( (1/N) e^{2pi i k i/N} )
    # dpen/da_k = sum_i dpen_dx_i * real( (1/N) e^{2pi i k i/N} )
    # Build via: dpen_da = fft-like.  d x_i/d a_k = (1/N) Re(e^{2pi i k i/N}).
    # sum_i dpen_dx_i Re(e^{2pi i k i/N}) = Re( sum_i dpen_dx_i e^{2pi i k i/N} ) = Re( fft(dpen_dx)_k )
    dpen_da = np.real(np.fft.fft(dpen_dx)) / N
    dpen_da = dpen_da + 0j
    d_da = dsoft_da + dpen_da
    # map d_da (complex, N) to flat gradient (real)
    g = np.empty(DIM)
    if NFREE:
        # flat[:2*NFREE] = [a1r,a1i,...];  d f / d a1r = Re(d_da[1]); /d a1i = -Im(d_da[1])
        for k in range(NFREE):
            g[2 * k] = d_da[k + 1].real
            g[2 * k + 1] = -d_da[k + 1].imag
    g[-1] = d_da[N // 2].real
    val = _objective(flat, beta, pen, alpha)
    return val, g


def _score(h):
    C = np.real(np.fft.ifft(np.abs(np.fft.fft(h)) ** 2))
    return HALF - np.min(C)


def _flat_to_h(flat, alpha=1.0):
    return _h_from_a(_build_a(flat), alpha)


def _h_to_flat(h):
    a = np.fft.fft(h)
    flat = np.empty(DIM)
    if NFREE:
        flat[:2 * NFREE] = np.column_stack([a[1:N // 2].real, a[1:N // 2].imag]).ravel()
    flat[-1] = a[N // 2].real
    return flat


def _refine_sum(h, target=HALF):
    """Adjust h so sum==target while keeping in [0,1], minimal change."""
    for _ in range(60):
        s = h.sum()
        d = target - s
        if abs(d) < 1e-12:
            break
        # distribute d to elements with room in the right direction
        if d > 0:
            room = 1.0 - h
            if room.max() <= 0:
                break
        else:
            room = h
            if room.max() <= 0:
                break
        idx = np.argmax(room)
        step = np.sign(d) * min(abs(d), room[idx])
        h[idx] += step
    return h


def _optimize(flat0, beta, pen, alpha, maxiter):
    from scipy.optimize import minimize
    res = minimize(_objective_grad, flat0, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'maxls': 20,
                            'ftol': 1e-14, 'gtol': 1e-8})
    return res


def construct():
    rng = np.random.default_rng(777)
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)

    alpha = 3.0  # sharpness of softplus (larger => closer to indicator)

    # ---- build diverse starts in DFT domain ----
    starts = []
    # (a) flat-ish: small random DFT
    for _ in range(6):
        f = np.zeros(DIM)
        f[:2 * NFREE] = rng.normal(0.0, 0.3, size=2 * NFREE)
        f[-1] = 0.5
        starts.append(f)
    # (b) low-frequency concentrated
    for nm in [1, 2, 3, 5, 8, 12, 20, 30]:
        f = np.zeros(DIM)
        for k in range(1, min(nm, NFREE) + 1):
            mag = 0.35
            f[2 * (k - 1)] = mag * np.cos(2 * np.pi * k / N)
            f[2 * (k - 1) + 1] = mag * np.sin(2 * np.pi * k / N)
        f[-1] = 0.5
        starts.append(f)
    # (c) random phases fixed mags
    for _ in range(6):
        f = np.zeros(DIM)
        mags = rng.uniform(0.1, 0.5, size=NFREE)
        ph = rng.uniform(0, 2 * np.pi, size=NFREE)
        for k in range(NFREE):
            f[2 * k] = mags[k] * np.cos(ph[k])
            f[2 * k + 1] = mags[k] * np.sin(ph[k])
        f[-1] = rng.uniform(0.0, 1.0)
        starts.append(f)

    order = rng.permutation(len(starts))
    for idx in order:
        if _time_left() < 8.0:
            break
        f0 = starts[idx].copy()
        try:
            res = _optimize(f0, beta=20.0, pen=300.0, alpha=alpha, maxiter=350)
            h = _flat_to_h(res.x, alpha)
            h = np.clip(h, 0.0, 1.0)
            h = _refine_sum(h)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
        except Exception:
            continue

    # ---- polish best: increase beta (sharper max surrogate), lower penalty ----
    if _time_left() > 20.0:
        try:
            f0 = _h_to_flat(best_h)
            res = _optimize(f0, beta=60.0, pen=1000.0, alpha=alpha, maxiter=1500)
            h = _flat_to_h(res.x, alpha)
            h = np.clip(h, 0.0, 1.0)
            h = _refine_sum(h)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
        except Exception:
            pass

    # ---- final soft-min (high beta) polish ----
    if _time_left() > 30.0:
        try:
            f0 = _h_to_flat(best_h)
            res = _optimize(f0, beta=200.0, pen=3000.0, alpha=alpha, maxiter=2500)
            h = _flat_to_h(res.x, alpha)
            h = np.clip(h, 0.0, 1.0)
            h = _refine_sum(h)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
        except Exception:
            pass

    # ---- safety ----
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _refine_sum(best_h)
    return best_h
