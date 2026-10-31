"""Minimize max|autoconv| / (sum(x))^2 using number-theoretic low-discrepancy
sequences (Van der Corput / Kronecker) combined with gradient refinement.
"""
import numpy as np
import time
import math

N = 400

def _autoconv_full(x):
    """Full autoconvolution via FFT."""
    n = len(x)
    size = 2 * n
    fx = np.fft.fft(x, size)
    return np.fft.ifft(fx * np.conj(fx)).real

def _autoconv_max_abs(x):
    ac = _autoconv_full(x)
    return float(np.max(np.abs(ac)))

def _score(x):
    s = float(np.sum(x))
    if abs(s) < 1e-14:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

# ---------- Number-theoretic sequence generators ----------

def _van_der_corput(n, base=2):
    """Return the first n terms of the Van der Corput sequence in [0,1)."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0 / base
        d = float(i)
        v = 0.0
        while d > 0:
            d = int(d)
            v += f * (d % base)
            d //= base
            f /= base
        result[i] = v
    return result

def _kronecker_sequence(n, alpha):
    """Kronecker sequence: {i * alpha} mod 1 for i = 0..n-1."""
    i = np.arange(n, dtype=np.float64)
    return (i * alpha) % 1.0

def _make_signs_from_fractional_parts(fracs, threshold=0.5):
    """Map fractional parts in [0,1) to signs in {+1, -1}.
    Use a smooth transition near threshold to allow real-valued optimization."""
    # Hard binary sign
    signs = np.where(fracs < threshold, 1.0, -1.0)
    return signs

def _make_smooth_signs_from_fractional_parts(fracs, threshold=0.5, sharpness=10.0):
    """Smooth transition using tanh for gradient-based refinement."""
    return np.tanh(sharpness * (threshold - fracs))

# ---------- Gradient helpers ----------

def _smooth_objective(x, lam=0.05):
    """Smooth approximation of max|autoconv| using log-sum-exp."""
    ac = _autoconv_full(x)
    abs_ac = np.abs(ac[:len(x)])  # positive lags only
    scaled = abs_ac / lam
    m = np.max(scaled)
    if m > 50:
        return lam * (m + np.log(np.sum(np.exp(scaled - m))))
    return lam * np.log(np.sum(np.exp(scaled)))

def _gradient_smooth(x, lam=0.05):
    """Gradient of smooth max|autoconv| w.r.t. x."""
    ac = _autoconv_full(x)
    n = len(x)
    abs_ac = np.abs(ac[:n])
    scaled = abs_ac / lam
    m = np.max(scaled)
    exp_s = np.exp(scaled - m)
    w = exp_s / np.sum(exp_s)
    
    grad = np.zeros(n)
    for k in range(n):
        if abs_ac[k] > 1e-14:
            sign_k = 1.0 if ac[k] > 0 else -1.0
        else:
            sign_k = 1.0
        wk = w[k] * sign_k
        if k == 0:
            grad += wk * 2.0 * x
        else:
            grad[:n-k] += wk * x[k:]
            grad[k:] += wk * x[:n-k]
    return grad

def _project(x, lo=-1.0, hi=1.0):
    return np.clip(x, lo, hi)

# ---------- Optimization ----------

def _gradient_refine(x_init, lr=0.02, max_iters=3000, lam=0.05, deadline=None):
    x = x_init.astype(np.float64).copy()
    best_x = x.copy()
    best_val = _score(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        g = _gradient_smooth(x, lam)
        # Normalize gradient step
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            x -= lr * g / gn * np.max(np.abs(x))
        x = _project(x)
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        # Adaptive LR decay
        if it % 500 == 499:
            lr *= 0.8
    return best_x, best_val

def _local_flip_search(x, deadline=None, max_flips=5000):
    """Try flipping individual elements or small groups to reduce score."""
    x = x.astype(np.float64).copy()
    best_val = _score(x)
    n = len(x)
    rng = np.random.RandomState(123)
    
    attempts = 0
    while attempts < max_flips:
        if deadline is not None and time.time() > deadline:
            break
        # Random position
        j = rng.randint(n)
        # Flip sign (for near-binary) or perturb
        x_old = x[j]
        if abs(x_old) > 0.5:
            x[j] = -x_old  # flip
        else:
            x[j] = 1.0 - x[j]  # complement-ish
        val = _score(x)
        if val < best_val:
            best_val = val
        else:
            x[j] = x_old
        attempts += 1
    return x, best_val

# ---------- Construct ----------

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin (evaluator kills at 150s)
    
    best_x = None
    best_val = float('inf')
    
    # Candidate 1: All ones (baseline)
    cands = [np.ones(N)]
    
    # Candidate 2: Van der Corput signs (base 2)
    vd = _van_der_corput(N, base=2)
    cands.append(_make_signs_from_fractional_parts(vd, 0.5))
    
    # Candidate 3: Van der Corput (base 3)
    vd3 = _van_der_corput(N, base=3)
    cands.append(_make_signs_from_fractional_parts(vd3, 0.5))
    
    # Candidate 4: Kronecker with golden ratio conjugate
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    alpha_gr = golden - 1.0  # ~0.618
    kr = _kronecker_sequence(N, alpha_gr)
    cands.append(_make_signs_from_fractional_parts(kr, 0.5))
    
    # Candidate 5: Kronecker with sqrt(2)/2
    alpha_sq2 = math.sqrt(2.0) / 2.0
    kr2 = _kronecker_sequence(N, alpha_sq2)
    cands.append(_make_signs_from_fractional_parts(kr2, 0.5))
    
    # Candidate 6: Kronecker with pi/4
    alpha_pi4 = math.pi / 4.0
    kr3 = _kronecker_sequence(N, alpha_pi4)
    cands.append(_make_signs_from_fractional_parts(kr3, 0.5))
    
    # Candidate 7: Kronecker with sqrt(3)/2
    alpha_sq3 = math.sqrt(3.0) / 2.0
    kr4 = _kronecker_sequence(N, alpha_sq3)
    cands.append(_make_signs_from_fractional_parts(kr4, 0.5))
    
    # Candidate 8: Combined low-discrepancy: interleave VdC and Kronecker
    inter = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            inter[i] = vd[i // 2] if i // 2 < len(vd) else 0.0
        else:
            inter[i] = kr[i // 2] if i // 2 < len(kr) else 0.0
    cands.append(_make_signs_from_fractional_parts(inter, 0.5))
    
    # Candidate 9: Smoothed VdC for gradient refinement
    cands.append(_make_smooth_signs_from_fractional_parts(vd, 0.5, sharpness=8.0))
    
    # Candidate 10: Smoothed Kronecker (golden)
    cands.append(_make_smooth_signs_from_fractional_parts(kr, 0.5, sharpness=8.0))
    
    # Evaluate and refine each candidate
    n_cands = len(cands)
    time_per_cand = (deadline - start_time) / n_cand
    
    for i, cand in enumerate(cands):
        if time.time() > deadline - 2.0:
            break
        cand = cand.astype(np.float64)
        
        # Quick score
        v = _score(cand)
        if v < best_val:
            best_val = v
            best_x = cand.copy()
        
        # Gradient refinement with limited iterations
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
        max_iters = int(remaining / (n_cands - i) * 300)
        max_iters = max(200, min(max_iters, 5000))
        
        x_opt, val_opt = _gradient_refine(cand, lr=0.03, max_iters=max_iters,
                                           lam=0.05, deadline=deadline - 2.0)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Final local flip search on best
    if best_x is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        max_flips = int(remaining * 500)
        x_opt, val_opt = _local_flip_search(best_x, deadline=deadline - 1.0,
                                             max_flips=max_flips)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Final gradient polish
    if best_x is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        max_iters = int(remaining * 200)
        max_iters = max(100, min(max_iters, 3000))
        x_opt, val_opt = _gradient_refine(best_x, lr=0.01, max_iters=max_iters,
                                           lam=0.03, deadline=deadline - 0.5)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    if best_x is None:
        return np.ones(N)
    return best_x

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
