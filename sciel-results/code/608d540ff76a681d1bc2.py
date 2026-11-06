"""Rewritten: spectral peak-suppression + atom warm start + coordinate polish."""
import time
import numpy as np
from numpy.fft import fft, ifft

N = 400
DEADLINE = 140.0  # safety margin under 150s kill

def _value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    R = np.convolve(h, h)
    return abs(2.0 * N * np.max(R) / (s * s))

def _atom_base(n_atoms=8, atom_len=10, seed=0):
    """Build h as sparse sum of shifted, scaled, sign-flipped Hamming atoms."""
    rng = np.random.default_rng(seed)
    atom = np.hamming(atom_len)
    atom = atom / np.sum(atom)
    h = np.zeros(N)
    # deterministic offsets spread across the domain
    offsets = np.linspace(0, N - atom_len, n_atoms, dtype=int)
    for i, off in enumerate(offsets):
        sign = 1.0 if (i % 2 == 0) else -1.0
        amp = 1.0 + 0.3 * np.sin(i * 1.7)
        h[off:off + atom_len] += sign * amp * atom
    return h

def _spectral_peak_suppress(h, iters=200, lr=0.05):
    """
    Iteratively reduce the maximum signed autocorrelation peak.
    At each step, find lag k* with max R(k*), then apply a frequency-domain
    update that reduces R(k*) while preserving DC (sum) and minimizing L2 change.
    """
    h = h.astype(np.float64).copy()
    best_h = h.copy()
    best_v = _value(h)
    n = len(h)
    # Precompute FFT size (next power of 2 for speed, but use exact for correctness)
    fft_n = 2 * n - 1
    # Pad to a convenient size
    pad_n = 1
    while pad_n < fft_n:
        pad_n *= 2
    H = fft(h, pad_n)
    H_abs2 = H * np.conj(H)
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        # Compute autocorrelation from spectrum
        R = np.real(ifft(H_abs2, pad_n))[:fft_n]
        k_star = int(np.argmax(R))
        R_max = R[k_star]
        # Target: reduce R(k_star). 
        # dR(k)/dH(f) = 2*Re{ H(f) * exp(-2j pi f k / n) } (for full conv)
        # We want a step in H that decreases R(k_star) most per unit L2 change in H,
        # subject to preserving DC (H[0] real, fixed).
        # Gradient of R(k) w.r.t. Re(H[f]) and Im(H[f]):
        # R(k) = sum_f |H[f]|^2 * exp(2j pi f k / n)  (for the full linear conv, with appropriate indexing)
        # Actually for linear autocorrelation: R[k] = sum_t h[t] h[t+k]
        # In freq: R = ifft(fft(h)*conj(fft(h)))
        # d R[k] / d Re(H[f]) = 2 Re{ H[f] * conj( exp(2j pi f k / n) ) } = 2 Re{ H[f] * exp(-2j pi f k / n) }
        # d R[k] / d Im(H[f]) = 2 Im{ H[f] * exp(-2j pi f k / n) } * (-1) ... let's just use complex gradient.
        # Simpler: treat as real optimization. Use subgradient: 
        # step H[f] -= lr * dR/dH[f] for f != 0, keep H[0] fixed.
        # dR/dH[f] (complex) = 2 * conj(H[f]) * exp(2j pi f k / n)  ... 
        # Let's use: we want to minimize R[k_star]. 
        # Gradient of R[k] w.r.t. H[f] (complex, treating H and conj(H) as independent for Wirtinger):
        # R[k] = (1/pad_n) sum_f |H[f]|^2 e^{2j pi f k / pad_n}
        # dR/d conj(H[f]) = (1/pad_n) H[f] e^{2j pi f k / pad_n}
        # So descent: H[f] -= lr * conj( dR/d conj(H[f]) ) = lr * conj(H[f]) e^{2j pi f k / pad_n} ... 
        # Actually for real h, H[-f] = conj(H[f]), so we can work with real FFT.
        # Let's use a simpler, robust approach: 
        # Modify h directly by subtracting lr * g where g is the "peak shape" 
        # g[t] = h[t + k_star] (shifted version), which reduces R[k_star] by lr * ||h||^2 approximately.
        # But this changes sum. Let's do it in frequency to preserve DC.
        
        # Frequency-domain approach:
        # We want delta_H such that sum_f |H+delta_H|^2 e^{2jpi f k*/n} is reduced.
        # First order: delta R ≈ 2 Re{ sum_f H[f] conj(delta_H[f]) e^{2j pi f k*/n} }
        # Choose delta_H[f] = -lr * H[f] e^{-2j pi f k*/n} for f != 0, delta_H[0] = 0.
        # Then delta R ≈ -2 lr Re{ sum_{f!=0} |H[f]|^2 } = -2 lr * (R[0] - |H[0]|^2/pad_n) < 0. Good.
        # This is a "peak reduction" step.
        f_idx = np.arange(pad_n)
        phase = np.exp(2j * np.pi * f_idx * k_star / pad_n)
        delta_H = -lr * H * phase
        delta_H[0] = 0.0  # preserve DC
        H = H + delta_H
        h_new = np.real(ifft(H, pad_n))[:n]
        v_new = _value(h_new)
        if v_new < best_v:
            best_v = v_new
            best_h = h_new.copy()
            h = h_new
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-6:
                break
    return best_h, best_v

def _coordinate_polish(h, iters=500):
    """Small per-element adjustments to shave residual peaks."""
    h = h.astype(np.float64).copy()
    best_h = h.copy()
    best_v = _value(h)
    n = len(h)
    fft_n = 2 * n - 1
    pad_n = 1
    while pad_n < fft_n:
        pad_n *= 2
    step = 0.01
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        R = np.convolve(h, h)
        k_star = int(np.argmax(R))
        # Try adjusting each element by +/- step, pick best
        best_local_v = best_v
        best_local_h = h.copy()
        # Only try a subset of elements for speed
        candidates = np.random.default_rng(it).choice(n, size=min(20, n), replace=False)
        for t in candidates:
            for delta in (step, -step):
                h[t] += delta
                v = _value(h)
                if v < best_local_v:
                    best_local_v = v
                    best_local_h = h.copy()
                h[t] -= delta
        if best_local_v < best_v:
            best_v = best_local_v
            best_h = best_local_h.copy()
            h = best_local_h
        else:
            step *= 0.7
            if step < 1e-8:
                break
    return best_h, best_v

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    # Try several atom-based initializations
    for seed in range(10):
        if time.time() > DEADLINE - 10:
            break
        h0 = _atom_base(n_atoms=6 + seed % 5, atom_len=8 + (seed % 3) * 2, seed=seed)
        # Normalize sum to 1
        s = np.sum(h0)
        if abs(s) < 1e-10:
            continue
        h0 = h0 / s
        v0 = _value(h0)
        if v0 < best_v:
            best_v = v0
            best_h = h0.copy()
    
    # Spectral peak suppression
    h1, v1 = _spectral_peak_suppress(best_h, iters=150, lr=0.03)
    if v1 < best_v:
        best_v = v1
        best_h = h1
    
    # Coordinate polish
    if time.time() < DEADLINE - 5:
        h2, v2 = _coordinate_polish(best_h, iters=300)
        if v2 < best_v:
            best_v = v2
            best_h = h2
    
    # Final safety: ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    return best_h
