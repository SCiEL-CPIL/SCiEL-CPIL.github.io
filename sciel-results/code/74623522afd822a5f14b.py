"""
New program: Sparse Atom-Based Step Height Construction.

Strategy:
1. Define a short 'atom' (length 10 Hamming window).
2. Model h as a sum of K shifted, scaled, sign-flipped atoms.
3. Parameters: [amp_0, off_0, sign_0, ..., amp_{K-1}, off_{K-1}, sign_{K-1}].
   Offsets are integers.
4. Use Differential Evolution to optimize these parameters.
5. Refine with local search on the full vector.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

N = 400
_t0 = time.time()
_DEADLINE = 115.0 # Leave buffer for evaluator overhead

# Precompute atom
_ATOM_LEN = 10
_atom = np.hamming(_ATOM_LEN)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _value_normalized(h):
    """Value with h normalized to sum=1. Value is scale-invariant."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    h_norm = h / s
    conv = _fft_convolve(h_norm)
    m = float(np.max(conv))
    return abs(2.0 * N * m)

def _build_h(params):
    """
    Build h from sparse atom parameters.
    params: [amp_0, off_0, sign_0, amp_1, off_1, sign_1, ...]
    K = len(params) // 3
    """
    K = len(params) // 3
    h = np.zeros(N)
    for k in range(K):
        amp = params[3 * k]
        off = int(round(params[3 * k + 1]))
        sign = 1.0 if params[3 * k + 2] >= 0 else -1.0
        
        # Clip offset to valid range
        if off < 0 or off >= N:
            continue
            
        # Determine slice
        start = off
        end = off + _ATOM_LEN
        
        # Handle boundaries
        if start < 0:
            # Atom extends left
            left_cut = -start
            h[0:end+start] += sign * amp * _atom[left_cut:]
        elif end > N:
            # Atom extends right
            right_cut = end - N
            h[start:N] += sign * amp * _atom[:N-start]
        else:
            # Fully inside
            h[start:end] += sign * amp * _atom
            
    return h

def _obj(params):
    """Objective function for DE."""
    h = _build_h(params)
    return _value_normalized(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
        
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v

    # Phase 1: Differential Evolution with sparse atom model
    # Try different numbers of atoms
    for K in [8, 12, 16]:
        if not deadline_ok():
            break
        try:
            n_params = 3 * K
            bounds = []
            for k in range(K):
                bounds.append((-1.0, 1.0))   # amplitude
                bounds.append((0.0, float(N - 1)))  # offset
                bounds.append((-1.0, 1.0))   # sign
                
            # Use small popsize and maxiter to fit in time
            res = differential_evolution(
                _obj, bounds, seed=42+K, maxiter=50, tol=1e-6,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _build_h(res.x)
                update_best(h)
        except Exception:
            pass

    # Phase 2: Local Search on full vector
    # We refine the best h found so far.
    if deadline_ok():
        try:
            h_work = best_h.copy()
            s_init = np.sum(h_work)
            if abs(s_init) < 1e-12:
                h_work = np.ones(N)
                s_init = N
            h_norm = h_work / s_init
            
            # Gradient-based local search
            # We want to minimize max(conv(h_norm)).
            # Let M = max(conv). The gradient of M w.r.t h is non-zero only at the peak lag.
            # However, M is non-smooth. We can use a soft-max approximation or just coordinate descent.
            # Coordinate descent is safer and often effective for this kind of problem.
            
            step_size = 0.05
            for it in range(100):
                if not deadline_ok():
                    break
                improved = False
                # Shuffle indices for randomization
                idxs = np.random.permutation(N)
                for i in idxs:
                    # Try increasing and decreasing h[i]
                    for delta in [step_size, -step_size]:
                        h_trial = h_norm.copy()
                        h_trial[i] += delta
                        v_trial = _value_normalized(h_trial)
                        v_curr = _value_normalized(h_norm)
                        if v_trial < v_curr - 1e-9:
                            h_norm = h_trial
                            improved = True
                            break
                if not improved:
                    step_size *= 0.7
                    if step_size < 1e-5:
                        break
                update_best(h_norm * s_init)
                
        except Exception:
            pass

    # Phase 3: Symmetric Atom Construction
    # Try to enforce symmetry which might help reduce peak.
    # h[i] = -h[N-1-i] or similar? 
    # Let's try a specific structured search: 
    # Atoms placed symmetrically with opposite signs.
    if deadline_ok():
        try:
            K_pairs = 6
            def _obj_sym(params):
                h = np.zeros(N)
                for k in range(K_pairs):
                    amp = params[2*k]
                    off = int(round(params[2*k+1]))
                    sign = 1.0 if amp >= 0 else -1.0
                    amp_abs = abs(amp)
                    
                    # Place atom at off
                    s1 = max(0, off)
                    e1 = min(N, off + _ATOM_LEN)
                    if s1 < e1:
                        h[s1:e1] += sign * amp_abs * _atom[s1-off:e1-off]
                    
                    # Place opposite atom at N - _ATOM_LEN - off
                    off2 = N - _ATOM_LEN - off
                    s2 = max(0, off2)
                    e2 = min(N, off2 + _ATOM_LEN)
                    if s2 < e2:
                        h[s2:e2] -= sign * amp_abs * _atom[s2-off2:e2-off2]
                        
                return _value_normalized(h)
            
            bounds_sym = [(-1.0, 1.0), (0.0, float(N - _ATOM_LEN))] * K_pairs
            res = differential_evolution(
                _obj_sym, bounds_sym, seed=100, maxiter=50, tol=1e-6,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = np.zeros(N)
                for k in range(K_pairs):
                    amp = res.x[2*k]
                    off = int(round(res.x[2*k+1]))
                    sign = 1.0 if amp >= 0 else -1.0
                    amp_abs = abs(amp)
                    
                    s1 = max(0, off)
                    e1 = min(N, off + _ATOM_LEN)
                    if s1 < e1:
                        h[s1:e1] += sign * amp_abs * _atom[s1-off:e1-off]
                    
                    off2 = N - _ATOM_LEN - off
                    s2 = max(0, off2)
                    e2 = min(N, off2 + _ATOM_LEN)
                    if s2 < e2:
                        h[s2:e2] -= sign * amp_abs * _atom[s2-off2:e2-off2]
                        
                update_best(h)
        except Exception:
            pass

    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
