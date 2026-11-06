import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # Safety margin

# Precompute indices
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    # Fast autoconvolution via FFT
    # Size needs to be at least 2*N - 1
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the objective.
    Uses LogSumExp to approximate max(conv).
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    # Shift to prevent overflow in exp
    m = np.max(conv)
    scaled = beta * (conv - m)
    # Clip to avoid numerical issues
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _dct_h(coeffs):
    """Reconstruct h from DCT-II coefficients."""
    h = scipy_idct(coeffs, type=2, n=N, norm='ortho')
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

def _dct_exact_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def _generate_initial_dct(rng, num_coeffs):
    """Generate random DCT coefficients with low-frequency bias."""
    coeffs = np.zeros(num_coeffs)
    coeffs[0] = rng.normal(0, 1.0) * 10.0
    weights = 1.0 / (1.0 + np.arange(num_coeffs))
    coeffs[1:] = rng.normal(0, 1.0, size=num_coeffs - 1) * weights[1:] * 2.0
    return coeffs

def _construct_atom(atom_len):
    """Construct a Hamming window atom."""
    if atom_len <= 1:
        return np.array([1.0])
    return np.hamming(atom_len)

def _sparse_atom_obj(params, atom, offsets, signs, beta=50.0):
    """
    Objective for sparse atom model.
    params: [amplitudes...] (len M)
    atom: fixed short vector
    offsets: fixed integer offsets
    signs: fixed signs (+1/-1)
    """
    M = len(params)
    h = np.zeros(N)
    for i in range(M):
        amp = params[i]
        off = offsets[i]
        sgn = signs[i]
        # Place atom at offset, wrapping around
        # Simple placement: if it goes out of bounds, it's truncated (or we can wrap)
        # Wrapping is better for periodic-like structures, but h is not necessarily periodic.
        # However, the autocorrelation is circular if we use FFT.
        # Let's use standard linear placement for now, but ensure it fits or wrap.
        # Actually, for N=400 and short atoms, linear placement is fine if offsets are chosen carefully.
        # But to allow any position, we can wrap.
        
        # Wrap-around placement
        indices = np.arange(off, off + len(atom)) % N
        h[indices] += sgn * amp * atom
        
    return _smooth_value(h, beta)

def construct():
    global _t0
    _t0 = time.time()
    
    # Start with the seed solution
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # --- Strategy 1: DCT Optimization (Original) ---
    K = 400
    restarts_dct = 10
    for r in range(restarts_dct):
        if not deadline_ok():
            break
        try:
            coeffs0 = _generate_initial_dct(rng, K)
            res = minimize(
                lambda c: _dct_smooth_obj(c, beta=20.0),
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if not deadline_ok():
                break
            coeffs_opt = res.x
            h_candidate = _dct_h(coeffs_opt)
            if np.all(np.isfinite(h_candidate)):
                update_best(h_candidate)
        except Exception:
            pass

    # --- Strategy 2: Sparse Atom Model ---
    # Try different atom lengths and numbers of atoms
    atom_lengths = [8, 16, 32]
    num_atoms_list = [5, 10, 20]
    
    for atom_len in atom_lengths:
        if not deadline_ok():
            break
        atom = _construct_atom(atom_len)
        
        for M in num_atoms_list:
            if not deadline_ok():
                break
                
            # Generate a set of offsets
            # We want to cover the range [0, N)
            # Random offsets or grid?
            # Let's try a few random configurations of offsets and signs
            
            num_configs = 5
            for cfg in range(num_configs):
                if not deadline_ok():
                    break
                
                try:
                    # Random offsets
                    offsets = rng.integers(0, N, size=M)
                    # Random signs
                    signs = rng.choice([-1, 1], size=M)
                    
                    # Initial amplitudes: random small values
                    params0 = rng.normal(0, 1.0, size=M)
                    
                    # Optimize amplitudes
                    # Use smooth surrogate
                    res = minimize(
                        lambda p: _sparse_atom_obj(p, atom, offsets, signs, beta=50.0),
                        params0,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    if not deadline_ok():
                        break
                    
                    params_opt = res.x
                    h_candidate = np.zeros(N)
                    for i in range(M):
                        amp = params_opt[i]
                        off = offsets[i]
                        sgn = signs[i]
                        indices = np.arange(off, off + len(atom)) % N
                        h_candidate[indices] += sgn * amp * atom
                    
                    if np.all(np.isfinite(h_candidate)):
                        update_best(h_candidate)
                        
                except Exception:
                    pass

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    # Simple self-test
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Shape: {h.shape}")
