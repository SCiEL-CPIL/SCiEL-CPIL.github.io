"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_t0 = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _value(h):
    """Compute the exact objective value."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _atom_basis(num_atoms, atom_len, offsets):
    """
    Build a set of basis vectors: shifted, sign-flipped copies of a Hamming atom.
    Returns a list of N-length vectors.
    """
    atom = np.hamming(atom_len)
    atom = atom / np.sum(atom) if np.sum(atom) != 0 else atom
    basis = []
    for sign in [1.0, -1.0]:
        for off in offsets:
            v = np.zeros(N)
            start = off
            end = start + atom_len
            if end <= N and start >= 0:
                v[start:end] = sign * atom
            elif start < 0:
                # Wrap or clip - just clip for simplicity
                v[0:end] = sign * atom[-start:]
            else:
                continue
            basis.append(v)
    return basis

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Known optimal: constant vector gives value = 2.0
    best_h = np.ones(N)
    best_v = 2.0
    
    # Structural approach: parameterize h as sparse linear combination of atoms
    # This reduces dimensionality from 400 to ~20-40 parameters
    
    atom_len = 10
    # Choose offsets spread across the domain
    num_offsets = 15
    offsets = list(range(0, N, N // num_offsets))
    # Ensure we have valid offsets
    offsets = [o for o in offsets if o + atom_len <= N]
    
    basis = _atom_basis(num_atoms=4, atom_len=atom_len, offsets=offsets)
    num_params = len(basis)
    
    if num_params == 0:
        return best_h
    
    basis_arr = np.array(basis)  # shape (num_params, N)
    
    def _reconstruct(params):
        return params @ basis_arr
    
    def _obj(params):
        h = _reconstruct(params)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2.0 * N * m / (s * s))
    
    # Try multiple initializations
    rng = np.random.default_rng(12345)
    
    initializations = []
    # 1. All positive equal weights (approximates constant)
    initializations.append(np.ones(num_params) / num_params)
    # 2. Random
    for _ in range(5):
        initializations.append(rng.standard_normal(num_params) * 0.1)
    # 3. Sparse random
    for _ in range(3):
        init = np.zeros(num_params)
        idx = rng.choice(num_params, size=max(1, num_params // 4), replace=False)
        init[idx] = rng.standard_normal(len(idx)) * 0.5
        initializations.append(init)
    
    for init in initializations:
        if not deadline_ok():
            break
        try:
            res = minimize(
                _obj,
                init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                h = _reconstruct(res.x)
                v = _value(h)
                if v < best_v - 1e-12:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Also try a direct low-dimensional parameterization:
    # h[i] = a + b*cos(2*pi*i/N) + c*sin(2*pi*i/N) + d*cos(4*pi*i/N)
    # This is a 4-parameter family that includes the constant (a, b=c=d=0)
    if deadline_ok():
        i = np.arange(N)
        def _obj4(params):
            a, b, c, d = params
            h = a + b * np.cos(2 * np.pi * i / N) + c * np.sin(2 * np.pi * i / N) + d * np.cos(4 * np.pi * i / N)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h)
            m = np.max(conv)
            return abs(2.0 * N * m / (s * s))
        
        for init4 in [np.array([1.0, 0, 0, 0]), np.array([1.0, 0.1, 0.1, 0]), np.array([1.0, -0.1, 0.05, 0.05])]:
            if not deadline_ok():
                break
            try:
                res4 = minimize(
                    _obj4,
                    init4,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * 4,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res4.x is not None:
                    a, b, c, d = res4.x
                    h = a + b * np.cos(2 * np.pi * i / N) + c * np.sin(2 * np.pi * i / N) + d * np.cos(4 * np.pi * i / N)
                    v = _value(h)
                    if v < best_v - 1e-12:
                        best_v = v
                        best_h = h.copy()
            except Exception:
                pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    else:
        # Return the provably optimal constant vector
        return np.ones(N)
# EVOLVE-BLOCK-END
