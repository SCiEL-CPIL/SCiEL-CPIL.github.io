"""
Sparse atom-based step height construction for third autocorrelation inequality.

Strategy:
- Model h as a sparse linear combination of shifted, scaled copies of a short Hamming atom.
- Use multiple optimization phases: DE for global search, L-BFGS-B for local refinement.
- Exploit mirror symmetry to reduce search space.
- Normalize sum(h) to decouple shape from scale.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 135.0

# Shorter atom for better resolution
_ATOM_LEN = 8
_atom = np.hamming(_ATOM_LEN)

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _value_normalized(h):
    """Value depends only on shape (normalized h)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    h_norm = h / s
    conv = _fft_convolve(h_norm)
    m = float(np.max(conv))
    return abs(2.0 * N * m)

def _build_h_from_params(params, n_atoms, use_symmetry=False):
    """
    Build h from atom parameters.
    params: [amp_0, offset_0, amp_1, offset_1, ..., base]
    If use_symmetry, each atom has a mirror partner with opposite sign.
    """
    h = np.zeros(N)
    for k in range(n_atoms):
        amp = params[2 * k]
        offset = int(round(params[2 * k + 1]))
        
        # Place atom at offset
        start = offset
        end = start + _ATOM_LEN
        s_idx = max(0, start)
        e_idx = min(N, end)
        if s_idx < e_idx:
            h[s_idx:e_idx] += amp * _atom[s_idx - start:e_idx - start]
        
        if use_symmetry:
            # Mirror position
            start2 = N - _ATOM_LEN - offset
            end2 = start2 + _ATOM_LEN
            s_idx2 = max(0, start2)
            e_idx2 = min(N, end2)
            if s_idx2 < e_idx2:
                h[s_idx2:e_idx2] -= amp * _atom[s_idx2 - start2:e_idx2 - start2]
    
    base = params[-1]
    h += base
    return h

def _objective(params, n_atoms, use_symmetry=False):
    """Objective function for optimization."""
    h = _build_h_from_params(params, n_atoms, use_symmetry)
    return _value_normalized(h)

def _objective_grad(params, n_atoms, use_symmetry=False):
    """Objective with gradient (finite differences for simplicity)."""
    h = _build_h_from_params(params, n_atoms, use_symmetry)
    val = _value_normalized(h)
    
    # Compute gradient via central differences
    grad = np.zeros_like(params)
    eps = 1e-4
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        h_plus = _build_h_from_params(params_plus, n_atoms, use_symmetry)
        val_plus = _value_normalized(h_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        h_minus = _build_h_from_params(params_minus, n_atoms, use_symmetry)
        val_minus = _value_normalized(h_minus)
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

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
    
    def local_refine(h_init, max_iter=100):
        """Refine h using L-BFGS-B with finite-difference gradients."""
        if not deadline_ok():
            return h_init
        
        h_work = h_init.copy()
        s_init = np.sum(h_work)
        if abs(s_init) < 1e-12:
            return h_init
        
        h_norm = h_work / s_init
        
        # Try different atom configurations
        for n_atoms in [6, 8, 10]:
            for use_sym in [False, True]:
                if not deadline_ok():
                    break
                
                # Initialize params from h_norm
                params0 = np.zeros(2 * n_atoms + 1)
                # Spread atoms evenly
                for k in range(n_atoms):
                    params0[2 * k + 1] = k * (N // n_atoms)
                
                # Run L-BFGS-B
                try:
                    res = minimize(
                        _objective,
                        params0,
                        args=(n_atoms, use_sym),
                        method='L-BFGS-B',
                        jac=None,  # Use finite differences
                        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
                    )
                    if res.success and res.fun < best_v:
                        h_candidate = _build_h_from_params(res.x, n_atoms, use_sym)
                        update_best(h_candidate)
                except Exception:
                    pass
        
        return best_h
    
    # Phase 1: Differential evolution with various configurations
    configs = [
        (6, False, 42),
        (8, False, 7),
        (10, False, 13),
        (5, True, 17),
        (7, True, 19),
    ]
    
    for n_atoms, use_sym, seed in configs:
        if not deadline_ok():
            break
        
        try:
            n_params = 2 * n_atoms + 1
            bounds = []
            for k in range(n_atoms):
                bounds.append((-3.0, 3.0))  # amplitude
                bounds.append((0.0, float(N - _ATOM_LEN)))  # offset
            bounds.append((-1.0, 1.0))  # base
            
            popsize = 20 if n_atoms <= 8 else 15
            maxiter = 80 if n_atoms <= 8 else 60
            
            res = differential_evolution(
                _objective,
                bounds,
                args=(n_atoms, use_sym),
                seed=seed,
                maxiter=maxiter,
                tol=1e-8,
                popsize=popsize,
                init='sobol',
                polish=False,
                update_interval=1
            )
            
            if res.x is not None:
                h_candidate = _build_h_from_params(res.x, n_atoms, use_sym)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Local refinement from best solution
    if deadline_ok():
        try:
            local_refine(best_h, max_iter=50)
        except Exception:
            pass
    
    # Phase 3: Random restarts with local refinement
    if deadline_ok():
        n_restarts = 10
        for r in range(n_restarts):
            if not deadline_ok():
                break
            
            try:
                # Random initialization
                n_atoms = 6 + (r % 3)
                use_sym = (r % 2 == 0)
                params0 = np.random.randn(2 * n_atoms + 1)
                params0[1::2] = np.random.randint(0, N - _ATOM_LEN, n_atoms)
                
                h_candidate = _build_h_from_params(params0, n_atoms, use_sym)
                v_candidate = _value_normalized(h_candidate)
                
                if v_candidate < best_v * 1.5:  # Only refine promising starts
                    # Quick local refinement
                    try:
                        res = minimize(
                            _objective,
                            params0,
                            args=(n_atoms, use_sym),
                            method='L-BFGS-B',
                            jac=None,
                            options={'maxiter': 30, 'ftol': 1e-8}
                        )
                        if res.x is not None:
                            h_candidate = _build_h_from_params(res.x, n_atoms, use_sym)
                            update_best(h_candidate)
                    except Exception:
                        pass
            except Exception:
                pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
