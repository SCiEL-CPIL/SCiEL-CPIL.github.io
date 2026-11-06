"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 10.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _apply_beta(psi, beta, n):
    """Apply exp(-i beta sum_k X_k) to state vector psi using vectorized operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    # For each qubit k, the X rotation acts on that qubit's 0/1 subspace
    # We use reshape to group by qubit k
    for k in range(n):
        # Reshape so that dimension k is the middle axis of size 2
        shape = (2**(n-k-1), 2, 2**k)
        psi_r = psi.reshape(shape)
        # Apply the 2x2 rotation matrix [[c, -is], [-is, c]] on the middle axis
        # Using einsum for clarity and efficiency
        psi_r = (c * psi_r + (-1j * s) * np.swapaxes(psi_r, 1, 2).reshape(shape) * 0)  # This won't work directly
        # Let's do it manually but vectorized
        p0 = psi_r[:, 0, :].copy()
        p1 = psi_r[:, 1, :].copy()
        psi_r[:, 0, :] = c * p0 - 1j * s * p1
        psi_r[:, 1, :] = -1j * s * p0 + c * p1
        psi = psi_r.reshape(-1)
    return psi

def _expected_cut_4params(n, cuts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut for 2-layer QAOA with 4 independent parameters."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: Gamma then Beta
    phase1 = np.exp(-1j * gamma1 * cuts)
    psi = psi * phase1
    psi = _apply_beta(psi, beta1, n)
    
    # Layer 2: Gamma then Beta
    phase2 = np.exp(-1j * gamma2 * cuts)
    psi = psi * phase2
    psi = _apply_beta(psi, beta2, n)
    
    # Expected cut
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """Optimize 4 QAOA parameters using structured multi-start Nelder-Mead with polish."""
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([np.pi, 2*np.pi, np.pi, 2*np.pi])
    
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        return -_expected_cut_4params(n, cuts, g1, b1, g2, b2)
    
    # Generate structured initial points based on QAOA heuristics
    # Known good starting points from literature and empirical tuning
    structured_starts = [
        np.array([0.5, 0.5, 0.5, 0.5]),          # Default
        np.array([0.3, 0.4, 0.3, 0.4]),          # Smaller angles
        np.array([0.7, 0.6, 0.7, 0.6]),          # Larger angles  
        np.array([0.4, 0.3, 0.5, 0.4]),          # Asymmetric
        np.array([0.6, 0.5, 0.4, 0.6]),          # Asymmetric reversed
        np.array([0.2, 0.5, 0.8, 0.3]),          # Wide spread
        np.array([0.8, 0.3, 0.2, 0.7]),          # Wide spread reversed
        np.array([0.5, 0.7, 0.5, 0.3]),          # Beta emphasis layer 1
        np.array([0.5, 0.3, 0.5, 0.7]),          # Beta emphasis layer 2
        np.array([0.3, 0.7, 0.7, 0.3]),          # Cross pattern
    ]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        structured_starts.append(rng.uniform(lower, upper))
    
    best_x = structured_starts[0].copy()
    best_val = obj(best_x)
    
    # Sort by initial objective to prioritize promising starts
    start_objs = [(obj(s), s) for s in structured_starts]
    start_objs.sort(key=lambda x: x[0])
    
    # Optimize from each start, prioritizing better initial points
    for i, (init_obj, x0) in enumerate(start_objs):
        if _remaining() < 5.0:
            break
        # If this start is already worse than best found, skip unless it's very early
        if i > 3 and init_obj > best_val + 1e-3:
            continue
            
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-7,
                                    'adaptive': True, 'initial_simplex': 0.1})
            
            if res.fun < 1e9 and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if _remaining() > 15.0 and best_val < 1e9:
        try:
            res_polish = minimize(obj, best_x, method='Nelder-Mead', 
                                  options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9,
                                           'adaptive': True})
            if res_polish.fun < 1e9 and res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
            
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.35, 0.25], [0.35, 0.25]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Optimize 4 parameters for p=2
    opt = _optimize_4params(n, cuts)
    
    # Clip to valid ranges to avoid numerical issues
    opt[0] = np.clip(opt[0], 0, np.pi)
    opt[1] = np.clip(opt[1], 0, 2*np.pi)
    opt[2] = np.clip(opt[2], 0, np.pi)
    opt[3] = np.clip(opt[3], 0, 2*np.pi)
    
    # Return as (p, 2) array: rows are (gamma, beta)
    result = np.array([
        [opt[0], opt[1]],  # gamma_1, beta_1
        [opt[2], opt[3]]   # gamma_2, beta_2
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
