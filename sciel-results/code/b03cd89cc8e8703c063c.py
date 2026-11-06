"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 12.0
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

def _apply_mixer(psi, n, beta):
    """Apply exp(-i*beta*sum X) using vectorized bit-flip operations."""
    if abs(beta) < 1e-14:
        return psi
    c = np.cos(beta)
    s = np.sin(beta)
    result = c * psi
    for k in range(n):
        # Flip bit k: swap positions where bit k is 0 vs 1
        stride = 2**k
        block = 2 * stride
        # Reshape to isolate bit k dimension
        shape = [1] * (n + 1)
        shape[n - k] = 2  # bit k position
        # Actually, let's use a simpler approach:
        # For bit k, the pattern repeats every 2^(k+1) elements
        # First 2^k elements have bit k = 0, next 2^k have bit k = 1
        # We need to swap them
        idx = np.arange(2**n)
        flipped = idx ^ (1 << k)
        result[flipped] -= 1j * s * psi[idx]
        # Wait, this isn't right. Let me use the proper X gate formula.
        # X|0> = |1>, X|1> = |0>
        # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
        # So: psi_new[bit_k=0] = c*psi[bit_k=0] - i*s*psi[bit_k=1]
        #     psi_new[bit_k=1] = -i*s*psi[bit_k=0] + c*psi[bit_k=1]
        # But we're applying to ALL qubits simultaneously (tensor product of X rotations)
        # This is NOT the same as applying X to each qubit sequentially in a simple way
        # because exp(-i*beta*(X1+X2+...+Xn)) = tensor product of exp(-i*beta*X) for each qubit
        # So we CAN apply them sequentially.
        pass
    
    # Use sequential application but vectorized per-qubit
    psi_out = psi.copy()
    for k in range(n):
        # Create index arrays for bit k = 0 and bit k = 1
        idx_0 = np.array([i for i in range(2**n) if (i >> k) & 1 == 0])
        idx_1 = idx_0 ^ (1 << k)
        
        p0 = psi_out[idx_0].copy()
        p1 = psi_out[idx_1].copy()
        
        psi_out[idx_0] = c * p0 - 1j * s * p1
        psi_out[idx_1] = -1j * s * p0 + c * p1
    
    return psi_out

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA via state-vector simulation.
    Uses vectorized mixer operations for speed.
    """
    num_states = 2**n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Precompute flip indices for each qubit (do this once per n)
    # We'll cache these
    if not hasattr(_expected_cut, '_flip_cache'):
        _expected_cut._flip_cache = {}
    if n not in _expected_cut._flip_cache:
        flip_indices = []
        for k in range(n):
            idx_0 = np.array([i for i in range(num_states) if (i >> k) & 1 == 0], dtype=np.int64)
            idx_1 = idx_0 ^ (1 << k)
            flip_indices.append((idx_0, idx_1))
        _expected_cut._flip_cache[n] = flip_indices
    flip_indices = _expected_cut._flip_cache[n]
    
    # Layer 1: cost Hamiltonian
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: mixer
    if abs(beta1) > 1e-12:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for idx_0, idx_1 in flip_indices:
            p0 = psi[idx_0].copy()
            p1 = psi[idx_1].copy()
            psi[idx_0] = c * p0 - 1j * s * p1
            psi[idx_1] = -1j * s * p0 + c * p1
    
    # Layer 2: cost Hamiltonian
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: mixer
    if abs(beta2) > 1e-12:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for idx_0, idx_1 in flip_indices:
            p0 = psi[idx_0].copy()
            p1 = psi[idx_1].copy()
            psi[idx_0] = c * p0 - 1j * s * p1
            psi[idx_1] = -1j * s * p0 + c * p1
    
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using physics-informed initialization
    and intensive L-BFGS-B local optimization with multiple restarts.
    """
    
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        # Wrap parameters to valid ranges
        g1 = g1 % (2 * np.pi)
        b1 = b1 % (2 * np.pi)
        g2 = g2 % (2 * np.pi)
        b2 = b2 % (2 * np.pi)
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    # Physics-informed starting points based on QAOA literature for MaxCut
    # For p=2, typical optimal ranges:
    # gamma1 ~ 0.3-0.8, beta1 ~ 0.2-0.6
    # gamma2 ~ 0.2-0.6, beta2 ~ 0.1-0.5
    inits = [
        [0.55, 0.40, 0.35, 0.25],
        [0.60, 0.35, 0.40, 0.20],
        [0.50, 0.45, 0.30, 0.30],
        [0.65, 0.30, 0.45, 0.15],
        [0.45, 0.50, 0.25, 0.35],
        [0.70, 0.25, 0.50, 0.10],
        [0.40, 0.55, 0.20, 0.40],
        [0.62, 0.38, 0.38, 0.22],
        [0.48, 0.48, 0.32, 0.28],
        [0.52, 0.42, 0.36, 0.26],
        [0.58, 0.36, 0.42, 0.18],
        [0.44, 0.52, 0.28, 0.32],
        [0.68, 0.28, 0.48, 0.12],
        [0.38, 0.58, 0.22, 0.38],
        [0.55, 0.38, 0.37, 0.23],
    ]
    
    best_x = None
    best_val = np.inf
    
    bounds = [(0.0, np.pi), (0.0, np.pi/2), (0.0, np.pi), (0.0, np.pi/2)]
    
    # Multi-start L-BFGS-B
    for i, init in enumerate(inits):
        if _remaining() < 4.0:
            break
        try:
            res = minimize(
                obj,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11, 'maxfun': 1500}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # If we haven't found anything, use a default
    if best_x is None:
        best_x = np.array([0.55, 0.40, 0.35, 0.25])
        best_val = obj(best_x)
    
    # Final high-precision polish
    if _remaining() > 5.0:
        try:
            res_polish = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxfun': 5000}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    # One more ultra-tight polish if time permits
    if _remaining() > 8.0:
        try:
            res_ultra = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 10000}
            )
            if res_ultra.fun < best_val:
                best_val = res_ultra.fun
                best_x = res_ultra.x.copy()
        except Exception:
            pass
    
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.55, 0.40], [0.35, 0.25]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Optimize 4 parameters for p=2
    opt = _optimize_4params(n, cuts)
    
    # Return as (p, 2) array: rows are (gamma, beta)
    result = np.array([
        [opt[0], opt[1]],  # gamma_1, beta_1
        [opt[2], opt[3]]   # gamma_2, beta_2
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
