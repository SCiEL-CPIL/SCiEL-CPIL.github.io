"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 15.0
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

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA.
    Optimized state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1
    psi = psi * np.exp(-1j * gamma1 * cuts)
    if abs(beta1) > 1e-9:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)
            
    # Layer 2
    psi = psi * np.exp(-1j * gamma2 * cuts)
    if abs(beta2) > 1e-9:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using physics-informed initialization and intensive local search.
    """
    
    # Define objective for minimization (negative expected cut)
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val

    # Bounds
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    
    # Physics-informed initializations + coarse grid
    # Literature-based starts
    inits = [
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.4, 0.6, 0.4],
        [1.0, 0.3, 0.5, 0.5],
        [0.3, 0.6, 0.7, 0.3],
        [0.5, 0.2, 0.8, 0.6],
        [np.pi/4, np.pi/4, np.pi/4, np.pi/4],
    ]
    
    # Coarse grid: 6 points per dimension for 4 params = 1296 points
    g_vals = np.linspace(0.1, np.pi - 0.1, 6)
    b_vals = np.linspace(0.05, np.pi/2 - 0.05, 6)
    
    for g1 in g_vals:
        for b1 in b_vals:
            for g2 in g_vals:
                for b2 in b_vals:
                    inits.append([g1, b1, g2, b2])
    
    best_x = inits[0]
    best_val = obj(best_x)
    
    # Multi-start local search with Nelder-Mead
    for i, init in enumerate(inits):
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                obj, 
                init, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    # Final high-precision polish with L-BFGS-B (gradient-based)
    if _remaining() > 10.0:
        try:
            res_polish = minimize(
                obj, 
                best_x, 
                method='L-BFGS-B', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x
        except Exception:
            pass
        
    # Another NM polish with very tight tolerance
    if _remaining() > 5.0:
        try:
            res_final = minimize(
                obj, 
                best_x, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x
        except Exception:
            pass
        
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
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
