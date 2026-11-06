"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 145.0
_SAFETY = 5.0
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
    
    # Layer 1: Cost unitary
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: Mixer unitary
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
            
    # Layer 2: Cost unitary
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: Mixer unitary
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

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Objective function for minimizers (we want to maximize expected cut)
    def obj(x):
        if _remaining() < 0.5:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    bounds = list(zip(lower, upper))
    
    best_x = [0.5, 0.5, 0.5, 0.5]
    best_val = _expected_cut(n, cuts, *best_x)
    
    # Use Differential Evolution for global search
    # popsize=20, maxiter=50 is a reasonable start for 4D
    # We adjust maxiter based on remaining time
    time_per_eval_est = 0.01 # Estimate ~10ms per eval for n=10
    remaining = _remaining()
    max_iters = max(10, int(remaining / (20 * 4 * time_per_eval_est)))
    
    try:
        # DE with polish=True will run L-BFGS-B on the best candidate at the end
        # We use seed for reproducibility if needed, but random is fine
        res = differential_evolution(
            obj, 
            bounds, 
            seed=42, 
            maxiter=max_iters, 
            popsize=20, 
            tol=1e-8, 
            polish=True, 
            init='sobol'
        )
        
        if res.fun < 0: # Valid result
            candidate_x = res.x
            candidate_val = -res.fun
            if candidate_val > best_val:
                best_val = candidate_val
                best_x = candidate_x
    except Exception:
        pass
    
    # Final high-precision polish on the best found
    if _remaining() > 2.0:
        try:
            res2 = minimize(
                obj,
                np.array(best_x),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res2.fun < 0:
                candidate_val = -res2.fun
                if candidate_val > best_val:
                    best_val = candidate_val
                    best_x = res2.x
        except Exception:
            pass

    result = np.array([
        [best_x[0], best_x[1]],
        [best_x[2], best_x[3]]
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
