"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Global deadline guard
_T0 = time.time()
_DEADLINE = 145.0  # Leave 5s margin from 150s limit

def _remaining():
    return _DEADLINE - 5.0 - (time.time() - _T0)

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
    # Initial state: |+>^n
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: exp(-i gamma1 C)
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: exp(-i beta1 sum X)
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
            
    # Layer 2: exp(-i gamma2 C)
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: exp(-i beta2 sum X)
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
    # Check deadline
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Define objective function for optimizers
    def obj(x):
        g1, b1, g2, b2 = x
        # Clamp to valid ranges to ensure smoothness for NM
        g1 = np.clip(g1, 0.0, np.pi)
        b1 = np.clip(b1, 0.0, np.pi/2)
        g2 = np.clip(g2, 0.0, np.pi)
        b2 = np.clip(b2, 0.0, np.pi/2)
        return -_expected_cut(n, cuts, g1, b1, g2, b2)
    
    bounds = [(0.0, np.pi), (0.0, np.pi/2), (0.0, np.pi), (0.0, np.pi/2)]
    
    best_params = np.array([0.5, 0.5, 0.5, 0.5])
    best_val = np.inf
    
    # Phase 1: Coarse Grid Search to identify promising regions
    # This provides a good initial guess for DE and NM
    start_time = time.time()
    call_deadline = start_time + 10.0
    
    g_vals = np.linspace(0.0, np.pi, 10)
    b_vals = np.linspace(0.0, np.pi/2, 10)
    
    top_candidates = []
    
    for g1 in g_vals:
        if time.time() > call_deadline:
            break
        for b1 in b_vals:
            if time.time() > call_deadline:
                break
            for g2 in g_vals:
                if time.time() > call_deadline:
                    break
                for b2 in b_vals:
                    val = _expected_cut(n, cuts, g1, b1, g2, b2)
                    top_candidates.append((-val, np.array([g1, b1, g2, b2])))
    
    # Sort by value (ascending because we stored -val)
    top_candidates.sort(key=lambda x: x[0])
    
    # Take top few candidates for local refinement
    num_starts = min(5, len(top_candidates))
    if num_starts > 0:
        # Use the best grid point as initial best
        best_val = -top_candidates[0][0]
        best_params = top_candidates[0][1]
    
    # Phase 2: Differential Evolution for global search
    # DE is robust to local minima and can explore the landscape well
    if _remaining() > 20.0:
        try:
            de_result = differential_evolution(
                obj, 
                bounds, 
                maxiter=30, 
                popsize=15,
                seed=42,
                polish=False, # We will polish manually
                tol=1e-4,
                init='sobol'
            )
            de_val = -de_result.fun
            if de_val < best_val:
                best_val = de_val
                best_params = de_result.x
        except Exception:
            pass
    
    # Phase 3: Local Refinement from multiple starts
    # Start from DE result and top grid candidates
    start_points = [best_params]
    for i in range(num_starts):
        if i == 0:
            continue # Already added best_params which might be grid or DE
        start_points.append(top_candidates[i][1])
    
    # Add some random perturbations around the best point
    for delta in [0.1, -0.1, 0.05, -0.05]:
        for j in range(4):
            p_new = best_params.copy()
            p_new[j] = np.clip(p_new[j] + delta, bounds[j][0], bounds[j][1])
            start_points.append(p_new)
    
    for x0 in start_points:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            val = -res.fun
            if val < best_val:
                best_val = val
                best_params = res.x
        except Exception:
            continue
    
    # Phase 4: Final High-Precision Polish
    if _remaining() > 2.0:
        try:
            final_res = minimize(
                obj, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            final_val = -final_res.fun
            if final_val < best_val:
                best_params = final_res.x
        except Exception:
            pass
    
    # Ensure parameters are within bounds
    for i in range(4):
        best_params[i] = np.clip(best_params[i], bounds[i][0], bounds[i][1])
        
    return np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ], dtype=np.float64)

# EVOLVE-BLOCK-END
