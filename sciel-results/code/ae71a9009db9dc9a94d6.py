"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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
        cuts[mask] += 1.0
    return cuts

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA.
    Optimized state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: cost unitary
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
            
    # Layer 2: cost unitary
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

def _fidelity_to_optimum(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute fidelity between QAOA output state and the uniform superposition
    of all optimal cut states. This is a smoother objective than expected cut.
    """
    # Find optimal cut value
    max_cut = np.max(cuts)
    opt_mask = cuts == max_cut
    num_opt = np.sum(opt_mask)
    
    # Target state: uniform superposition over optimal cuts
    target = np.zeros(2**n, dtype=np.complex128)
    target[opt_mask] = 1.0 / np.sqrt(num_opt)
    
    # Compute QAOA state
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
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

    # Fidelity: |<target|psi>|^2
    fidelity = np.abs(np.vdot(target, psi))**2
    return fidelity

def _hybrid_objective(n, cuts, gamma1, beta1, gamma2, beta2, fidelity_weight=0.3):
    """
    Hybrid objective: combination of expected cut and fidelity to optimum.
    This may provide a smoother landscape for optimization.
    """
    exp_cut = _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2)
    max_cut = np.max(cuts)
    
    # Normalize expected cut to [0,1]
    norm_exp_cut = exp_cut / max_cut if max_cut > 0 else 0.0
    
    # Fidelity is already in [0,1]
    fidelity = _fidelity_to_optimum(n, cuts, gamma1, beta1, gamma2, beta2)
    
    # Combined objective (maximize)
    return fidelity_weight * fidelity + (1 - fidelity_weight) * norm_exp_cut

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using differential evolution + local polish.
    Uses a hybrid objective combining expected cut and fidelity.
    """
    
    # Define objective for minimization (negative combined value)
    def obj(x):
        if _remaining() < 1.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _hybrid_objective(n, cuts, g1, b1, g2, b2, fidelity_weight=0.3)
        return -val
    
    # Bounds
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    
    best_x = None
    best_val = np.inf
    
    # Phase 1: Differential evolution for global exploration
    if _remaining() > 30.0:
        try:
            de_res = differential_evolution(
                obj, 
                bounds=list(zip(lower, upper)),
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                init='sobol',
                polish=False,
                mutation=(0.3, 1.2),
                recombination=0.9
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x
        except Exception:
            pass
    
    # Phase 2: Multi-start local polish from multiple points
    if best_x is not None and _remaining() > 5.0:
        # Use the best from DE as one start
        start_points = [best_x.copy()]
        
        # Add some diverse starting points
        np.random.seed(123)
        for _ in range(8):
            sp = np.array([
                np.random.uniform(0.2, 1.2),
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.2, 1.2),
                np.random.uniform(0.1, 0.9)
            ])
            start_points.append(sp)
        
        # Also add some structured points
        start_points.append([0.5, 0.5, 0.5, 0.5])
        start_points.append([0.8, 0.4, 0.6, 0.4])
        start_points.append([1.0, 0.3, 0.5, 0.5])
        start_points.append([0.3, 0.6, 0.7, 0.3])
        
        for sp in start_points:
            if _remaining() < 3.0:
                break
            try:
                # Clip to bounds
                sp = np.clip(sp, lower, upper)
                
                res = minimize(
                    obj, 
                    sp, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lower, upper)),
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue
    
    # Phase 3: Final tight polish
    if best_x is not None and _remaining() > 2.0:
        try:
            res_polish = minimize(
                obj, 
                best_x, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    # Fallback if nothing found
    if best_x is None:
        best_x = np.array([0.5, 0.5, 0.5, 0.5])
    
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
