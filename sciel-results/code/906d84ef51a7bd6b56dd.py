"""Shallow-circuit Ising ground states - optimized simulation and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_INDEX_CACHE = {}
_HAMILTONIAN_CACHE = {}

def _get_index_masks(n):
    """Pre-compute indices for qubit rotations."""
    if n in _INDEX_CACHE:
        return _INDEX_CACHE[n]
    
    dim = 1 << n
    indices = np.arange(dim)
    masks = []
    for i in range(n):
        bit = 1 << i
        idx0 = indices[(indices & bit) == 0]
        idx1 = indices[(indices & bit) != 0]
        masks.append((idx0, idx1))
    _INDEX_CACHE[n] = masks
    return masks

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    key = (n, coupling, field)
    if key in _HAMILTONIAN_CACHE:
        return _HAMILTONIAN_CACHE[key]
        
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        bit_i = 1 << i
        bit_ip1 = 1 << (i + 1)
        for state_idx in range(dim):
            # Z eigenvalue is +1 if bit is 0, -1 if bit is 1
            z_i = 1.0 if (state_idx & bit_i) == 0 else -1.0
            z_ip1 = 1.0 if (state_idx & bit_ip1) == 0 else -1.0
            H[state_idx, state_idx] -= coupling * z_i * z_ip1
            
    # X_i terms
    for i in range(n):
        bit_i = 1 << i
        for state_idx in range(dim):
            partner = state_idx ^ bit_i
            # X maps |0> to |1> and |1> to |0>
            # H_{state, partner} += -field
            # Since X is symmetric, H is symmetric
            H[state_idx, partner] -= field
            
    _HAMILTONIAN_CACHE[key] = H
    return H

def _circuit_state(theta, n, layers, masks):
    """Compute the statevector of the circuit using pre-computed masks."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0  # |0...0>
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = masks[i]
            
            # We must copy because the slices overlap in memory logic 
            # (though not in indices, the update depends on old values)
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
            
        if l < layers - 1:
            # Apply CZ on adjacent pairs (i, i+1)
            # CZ flips sign if both bits are 1
            for i in range(n - 1):
                bit_i = 1 << i
                bit_ip1 = 1 << (i + 1)
                # Find indices where both bits are set
                # We can iterate or use a pre-computed mask
                # For small n, simple loop over states is fast enough, 
                # but let's try to be vectorized if possible.
                # Actually, for n=6, dim=64, a loop is fine.
                # But we can pre-compute CZ masks too.
                pass
                
            # Let's pre-compute CZ sign flips for efficiency
            # We will handle CZ inside the loop using a pre-computed sign array
            # Or just apply it here.
            # To keep it clean, let's pre-compute the CZ operation as a diagonal matrix or sign vector.
            
    return state

# Let's refactor _circuit_state to use pre-computed CZ signs
def _get_cz_signs(n):
    if n in _INDEX_CACHE and 'cz' in _INDEX_CACHE:
        return _INDEX_CACHE['cz']
    dim = 1 << n
    signs = np.ones(dim, dtype=float)
    for i in range(n - 1):
        bit_i = 1 << i
        bit_ip1 = 1 << (i + 1)
        for idx in range(dim):
            if (idx & bit_i) != 0 and (idx & bit_ip1) != 0:
                signs[idx] *= -1.0
    _INDEX_CACHE['cz'] = signs
    return signs

def _circuit_state_optimized(theta, n, layers, masks, cz_signs):
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = masks[i]
            
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
            
        if l < layers - 1:
            state *= cz_signs
            
    return state

def _mean_field_init(n, coupling, field):
    """
    Initialize angles based on mean-field theory intuition.
    """
    theta = np.zeros((3, n))
    
    ratio = field / coupling
    # Transition is around ratio=1. 
    # If ratio is small (strong coupling), we want Z-aligned, so theta ~ 0.
    # If ratio is large (strong field), we want X-aligned, so theta ~ pi/2.
    
    # Smooth interpolation
    t_target = np.arctan(ratio) * 2.0 # Maps (0, inf) to (0, pi/2)
    # Clamp to reasonable range
    t_target = np.clip(t_target, 0, np.pi/2)
    
    # Distribute rotation across layers
    # Usually, putting more rotation in the middle/end helps with entanglement generation
    theta[0] = t_target * 0.2
    theta[1] = t_target * 0.5
    theta[2] = t_target * 0.3
    
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    H = _build_hamiltonian(n, coupling, field)
    masks = _get_index_masks(n)
    cz_signs = _get_cz_signs(n)
    
    def objective(x):
        theta = x.reshape((layers, n))
        state = _circuit_state_optimized(theta, n, layers, masks, cz_signs)
        # Energy = <psi|H|psi>
        # H is symmetric, state is real
        return float(state @ (H @ state))
    
    # Initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Mean-field init
    candidates.append(_mean_field_init(n, coupling, field))
    
    # 2. Pure X-aligned (strong field limit)
    theta_x = np.full((layers, n), np.pi / 2)
    candidates.append(theta_x)
    
    # 3. Pure Z-aligned (strong coupling limit)
    theta_z = np.zeros((layers, n))
    candidates.append(theta_z)
    
    # 4. Random inits
    for _ in range(50):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(layers, n)))
        
    # 5. Structured inits: alternating
    for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        theta_alt = np.zeros((layers, n))
        for i in range(n):
            theta_alt[:, i] = phase + (i % 2) * np.pi/2
        candidates.append(theta_alt)

    best_theta = None
    best_energy = float('inf')
    
    # Phase 1: Quick local optimization from all candidates
    for tc in candidates:
        if time.time() > deadline - 20:
            break
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            # If optimization fails, try the raw candidate
            e = objective(tc.flatten())
            if e < best_energy:
                best_energy = e
                best_theta = tc.copy()
                
    # Phase 2: More aggressive random restarts around the best
    if best_theta is not None and time.time() < deadline - 30:
        restart_budget = 20
        for r in range(restart_budget):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution
            tc = best_theta + rng.normal(0, 0.5, size=(layers, n))
            try:
                res = opt.minimize(
                    objective,
                    tc.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_theta = res.x.reshape((layers, n))
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if best_theta is not None and time.time() < deadline - 5:
        try:
            # Use Nelder-Mead for final polish as it can handle non-smoothness better
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
        
        # One more L-BFGS-B pass for extreme precision
        if time.time() < deadline - 2:
            try:
                res = opt.minimize(
                    objective,
                    best_theta.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_theta = res.x.reshape((layers, n))
            except Exception:
                pass

    if best_theta is None:
        best_theta = np.zeros((layers, n))
        
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((layers, n))
        
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
