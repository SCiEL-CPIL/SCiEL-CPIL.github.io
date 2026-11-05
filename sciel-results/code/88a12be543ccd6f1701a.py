"""Shallow-circuit Ising ground states - optimized with analytic gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_CZ_MASKS = {}
_IDX_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _get_cz_masks(n):
    key = n
    if key not in _CZ_MASKS:
        dim = 1 << n
        base = np.arange(dim)
        masks = []
        for i in range(n - 1):
            mask_i = 1 << i
            mask_ip1 = 1 << (i + 1)
            both_set = ((base & mask_i) != 0) & ((base & mask_ip1) != 0)
            masks.append(both_set)
        _CZ_MASKS[key] = masks
    return _CZ_MASKS[key]

def _apply_cz_fast(state, n):
    """Apply CZ gates on adjacent pairs in ascending order - optimized."""
    masks = _get_cz_masks(n)
    for m in masks:
        state[m] *= -1.0
    return state

# Precompute index arrays for speed
_IDX_CACHE = {}

def _get_idx_arrays(n):
    key = n
    if key not in _IDX_CACHE:
        dim = 1 << n
        base = np.arange(dim)
        arrays = {}
        for i in range(n):
            mask = 1 << i
            idx0 = base[(base & mask) == 0]
            idx1 = idx0 | mask
            arrays[i] = (idx0, idx1)
        _IDX_CACHE[key] = arrays
    return _IDX_CACHE[key]

def _circuit_energy_fast(theta, n, layers, H):
    """Fast energy computation using precomputed index arrays."""
    dim = 1 << n
    idx_arrays = _get_idx_arrays(n)
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            _apply_cz_fast(state, n)
    
    return float(state @ (H @ state))

def _circuit_energy_grad(theta, n, layers, H):
    """Compute energy and analytic gradient w.r.t. theta.
    
    Uses the identity: d/dt [Ry(t) psi] = 0.5 * X_i Ry(t) psi
    and backpropagation through the circuit.
    """
    dim = 1 << n
    idx_arrays = _get_idx_arrays(n)
    
    # Forward pass: store states after each Ry gate
    # states[l, i] = state after applying Ry(theta[l,i]) on qubit i in layer l
    # We store the state BEFORE each Ry to compute gradients
    states_before = []  # states_before[l][i] = state before Ry(l,i)
    
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        layer_states = []
        for i in range(n):
            layer_states.append(state.copy())
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        states_before.append(layer_states)
        
        if l < layers - 1:
            _apply_cz_fast(state, n)
    
    # Final state
    psi_final = state
    Hpsi = H @ psi_final
    energy = float(psi_final @ Hpsi)
    
    # Backward pass
    # grad[l][i] = <psi_after_Ry(l,i) | d/dt Ry(t) | psi_before_Ry(l,i)> * 0.5
    # But we need to account for all subsequent gates.
    # 
    # Actually, let's use a cleaner approach:
    # Let |psi_k> be the state after the k-th elementary gate (Ry or CZ).
    # E = <psi_final|H|psi_final>
    # dE/dtheta[l,i] = 2 Re[ <psi_final| ... | H | psi_final> chain ]
    #
    # Simpler: use the fact that for a unitary circuit U(theta),
    # E = <0|U†HU|0>
    # dE/dtheta_j = 2 Re[<0|U†H dU/dtheta_j|0>]
    #
    # Let's do proper backprop:
    # adjoint state: start with |phi> = H|psi_final>
    # propagate backward through gates
    
    # adjoint: |phi> starts as H|psi_final>
    phi = Hpsi.copy()
    
    grad = np.zeros_like(theta)
    
    # Backprop through layers in reverse
    for l in range(layers - 1, -1, -1):
        # If l < layers-1, there was a CZ after this layer's Ry's
        # We need to undo the CZ
        if l < layers - 1:
            # CZ is self-inverse
            _apply_cz_fast(phi, n)
        
        # Backprop through Ry gates in reverse order (i = n-1 down to 0)
        for i in range(n - 1, -1, -1):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            
            # phi is currently the adjoint after Ry(l,i)
            # phi_before = Ry†(t) phi
            # dE/dt = 2 Re[<phi_before | X_i/2 | psi_before>]
            #       = Re[<phi_before | X_i | psi_before>]
            
            # Compute <phi_before | X_i | psi_before>
            # X_i |psi_before> = swap components: psi_before[idx1] -> idx0, psi_before[idx0] -> idx1
            psi_b = states_before[l][i]
            Xpsi0 = psi_b[idx1]  # X maps |1> to |0>
            Xpsi1 = psi_b[idx0]  # X maps |0> to |1>
            
            # phi_before = Ry†(t) phi
            # Ry†(t) = [[c, s], [-s, c]]
            phi0 = phi[idx0]
            phi1 = phi[idx1]
            phi_before0 = c * phi0 + s * phi1
            phi_before1 = -s * phi0 + c * phi1
            
            # <phi_before | X | psi_before> = phi_before0.conj() * Xpsi0 + phi_before1.conj() * Xpsi1
            # All real here
            grad_val = np.sum(phi_before0 * Xpsi0) + np.sum(phi_before1 * Xpsi1)
            grad[l, i] = grad_val
            
            # Update phi: phi = Ry†(t) phi (continue backprop)
            phi[idx0] = phi_before0
            phi[idx1] = phi_before1
    
    return energy, grad

def _mean_field_init(n, coupling, field):
    """Mean-field / variational product state initialization."""
    theta = np.zeros((3, n))
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(min(ratio / 2.0, 1.0))
    else:
        t_opt = np.pi / 2
    theta[0] = t_opt * 0.5
    theta[1] = t_opt * 0.3
    theta[2] = t_opt * 0.2
    return theta

def _physics_init(n, coupling, field):
    """Physics-inspired initialization based on mean-field theory."""
    ratio = field / coupling
    if ratio < 2.0:
        t_phys = np.arcsin(ratio / 2.0)
    else:
        t_phys = np.pi / 2
    
    # Final layer does most work, earlier layers provide entanglement
    theta = np.zeros((3, n))
    theta[2] = t_phys * 0.8
    theta[1] = t_phys * 0.15
    theta[0] = t_phys * 0.05
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    
    def objective(x):
        theta = x.reshape((3, n))
        return _circuit_energy_fast(theta, n, 3, H)
    
    def objective_grad(x):
        theta = x.reshape((3, n))
        e, g = _circuit_energy_grad(theta, n, 3, H)
        return e, g.flatten()
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(12345)
    candidates = []
    
    # Mean-field init
    theta_mf = _mean_field_init(n, coupling, field)
    candidates.append(theta_mf)
    
    # Physics-inspired init
    theta_ph = _physics_init(n, coupling, field)
    candidates.append(theta_ph)
    
    # Simple heuristic
    theta_simple = np.zeros((3, n))
    theta_simple[-1] = np.pi / 2
    candidates.append(theta_simple)
    
    # Uniform
    theta_uniform = np.full((3, n), np.pi / 4)
    candidates.append(theta_uniform)
    
    # Symmetric init based on physics
    ratio = field / coupling
    if ratio < 2.0:
        t_phys = np.arcsin(ratio / 2.0)
    else:
        t_phys = np.pi / 2
    theta_phys = np.full((3, n), t_phys / 3.0)
    candidates.append(theta_phys)
    
    # Physics-inspired: final layer does most of the work
    theta_final = np.zeros((3, n))
    theta_final[2] = t_phys
    theta_final[1] = t_phys * 0.3
    theta_final[0] = t_phys * 0.2
    candidates.append(theta_final)
    
    # Random starts
    for _ in range(40):
        candidates.append(rng.uniform(-np.pi, np.pi, size=(3, n)))
    
    # Evaluate all candidates
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline - 5:
            break
        e = objective(tc.flatten())
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # L-BFGS-B with analytic gradient - main workhorse
    if time.time() < deadline - 25:
        try:
            res = opt.minimize(
                objective_grad,
                best_theta.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 50}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Powell polish (derivative-free, good for final adjustments)
    if time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 5000, 'xtol': 1e-16, 'ftol': 1e-20}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Additional random restarts with gradient-based optimization
    restart_budget = 25
    for r in range(restart_budget):
        if time.time() > deadline - 10:
            break
        tc = best_theta + rng.normal(0, 0.6, size=(3, n))
        try:
            res = opt.minimize(
                objective_grad,
                tc.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 40}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final ultra-precision polish with gradient
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective_grad,
                best_theta.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-22, 'gtol': 1e-18, 'maxls': 60}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # One more Powell at the very end
    if time.time() < deadline - 3:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-17, 'ftol': 1e-21}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Ensure output is finite
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((3, n))
        best_theta[-1] = np.pi / 2
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
