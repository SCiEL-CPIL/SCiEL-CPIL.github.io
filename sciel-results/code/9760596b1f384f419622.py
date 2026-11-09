"""Optimize shallow circuit parameters for Ising ground states using analytic gradient descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the transverse-field Ising model."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for k in range(dim):
            z_i = 1 if (k >> i) & 1 == 0 else -1
            z_j = 1 if (k >> (i + 1)) & 1 == 0 else -1
            H[k, k] += -coupling * z_i * z_j
            
    # X_i terms
    for i in range(n):
        for k in range(dim):
            k_flip = k ^ (1 << i)
            H[k, k_flip] += -field / 2.0
            H[k_flip, k] += -field / 2.0
            
    return H

def _get_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])

def _energy_and_grad(params, n, layers, H, init_state):
    """
    Compute energy and analytic gradient for the circuit.
    params: (layers, n) array of Ry angles.
    Returns: energy, grad (layers, n)
    """
    dim = 1 << n
    state = init_state.copy()
    grad = np.zeros((layers, n), dtype=float)
    
    # Precompute masks for CZ
    cz_masks = []
    for i in range(n - 1):
        mask = (1 << i) | (1 << (i + 1))
        indices = np.arange(dim)
        # Indices where both bits are set
        inds = indices[(indices & mask) == mask]
        cz_masks.append(inds)
        
    shape = [2] * n
    
    # Forward pass to get final state and store intermediate states for backprop
    # Storing state after each layer is sufficient since CZ is diagonal and Ry is local
    states = [init_state]
    
    for l in range(layers):
        # Apply Ry
        for i in range(n):
            theta = params[l, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            state_tensor = state.reshape(shape)
            state_tensor = np.moveaxis(state_tensor, i, 0)
            
            s0 = state_tensor[0].copy()
            s1 = state_tensor[1].copy()
            
            new_s0 = c * s0 + s * s1
            new_s1 = -s * s0 + c * s1
            
            state_tensor[0] = new_s0
            state_tensor[1] = new_s1
            
            state_tensor = np.moveaxis(state_tensor, 0, i)
            state = state_tensor.reshape(-1)
            
        # Apply CZ if not last layer
        if l < layers - 1:
            for inds in cz_masks:
                state[inds] *= -1.0
                
        states.append(state.copy())
        
    # Final state is states[-1]
    final_state = states[-1]
    
    # Energy
    Hs = H.dot(final_state)
    energy = np.real(np.vdot(final_state, Hs))
    
    # Backpropagation for gradients
    # dE/dtheta = 2 * Re( <psi_final| dU/dtheta | psi_initial> ) ... wait, 
    # E = <psi|H|psi>. dE = <dpsi|H|psi> + <psi|H|dpsi> = 2 Re(<dpsi|H|psi>)
    # Let A = H |psi_final>. We need Re(<dpsi|A>)
    
    A = H.dot(final_state)
    
    # We traverse layers in reverse
    # Let |psi_l> be state after layer l.
    # |psi_{l+1}> = U_l |psi_l>
    # We have adjoint states |chi_l> such that dE = 2 Re( sum <chi_l| dU_l |psi_l> )
    # Actually, standard BPTD:
    # |chi_final> = H |psi_final> (if we consider E = <psi|H|psi>, gradient wrt U is 2 Re <psi|H|dpsi>)
    # Let's define |chi_k> recursively.
    # dE = 2 Re( <psi_final| H |dpsi_final> )
    # Let |chi_final> = H |psi_final>. Then dE = 2 Re( <chi_final| dpsi_final> )? No, <psi|H|dpsi> is not <chi|dpsi> unless H is real/symmetric and we handle conjugates carefully.
    # Since H is Hermitian, <psi|H|dpsi> + c.c. = 2 Re(<psi|H|dpsi>).
    # Let |v> = H |psi_final>. Then <psi|H|dpsi> = <psi|v>|dpsi>... no.
    # <psi|H|dpsi> = (H^dag |psi>)^dag |dpsi> = (H |psi>)^dag |dpsi> = <H|psi> |dpsi> = <v|dpsi>.
    # So dE = 2 Re( <v|dpsi_final> ).
    
    v = H.dot(final_state)
    
    # Propagate v backwards
    # dpsi_{l+1} = U_l dpsi_l
    # <v_{l+1}| dpsi_{l+1}> = <v_{l+1}| U_l |dpsi_l> = <U_l^dag v_{l+1}| dpsi_l>
    # So v_l = U_l^dag v_{l+1}
    
    for l in range(layers - 1, -1, -1):
        v_current = v.copy()
        
        # Undo CZ
        if l < layers - 1:
            for inds in cz_masks:
                v_current[inds] *= -1.0
                
        # Undo Ry and compute gradient
        for i in range(n):
            theta = params[l, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            v_tensor = v_current.reshape(shape)
            v_tensor = np.moveaxis(v_tensor, i, 0)
            
            v0 = v_tensor[0].copy()
            v1 = v_tensor[1].copy()
            
            # U_Ry = [[c, s], [-s, c]]
            # U_Ry^dag = [[c, -s], [s, c]]
            # v_prev = U^dag v_current
            # v_prev[0] = c*v0 - s*v1
            # v_prev[1] = s*v0 + c*v1
            
            # Gradient contribution:
            # dU/dtheta = [[-s/2, c/2], [-c/2, -s/2]]
            # dE_contribution = 2 Re( <v_current| dU |psi_l> )
            # Let |psi_l> be the state BEFORE this Ry.
            # We have |psi_{l+1}> (after CZ if applicable, but we undid CZ above so v_current corresponds to state after Ry).
            # Wait, v_current is the adjoint state corresponding to the state AFTER the Ry gate in layer l.
            # Let |psi_after> be the state after Ry.
            # We need |psi_before>.
            # |psi_after> = U |psi_before> => |psi_before> = U^dag |psi_after>.
            # We have |psi_after> stored in states[l+1] (if no CZ) or we need to recover it.
            # Actually, we stored states after each layer (including CZ).
            # For layer l:
            # state_before_ry = states[l]
            # state_after_ry = state_after_ry_temp (not stored directly, but states[l+1] is after CZ)
            # If l < layers-1, states[l+1] = CZ * state_after_ry.
            # So state_after_ry = CZ * states[l+1] (since CZ is its own inverse).
            
            if l < layers - 1:
                state_after_ry = states[l+1].copy()
                for inds in cz_masks:
                    state_after_ry[inds] *= -1.0
            else:
                state_after_ry = states[l+1]
                
            # Apply U^dag to get state_before_ry
            psi_tensor = state_after_ry.reshape(shape)
            psi_tensor = np.moveaxis(psi_tensor, i, 0)
            
            p0 = psi_tensor[0].copy()
            p1 = psi_tensor[1].copy()
            
            # U^dag = [[c, -s], [s, c]]
            # p_before_0 = c*p0 - s*p1
            # p_before_1 = s*p0 + c*p1
            
            # We need <v_current| dU |psi_before>
            # dU = [[-s/2, c/2], [-c/2, -s/2]]
            # dU |psi_before> = [ -s/2 * p_b0 + c/2 * p_b1, -c/2 * p_b0 - s/2 * p_b1 ]
            
            # We can compute p_b0, p_b1
            p_b0 = c * p0 - s * p1
            p_b1 = s * p0 + c * p1
            
            # dpsi_0 = -0.5*s*p_b0 + 0.5*c*p_b1
            # dpsi_1 = -0.5*c*p_b0 - 0.5*s*p_b1
            
            # inner = v0 * conj(dpsi_0) + v1 * conj(dpsi_1)
            # Note: v is adjoint, so we use v directly (not conj) if we are doing <v|dpsi>
            # <v|dpsi> = v0 * conj(dpsi_0) + v1 * conj(dpsi_1)
            
            dpsi_0 = -0.5 * s * p_b0 + 0.5 * c * p_b1
            dpsi_1 = -0.5 * c * p_b0 - 0.5 * s * p_b1
            
            inner = np.vdot(v0, dpsi_0) + np.vdot(v1, dpsi_1)
            
            # grad = 2 * Re(inner)
            grad[l, i] = 2.0 * np.real(inner)
            
            # Update v for backprop
            # v_prev = U^dag v_current
            new_v0 = c * v0 - s * v1
            new_v1 = s * v0 + c * v1
            
            v_tensor[0] = new_v0
            v_tensor[1] = new_v1
            
            v_tensor = np.moveaxis(v_tensor, 0, i)
            v_current = v_tensor.reshape(-1)
            
        # After undoing Ry for all qubits, v_current is the adjoint state for the start of layer l
        # Which corresponds to states[l]
        v = v_current
        
    return energy, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    dim = 1 << n
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _get_ground_energy(n, coupling, field)
    
    # Initial state |0>^n
    init_state = np.zeros(dim, dtype=complex)
    init_state[0] = 1.0
    
    # Adam Optimizer
    def adam_step(params, grads, lr, beta1, beta2, eps, m, v, t):
        m_t = beta1 * m + (1 - beta1) * grads
        v_t = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)
        params_new = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        return params_new, m_t, v_t, t + 1
    
    best_params = None
    best_energy = np.inf
    
    # Multiple restarts to avoid local minima
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize parameters
        # Physics-based init: try to align with ground state expectations
        # For small field, Z alignment is good. For large field, X alignment.
        # Let's use random init with different seeds for diversity
        rng = np.random.default_rng(42 + r)
        params = rng.uniform(-np.pi, np.pi, size=(layers, n))
        
        # Adam hyperparameters
        lr = 0.1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
        
        # Max steps per restart
        max_steps = 2000
        steps = 0
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            energy, grad = _energy_and_grad(params, n, layers, H, init_state)
            
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
                
            # Adaptive learning rate decay
            current_lr = lr / (1 + 0.01 * steps)
            
            params, m, v, t = adam_step(params, grad, current_lr, beta1, beta2, eps, m, v, t)
            steps += 1
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-8:
                break
                
    # If best_params is None, fallback to random
    if best_params is None:
        best_params = np.zeros((layers, n))
        
    return best_params
# EVOLVE-BLOCK-END
