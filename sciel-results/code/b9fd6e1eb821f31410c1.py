"""Quantum information: shallow-circuit Ising ground states.
Uses symmetry reduction and global grid search followed by local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_energy(H):
    """Compute exact ground state energy."""
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def _circuit_state(params, n, layers):
    """
    Apply the circuit to |0...0> and return the final state vector.
    params: array of shape (layers, n) with angles.
    Circuit: for each layer, apply Ry(theta) to all qubits, then CZ on adjacent pairs (except after last layer).
    """
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Apply Ry on qubit q
            for idx in range(dim):
                bit = (idx >> q) & 1
                if bit == 0:
                    # |0> -> c|0> + s|1>
                    idx1 = idx | (1 << q)
                    psi[idx1] += s * psi[idx]
                    psi[idx] *= c
                else:
                    # |1> -> -s|0> + c|1>
                    idx0 = idx & ~(1 << q)
                    psi[idx0] -= s * psi[idx]
                    psi[idx] *= c
        
        # Apply CZ on adjacent pairs (not after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                for idx in range(dim):
                    bit_q = (idx >> q) & 1
                    bit_q1 = (idx >> (q + 1)) & 1
                    if bit_q == 1 and bit_q1 == 1:
                        psi[idx] *= -1.0
    
    return psi


def _energy_from_state(psi, H):
    """Compute energy <psi|H|psi>."""
    return np.vdot(psi, H @ psi).real


def _circuit_forward_backward(params, n, layers):
    """
    Computes the final state and the gradients of the state w.r.t parameters.
    Uses the adjoint method (reverse-mode automatic differentiation) for efficiency.
    
    Returns:
        psi_final: final state vector
        grads: list of arrays, each is d|psi_final>/dtheta_k
    """
    dim = 2 ** n
    num_params = layers * n
    
    # Store all intermediate states
    # states[0] is |0...0>
    # states[k+1] is state after applying gate k
    # We have layers*n Ry gates and (layers-1)*(n-1) CZ gates.
    # But we only need gradients w.r.t Ry parameters.
    
    # Let's store state before each Ry gate and the state after all gates.
    # Actually, it's easier to store the state after every gate application.
    
    # Gate sequence:
    # L=0: Ry(0), Ry(1), ..., Ry(n-1), CZ(0), ..., CZ(n-2)
    # L=1: Ry(0), Ry(1), ..., Ry(n-1), CZ(0), ..., CZ(n-2)
    # L=2: Ry(0), Ry(1), ..., Ry(n-1)
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # We need to store the state *before* each Ry gate to compute the gradient contribution.
    # Let states_before_ry[k] be the state before applying Ry gate k.
    # Let angles[k] be the angle for Ry gate k.
    
    states_before_ry = []
    angles = []
    
    # Forward pass
    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(psi.copy())
            angles.append(params[layer * n + q])
            psi = _apply_ry(psi, q, angles[-1], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    psi_final = psi
    
    # Backward pass (Adjoint)
    # We want d|psi_final>/dtheta_k.
    # Let |psi_final> = U_after_k * Ry_k(theta_k) * |psi_before_k>
    # d|psi_final>/dtheta_k = U_after_k * (dRy_k/dtheta_k) * |psi_before_k>
    
    # Instead of applying U_after_k explicitly, we propagate the adjoint state backwards.
    # Let |chi> be the adjoint state, initialized as |psi_final>.
    # We propagate |chi> backwards through the gates.
    # When we pass through a Ry gate, we compute the gradient contribution.
    
    # The gradient of the energy E = <psi|H|psi> is 2 Re[ <H psi | dpsi/dtheta> ].
    # Let |h_psi> = H |psi_final>.
    # We can compute g_k = 2 Re[ <h_psi | dpsi/dtheta_k> ].
    
    # Let's propagate |chi> = |h_psi> backwards.
    # Initially |chi> = H @ psi_final.
    chi = np.vdot(H, psi_final).conjugate() # Wait, H is real symmetric. H @ psi is complex.
    # <H psi | v> = (H psi)^dagger v = psi^dagger H v.
    # We want <H psi | dpsi_k>.
    # Let's define the adjoint vector |a> such that <H psi | dpsi_k> = <a | something>.
    
    # Standard adjoint method for quantum circuits:
    # We want dE/dtheta_k.
    # E = <psi_final|H|psi_final>.
    # dE/dtheta_k = <psi_final|H| dpsi_k/dtheta_k> + <dpsi_k/dtheta_k|H|psi_final>
    #             = 2 Re[ <psi_final|H| dpsi_k/dtheta_k> ]
    
    # Let |phi> = H |psi_final>.
    # Then dE/dtheta_k = 2 Re[ <psi_final| |phi> ... no.
    # <psi_final| H | dpsi_k> = (H^dagger |psi_final>)^dagger |dpsi_k> = <phi | dpsi_k> where phi = H psi_final? No.
    # (H psi) is a vector. <psi| H |dpsi> = (H psi)^dagger dpsi? No.
    # <psi| H |dpsi> = sum_i psi_i^* (H dpsi)_i.
    # (H psi)^dagger dpsi = sum_i (H psi)_i^* dpsi_i.
    # Since H is Hermitian, <psi| H |dpsi> = <H psi | dpsi>.
    # So we need <H psi_final | dpsi_k/dtheta_k>.
    
    # Let's propagate the vector |v> = H @ psi_final backwards.
    # Actually, it's easier to propagate the bra <psi_final| H = <psi_final| H.
    # Let's stick to the state vector propagation.
    
    # We have the chain:
    # |psi_final> = G_L ... G_1 |psi_0>
    # d|psi_final>/dtheta_k = G_L ... G_{k+1} (dG_k/dtheta_k) G_{k-1} ... G_1 |psi_0>
    
    # Let |adj> = |psi_final> initially? No.
    # Let's define the adjoint state |a> such that <a | dpsi_k> = <H psi_final | dpsi_k>.
    # So |a> should be H |psi_final>?
    # Check: <H psi_final | dpsi_k> = (H psi_final)^dagger dpsi_k.
    # If we propagate |a> = H psi_final backwards, we need to be careful with the adjoint of the gates.
    
    # Let's use the standard reverse-mode AD:
    # Start with |a> = H |psi_final> (this is the "output gradient" for the energy, effectively).
    # Actually, the gradient of E w.r.t the final state is H |psi_final>.
    # So we start with |a> = H |psi_final>.
    
    # Wait, E = <psi|H|psi>. dE = <dpsi|H|psi> + <psi|H|dpsi> = 2 Re[<dpsi|H|psi>].
    # So the "gradient" of E w.r.t psi is 2 H psi (real part handling aside).
    # Let's just compute the complex gradient <H psi | dpsi_k> and take 2*Re at the end.
    
    # So, initialize adjoint vector |a> = H @ psi_final.
    a = H @ psi_final
    
    grads = np.zeros(num_params, dtype=np.float64)
    
    # We need to reverse the gate sequence.
    # The gates were:
    # L=2: Ry(0..n-1)
    # L=1: Ry(0..n-1), CZ(0..n-2)
    # L=0: Ry(0..n-1), CZ(0..n-2)
    
    # Reverse order:
    # 1. Undo L=2 Ry gates (from n-1 down to 0)
    # 2. Undo L=1 CZ gates (from n-2 down to 0)
    # 3. Undo L=1 Ry gates (from n-1 down to 0)
    # 4. Undo L=0 CZ gates (from n-2 down to 0)
    # 5. Undo L=0 Ry gates (from n-1 down to 0)
    
    # Helper to undo Ry:
    # Ry(theta) = [[c, -s], [s, c]]
    # Ry(theta)^dagger = Ry(-theta) = [[c, s], [-s, c]]
    # If |psi_out> = Ry |psi_in>, then |psi_in> = Ry^dagger |psi_out>.
    
    # Helper to undo CZ:
    # CZ is Hermitian and unitary, so CZ^dagger = CZ.
    
    # We need to compute <a | dpsi_k> at the point of the gate.
    # When we are at the position of Ry gate k in the reverse pass, we have the adjoint state |a> that has been propagated back through all subsequent gates.
    # The contribution to the gradient is:
    # g_k = <a | (dRy_k/dtheta_k) |psi_before_k>
    
    # Note: |a> here is the state after applying the adjoints of all subsequent gates to the initial adjoint vector.
    
    # Let's implement the reverse pass carefully.
    
    # We stored states_before_ry.
    # We also need to know the angles.
    
    # Indexing:
    # k = layer * n + q
    
    # Reverse loop:
    for layer in range(layers - 1, -1, -1):
        # Undo Ry gates of this layer (reverse order)
        for q in range(n - 1, -1, -1):
            k = layer * n + q
            theta = angles[k]
            psi_before = states_before_ry[k]
            
            # Calculate gradient contribution
            # dRy/dtheta = 0.5 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
            # Let's compute <a | dRy | psi_before>
            
            # Apply Ry^dagger to a to get a_in? No, we need <a | dRy | psi_before>.
            # This is equivalent to <(Ry^dagger a) | (Ry^dagger dRy | psi_before)>? No.
            # Just compute the inner product directly.
            
            # dRy |psi_before>
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # dRy matrix:
            # 0.5 * [ [-s, -c], [c, -s] ]
            
            # We need to apply dRy to psi_before on qubit q.
            # And take inner product with a on qubit q.
            
            # Move qubit q to last axis for both a and psi_before
            a_moved = np.moveaxis(a, q, -1)
            psi_moved = np.moveaxis(psi_before, q, -1)
            
            # dpsi_0 = 0.5 * (-s * psi_0 - c * psi_1)
            # dpsi_1 = 0.5 * (c * psi_0 - s * psi_1)
            
            dpsi_0 = 0.5 * (-s * psi_moved[..., 0] - c * psi_moved[..., 1])
            dpsi_1 = 0.5 * (c * psi_moved[..., 0] - s * psi_moved[..., 1])
            dpsi = np.stack([dpsi_0, dpsi_1], axis=-1)
            
            # Inner product: sum conj(a) * dpsi
            # a_moved shape: (..., 2)
            # dpsi shape: (..., 2)
            prod = np.conj(a_moved) * dpsi
            val = np.sum(prod)
            
            grads[k] = 2.0 * val.real
            
            # Propagate a backwards through Ry gate
            # a_in = Ry^dagger a
            # Ry^dagger = [[c, s], [-s, c]]
            a_new_0 = c * a_moved[..., 0] + s * a_moved[..., 1]
            a_new_1 = -s * a_moved[..., 0] + c * a_moved[..., 1]
            a_new = np.stack([a_new_0, a_new_1], axis=-1)
            a = np.moveaxis(a_new, -1, q)
            
        # Undo CZ gates of this layer (if not last layer)
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                # CZ is its own inverse
                a = _apply_cz(a, q, n)
                
    return psi_final, grads


def _expand_symmetric_params(unique_params, n, layers):
    """
    Expand symmetry-reduced parameters to full parameters.
    Symmetry: theta[l, q] = theta[l, n-1-q]
    unique_params: array of shape (layers, half_n) where half_n = (n+1)//2
    Returns: array of shape (layers, n)
    """
    half_n = (n + 1) // 2
    full_params = np.zeros((layers, n))
    for l in range(layers):
        for q in range(half_n):
            val = unique_params[l, q]
            full_params[l, q] = val
            full_params[l, n - 1 - q] = val
    return full_params


def _objective_unique(unique_params, n, layers, H, e0):
    """Objective function for symmetry-reduced parameters."""
    full_params = _expand_symmetric_params(unique_params, n, layers)
    return _energy_from_params(full_params, n, layers, H, e0)


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles using grid search over symmetry-reduced space + local polish."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(H)
    
    half_n = (n + 1) // 2
    num_unique = layers * half_n
    
    def obj_flat(unique_flat):
        unique_2d = unique_flat.reshape(layers, half_n)
        return _objective_unique(unique_2d, n, layers, H, e0)
    
    # Physics-based initial guess
    ratio = field / coupling if coupling > 0 else 1.0
    base_angle = np.arcsin(np.tanh(ratio))
    
    # Grid search over reduced space
    # Use a moderate grid resolution to balance quality and speed
    grid_points = 8
    grid_vals = np.linspace(-np.pi / 2, np.pi / 2, grid_points)
    
    best_val = np.inf
    best_unique = np.full((layers, half_n), base_angle)
    
    # Generate grid points
    grid_indices = np.ndindex(*([grid_points] * num_unique))
    
    count = 0
    for idx_tuple in grid_indices:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 30:
            break
        
        unique_flat = np.array([grid_vals[i] for i in idx_tuple])
        val = obj_flat(unique_flat)
        if val < best_val:
            best_val = val
            best_unique = unique_flat.reshape(layers, half_n).copy()
        count += 1
    
    # Local polish from best grid point
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 20:
        try:
            res = opt.minimize(
                obj_flat,
                best_unique.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-10
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_unique = res.x.reshape(layers, half_n).copy()
        except Exception:
            pass
    
    # Additional local polish from a few other good grid points
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 15:
        # Try polishing from base_angle
        try:
            res = opt.minimize(
                obj_flat,
                np.full(num_unique, base_angle),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-10,
                    'fatol': 1e-10
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_unique = res.x.reshape(layers, half_n).copy()
        except Exception:
            pass
    
    # Try from zero
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 10:
        try:
            res = opt.minimize(
                obj_flat,
                np.zeros(num_unique),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-10,
                    'fatol': 1e-10
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_unique = res.x.reshape(layers, half_n).copy()
        except Exception:
            pass
    
    full_params = _expand_symmetric_params(best_unique, n, layers)
    theta = full_params
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
