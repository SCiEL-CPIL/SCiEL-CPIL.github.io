import numpy as np
import scipy.optimize as opt
import time

# Cache for optimized angles
_cache = {}

def _apply_ry(state, i, theta, n):
    """
    Apply Ry(theta) to qubit i of the state vector.
    Uses vectorized operations for speed.
    """
    dim = 2 ** n
    # We need to reshape the state to apply the gate on axis i
    # The state is indexed by binary number b_{n-1}...b_0.
    # Qubit i corresponds to the (n-1-i)th axis in a tensor of shape (2,)*n if we map index to bits correctly.
    # Let's use a simpler approach: reshape to (2,)*n and swap axes.
    
    # Index mapping:
    # If state is a vector of length 2^n, index k has bits k = sum b_j 2^j.
    # To apply gate on qubit i (bit i), we need to group indices where bit i is 0 and 1.
    # This is equivalent to reshaping to (2, 2, ..., 2) and applying on the axis corresponding to bit i.
    
    # Let's determine the axis.
    # If we reshape to (2,)*n, the first axis is the most significant bit (MSB, bit n-1).
    # So bit i corresponds to axis n-1-i.
    
    axis = n - 1 - i
    
    # Reshape to tensor
    tensor = state.reshape([2] * n)
    
    # Move the target axis to the last position for easy matrix multiplication
    # Or just use np.tensordot or einsum.
    # Let's use np.einsum or simple matrix multiplication on the reshaped view.
    
    # Construct Ry matrix
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    
    # Apply matrix multiplication along the axis
    # We can move the axis to the end, reshape to (..., 2), multiply, and move back.
    tensor = np.moveaxis(tensor, axis, -1)
    # Now the last axis is the qubit i.
    # Apply ry: new_tensor[..., 0] = c*old[...,0] - s*old[...,1]
    #           new_tensor[..., 1] = s*old[...,0] + c*old[...,1]
    
    # Vectorized update
    old_0 = tensor[..., 0]
    old_1 = tensor[..., 1]
    
    new_0 = c * old_0 - s * old_1
    new_1 = s * old_0 + c * old_1
    
    # We need to create a new tensor or update in place carefully.
    # Since we moved axes, we need to put it back.
    # Let's create the new tensor.
    # The shape of tensor is (2, 2, ..., 2, 2) where the last is the qubit.
    # We can create a new array of the same shape.
    new_tensor = np.empty_like(tensor)
    new_tensor[..., 0] = new_0
    new_tensor[..., 1] = new_1
    
    # Move axis back
    new_tensor = np.moveaxis(new_tensor, -1, axis)
    
    return new_tensor.reshape(-1)

def _apply_cz(state, i, j, n):
    """
    Apply CZ gate between qubit i and qubit j.
    """
    dim = 2 ** n
    tensor = state.reshape([2] * n)
    
    axis_i = n - 1 - i
    axis_j = n - 1 - j
    
    # Move axes to the end for easy access
    # We need to flip the sign of the component where bit i=1 and bit j=1.
    # In the tensor representation, this is tensor[..., 1, ..., 1, ...] at positions axis_i and axis_j.
    
    # It's easier to just iterate over the basis states for CZ since it's diagonal and sparse in a way.
    # But for speed, let's use masks.
    # Mask for bit i being 1
    mask_i = 1 << i
    mask_j = 1 << j
    
    # Indices where both bits are 1
    # We can generate indices or use vectorized logic.
    # For n=6, 64 elements. Looping is fine, but let's try vectorized.
    
    # Create an array of indices 0 to 2^n - 1
    idx = np.arange(dim)
    # Check condition
    cond = ((idx & mask_i) != 0) & ((idx & mask_j) != 0)
    
    state = state.copy()
    state[cond] *= -1
    
    return state

def _circuit_energy(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value for the given circuit parameters.
    params: 1D array of angles (layers * n)
    """
    thetas = params.reshape(layers, n)
    
    # Initial state |0>^n
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    # Apply circuit
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            state = _apply_ry(state, i, thetas[l, i], n)
        
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    # Calculate Energy E = <psi| H |psi>
    # H = -coupling * sum(Z_i Z_{i+1}) - field * sum(X_i)
    
    # Term 1: Z Z interactions (Diagonal)
    E_zz = 0.0
    probs = np.abs(state)**2
    # Precompute Z eigenvalues for all basis states
    # Z_i = +1 if bit i is 0, -1 if bit i is 1
    # ZZ = Z_i * Z_{i+1}
    
    # Vectorized calculation of ZZ term
    idx = np.arange(2**n)
    zz_sum = np.zeros(2**n)
    for i in range(n - 1):
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_sum += z_i * z_ip1
        
    E_zz = np.sum(probs * zz_sum)
    E_1 = -coupling * E_zz
    
    # Term 2: X interactions (Off-diagonal)
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        # <psi| X_i |psi> = sum_k psi^*_k psi_{k^f}
        # This is sum over all k.
        # Note: sum_k psi^*_k psi_{k^f} = 2 * Re( sum_{k < k^f} psi^*_k psi_{k^f} )
        # But the direct sum is fine.
        E_x += np.sum(np.conj(state) * state[idx_flipped])
        
    E_x = np.real(E_x)
    E_2 = -field * E_x
    
    total_energy = E_1 + E_2
    
    return total_energy

def _get_initial_guess(n, coupling, field, layers):
    """
    Generate a good initial guess for the angles.
    We compute the exact ground state and try to match the single-qubit magnetizations.
    """
    # Build Hamiltonian
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for k in range(dim):
            bit_i = (k >> i) & 1
            bit_ip1 = (k >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[k, k] += -coupling * z_i * z_ip1
            
    # X terms
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            k_flipped = k ^ mask
            H[k, k_flipped] += -field
            H[k_flipped, k] += -field
            
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    
    # For each layer, we want to choose Ry angles such that the state after that layer
    # approximates the ground state's local properties.
    # A simple heuristic: Start from |0> and apply Ry to match the Z-expectation of the ground state.
    # <Z_i> = |0><0| - |1><1| = 1 - 2*P(1).
    # For a single qubit in state cos(theta/2)|0> + sin(theta/2)|1>, <Z> = cos(theta).
    # So theta = arccos(<Z_i>_ground).
    
    # However, the circuit has entanglers. A more sophisticated approach would be to
    # back-propagate or use a variational ansatz.
    # Given the shallow depth (3 layers), we can just use the ground state's local magnetizations
    # as a target for the *final* layer, and assume intermediate layers can be tuned to help.
    # Actually, let's just use the ground state's single-qubit reduced density matrices to set
    # the angles for the LAST layer, and set previous layers to 0 or small random values.
    # Or better: Use the ground state to guide the optimization by starting near the "classical" solution.
    
    # Let's calculate <Z_i> for the ground state.
    probs = np.abs(ground_state)**2
    idx = np.arange(dim)
    z_expectations = np.zeros(n)
    for i in range(n):
        bit_i = (idx >> i) & 1
        z_vals = 1 - 2 * bit_i
        z_expectations[i] = np.sum(probs * z_vals)
        
    # Target angles for the last layer to achieve these Z expectations
    # cos(theta) = <Z> => theta = arccos(<Z>)
    # Note: Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    # <Z> = cos^2(theta/2) - sin^2(theta/2) = cos(theta).
    # So theta = arccos(<Z>).
    
    theta_last = np.arccos(np.clip(z_expectations, -1, 1))
    
    # For the first two layers, we can try to "pre-rotate" to help the entanglers.
    # A common strategy is to set them to 0 or small values, or use a random initialization.
    # Let's use a few different initializations:
    # 1. All zeros
    # 2. Last layer from ground state, others zero
    # 3. All layers from ground state (approximation)
    
    inits = []
    
    # Init 1: Zeros
    inits.append(np.zeros(n * layers))
    
    # Init 2: Only last layer set
    init2 = np.zeros(n * layers)
    init2[-n:] = theta_last
    inits.append(init2)
    
    # Init 3: All layers set to theta_last
    init3 = np.tile(theta_last, layers)
    inits.append(init3)
    
    # Init 4: Random
    inits.append(np.random.uniform(0, 2*np.pi, n * layers))
    
    return inits

def _optimize(n, coupling, field, layers, timeout=110.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # Get initial guesses
    inits = _get_initial_guess(n, coupling, field, layers)
    
    best_params = inits[0]
    best_energy = _circuit_energy(best_params, n, coupling, field, layers)
    
    # Optimize from each initial guess
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        # Use L-BFGS-B for local optimization
        # We need to provide bounds or let it handle it. Angles are periodic, but L-BFGS-B works on R^n.
        # We can wrap the angles in the objective function or just let it optimize in R^n and map back.
        # Since the energy function is periodic, local minima in R^n correspond to minima on the torus.
        
        def objective(params):
            if time.time() - start_time > timeout:
                raise TimeoutError("Optimization timeout")
            return _circuit_energy(params, n, coupling, field, layers)
        
        try:
            res = opt.minimize(
                objective, 
                init, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except TimeoutError:
            continue
        except Exception:
            continue
            
    # Final polish with Nelder-Mead if time permits
    current_time = time.time()
    if current_time - start_time < timeout - 5:
        def objective_nm(params):
            if time.time() - start_time > timeout:
                raise TimeoutError("Optimization timeout")
            return _circuit_energy(params, n, coupling, field, layers)
            
        try:
            res_nm = opt.minimize(
                objective_nm, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res_nm.fun < best_energy:
                best_energy = res_nm.fun
                best_params = res_nm.x
        except TimeoutError:
            pass
        except Exception:
            pass

    # Reshape to (layers, n)
    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    # Ensure types are consistent
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
