"""QAOA parameter design with gradient-based optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _precompute_cut_values(n, edges):
    """
    Precompute the cut value for every possible state mask.
    Returns an array of shape (2^n,).
    """
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        mask_u = 1 << u
        mask_v = 1 << v
        # For each bit position, we want to add 1 if bits u and v differ.
        # This is equivalent to adding 1 to indices where (mask_u ^ mask_v) bit is set?
        # No, simpler:
        # cut_val(x) = sum_edges [x_u != x_v]
        # We can compute this by iterating over edges and adding to the appropriate masks.
        # Or vectorized:
        # Let's use the bit trick.
        # For a specific edge (u,v), the contribution is 1 if bit u and bit v differ.
        # This is equivalent to:
        # indices where (x >> u) & 1 != (x >> v) & 1
        # This is a bit complicated to vectorize directly without loops over edges.
        # Given n <= 10, dim <= 1024, and edges <= 45, a loop over edges is fine.
        pass
    
    # Efficient vectorized approach:
    # cut_vals = sum over edges of (x_u XOR x_v)
    # We can precompute bit arrays for each node.
    # bits[i] is an array of shape (dim,) where bits[i][mask] = (mask >> i) & 1
    # Then cut_vals = sum_{(u,v) in edges} (bits[u] ^ bits[v])
    
    # This is O(E * 2^n) which is 45 * 1024 ~ 46k ops. Very fast.
    bits = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        # Create array of bits for node i
        # mask from 0 to dim-1
        # bit i is set if (mask >> i) & 1
        # We can do:
        # masks = np.arange(dim)
        # bits[i] = (masks >> i) & 1
        # But (masks >> i) & 1 creates an array.
        # Let's do it.
        masks = np.arange(dim)
        bits[i] = ((masks >> i) & 1).astype(np.float64)
        
    for u, v in edges:
        cut_vals += (bits[u] ^ bits[v])
        
    return cut_vals

def _apply_mixer_fast(psi, n, beta):
    """
    Apply exp(-i * beta * sum(X_i)) to psi.
    """
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    # psi shape (2^n,)
    # Reshape to (2, 2, ..., 2)
    shape = (2,) * n
    psi_tensor = psi.reshape(shape)
    
    # We need to apply the mixer on each qubit.
    # The mixer is tensor product of single-qubit mixers.
    # Single qubit mixer matrix:
    # [[cb, sb], [sb, cb]]
    
    # We can use einsum or broadcasting.
    # Let's use a loop over qubits, which is simple and fast enough for n<=10.
    # Or we can construct the full matrix? No, 2^n x 2^n is too big.
    # Loop is fine.
    
    new_psi = np.empty_like(psi_tensor)
    
    for i in range(n):
        # Move axis i to front? Or just slice.
        # Slicing is easier.
        slice_0 = [slice(None)] * n
        slice_1 = [slice(None)] * n
        slice_0[i] = 0
        slice_1[i] = 1
        
        half_0 = psi_tensor[tuple(slice_0)]
        half_1 = psi_tensor[tuple(slice_1)]
        
        new_psi[tuple(slice_0)] = cb * half_0 + sb * half_1
        new_psi[tuple(slice_1)] = sb * half_0 + cb * half_1
        
    return new_psi.reshape(-1)

def _qaoa_state(n, edges, gammas, betas, cut_vals):
    """
    Compute the final state vector.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for k in range(len(gammas)):
        # Cost
        multipliers = np.exp(-1j * gammas[k] * cut_vals)
        psi = psi * multipliers
        # Mixer
        psi = _apply_mixer_fast(psi, n, betas[k])
        
    return psi

def _qaoa_expectation(n, edges, gammas, betas, cut_vals):
    """
    Compute expectation value of cut.
    """
    psi = _qaoa_state(n, edges, gammas, betas, cut_vals)
    exp_cut = np.vdot(psi, cut_vals * psi).real
    return exp_cut

def _qaoa_gradient(n, edges, gammas, betas, cut_vals):
    """
    Compute gradient of expectation value w.r.t. parameters.
    Uses parameter shift rule or analytic gradient?
    For QAOA, analytic gradient is efficient.
    grad_gamma_k = 2 Im <psi_{k-1} | C | psi_k> ... wait, indices.
    
    Let's use the standard backpropagation for expectation value.
    E = <psi_final| C | psi_final>
    
    Alternatively, since p is small (p=2), we can just use finite differences?
    No, analytic is better.
    
    Let's derive/verify analytic gradient.
    E = Re <psi| C | psi>
    dE/dtheta = 2 Re <dpsi/dtheta| C | psi>
    
    dpsi/dtheta involves the generator of the unitary.
    U_theta = exp(-i theta H)
    dU/dtheta = -i H U
    dpsi/dtheta = -i H psi
    
    So dE/dgamma_k = 2 Re <(-i C psi_{k-1})| C | psi_k> ? No, the state changes.
    
    Let's stick to the backpropagation method implemented in the seed, but optimized.
    Or, since p=2, we can compute gradients using the chain rule manually.
    
    Actually, for p=2, the state is:
    psi_0 = |+>^n
    psi_1 = U_cost(g0) psi_0
    psi_2 = U_mix(b0) psi_1
    psi_3 = U_cost(g1) psi_2
    psi_4 = U_mix(b1) psi_3
    
    E = <psi_4| C | psi_4>
    
    Gradients:
    dE/dg1 = 2 Im <psi_3| C | psi_3> ? No.
    dE/dtheta = 2 Im <psi| H | psi> for a single step?
    
    Let's use the formula:
    dE/dtheta_j = 2 Im <psi_j| H_j | psi_j> is only valid if the state is an eigenstate? No.
    
    Correct formula for expectation value E = <psi| O | psi>:
    dE/dtheta = 2 Re <dpsi/dtheta| O | psi>
    
    For U = exp(-i theta H), dpsi/dtheta = -i H psi.
    So dE/dtheta = 2 Re <-i <psi| H | O | psi>> = 2 Im <psi| H O | psi> ?
    Wait, <psi| H | O | psi> is not necessarily real or imaginary.
    
    Let's use the backpropagation approach from the seed, it was correct.
    I will reuse the logic but ensure it's efficient.
    
    Actually, for p=2, I can just compute the gradient using the parameter shift rule?
    No, analytic is faster.
    
    Let's implement a clean backprop.
    """
    dim = 1 << n
    p = len(gammas)
    
    # Forward pass
    states = [np.ones(dim, dtype=complex) / np.sqrt(dim)]
    for k in range(p):
        # Cost
        psi = states[-1] * np.exp(-1j * gammas[k] * cut_vals)
        states.append(psi)
        # Mixer
        psi = _apply_mixer_fast(psi, n, betas[k])
        states.append(psi)
        
    psi_final = states[-1]
    exp_cut = np.vdot(psi_final, cut_vals * psi_final).real
    
    # Backward pass
    # chi_final = C psi_final
    chi = cut_vals * psi_final
    
    grads = np.zeros(2 * p)
    
    for k in range(p - 1, -1, -1):
        # Mixer step
        # psi_{2k+2} = U_mix(b_k) psi_{2k+1}
        # H_mix = sum X_i
        # dE/dbeta_k = 2 Im <chi_{2k+1}| H_mix | psi_{2k+2}>
        
        psi_after_mixer = states[2 * k + 2]
        psi_before_mixer = states[2 * k + 1]
        
        # Apply H_mix to psi_after_mixer
        # H_mix = sum X_i
        # X_i swaps halves along axis i.
        h_psi_after = np.zeros_like(psi_after_mixer)
        shape = (2,) * n
        psi_tensor = psi_after_mixer.reshape(shape)
        for i in range(n):
            slice_0 = [slice(None)] * n
            slice_1 = [slice(None)] * n
            slice_0[i] = 0
            slice_1[i] = 1
            h_psi_after.reshape(shape)[tuple(slice_0)] = psi_tensor[tuple(slice_1)]
            h_psi_after.reshape(shape)[tuple(slice_1)] = psi_tensor[tuple(slice_0)]
            
        # chi_before_mixer = U_mix^dagger chi
        chi_before_mixer = _apply_mixer_fast(chi, n, -betas[k])
        
        inner = np.vdot(chi_before_mixer, h_psi_after)
        grads[2 * k + 1] = 2 * inner.imag
        
        # Cost step
        # psi_{2k+1} = U_cost(g_k) psi_{2k}
        # H_cost = C
        # dE/dgamma_k = 2 Im <chi_{2k}| C | psi_{2k+1}>
        
        psi_after_cost = states[2 * k + 1]
        
        # chi_before_cost = U_cost^dagger chi_before_mixer
        chi_before_cost = chi_before_mixer * np.exp(1j * gammas[k] * cut_vals)
        
        c_psi_after = cut_vals * psi_after_cost
        inner_g = np.vdot(chi_before_cost, c_psi_after)
        grads[2 * k] = 2 * inner_g.imag
        
        chi = chi_before_cost
        
    return exp_cut, grads

def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using L-BFGS-B with multiple restarts.
    """
    cut_vals = _precompute_cut_values(n, edges)
    
    def objective_and_grad(params):
        g0, b0, g1, b1 = params
        gammas = np.array([g0, g1])
        betas = np.array([b0, b1])
        try:
            exp_cut, grads = _qaoa_gradient(n, edges, gammas, betas, cut_vals)
            return -exp_cut, -grads
        except Exception:
            return 1e9, np.zeros(4)

    best_score = -np.inf
    best_params = None
    
    np.random.seed(42)
    
    # Use more restarts if time permits
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        # Heuristic initialization
        # g in [0, pi], b in [0, pi]
        init = np.random.uniform(0, np.pi, size=4)
        
        options = {
            'maxiter': 50,
            'ftol': 1e-8,
            'gtol': 1e-6
        }
        
        try:
            res = scipy.optimize.minimize(objective_and_grad, init, method='L-BFGS-B', options=options)
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
            
    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])

def angles(n, edges, p):
    start_time = time.time()
    timeout = 145
    
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
# EVOLVE-BLOCK-END
