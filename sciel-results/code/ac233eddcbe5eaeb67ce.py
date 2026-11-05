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
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals

def _apply_mixer(psi, n, beta):
    """
    Apply exp(-i * beta * sum(X_i)) to the state vector psi.
    psi is a flat array of length 2^n.
    Returns a new array.
    """
    shape = (2,) * n
    psi_tensor = psi.reshape(shape)
    
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    # Pre-allocate result
    new_psi = np.empty_like(psi_tensor)
    
    for i in range(n):
        slice_0 = [slice(None)] * n
        slice_1 = [slice(None)] * n
        slice_0[i] = 0
        slice_1[i] = 1
        
        half_0 = psi_tensor[tuple(slice_0)]
        half_1 = psi_tensor[tuple(slice_1)]
        
        new_psi[tuple(slice_0)] = cb * half_0 + sb * half_1
        new_psi[tuple(slice_1)] = sb * half_0 + cb * half_1
        
    return new_psi.reshape(-1)

def _apply_mixer_dagger(psi, n, beta):
    """
    Apply exp(+i * beta * sum(X_i)) to the state vector psi.
    """
    return _apply_mixer(psi, n, -beta)

def _apply_cost(psi, cut_vals, gamma):
    """
    Apply exp(-i * gamma * C) to psi.
    """
    multipliers = np.exp(-1j * gamma * cut_vals)
    return psi * multipliers

def _apply_cost_dagger(psi, cut_vals, gamma):
    """
    Apply exp(+i * gamma * C) to psi.
    """
    multipliers = np.exp(1j * gamma * cut_vals)
    return psi * multipliers

def _apply_mixer_h(psi, n):
    """
    Apply H_mixer = sum(X_i) to psi.
    """
    # X_i flips bit i.
    # We can do this by swapping the two halves along each axis.
    shape = (2,) * n
    psi_tensor = psi.reshape(shape)
    result = np.empty_like(psi_tensor)
    
    for i in range(n):
        slice_0 = [slice(None)] * n
        slice_1 = [slice(None)] * n
        slice_0[i] = 0
        slice_1[i] = 1
        
        result[tuple(slice_0)] = psi_tensor[tuple(slice_1)]
        result[tuple(slice_1)] = psi_tensor[tuple(slice_0)]
        
    return result.reshape(-1)

def _qaoa_expectation_and_grad(n, edges, gammas, betas, cut_vals):
    """
    Compute expectation and gradient.
    gammas, betas: arrays of length p.
    """
    p = len(gammas)
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Store states: psi_0, psi_1, ..., psi_2p
    # psi_{2k} is state after mixer k-1 (or start)
    # psi_{2k+1} is state after cost k
    # psi_{2k+2} is state after mixer k
    states = [psi]
    
    for k in range(p):
        # Cost
        psi = _apply_cost(psi, cut_vals, gammas[k])
        states.append(psi)
        # Mixer
        psi = _apply_mixer(psi, n, betas[k])
        states.append(psi)
        
    # psi_final is states[2p]
    psi_final = states[-1]
    exp_cut = np.vdot(psi_final, cut_vals * psi_final).real
    
    # Backpropagation
    # chi_{2p} = C * psi_final
    chi = cut_vals * psi_final
    
    grads = np.zeros(2 * p)
    
    # Backward loop
    for k in range(p - 1, -1, -1):
        # Current chi is chi_{2k+2} (state after mixer k, propagated back)
        # We need to compute grad for beta_k and propagate chi to chi_{2k+1}
        
        # H_mixer = sum X_i
        # grad_beta_k = 2 * Im <chi_{2k+1} | H_mixer | psi_{2k+2}>
        # Wait, the formula derived was:
        # g_theta = 2 Im <chi_j | H_j | psi_{j+1}>
        # Here j = 2k+2 for beta? No.
        # Let's trace indices carefully.
        # Step 2k+2 is Mixer. Input psi_{2k+1}, Output psi_{2k+2}.
        # H = H_mixer.
        # grad_beta_k = 2 Im <chi_{2k+1} | H_mixer | psi_{2k+2}>
        
        # We have chi_{2k+2} currently in `chi`.
        # We need chi_{2k+1} = U_mixer^dagger chi_{2k+2}.
        # We also need psi_{2k+1} and psi_{2k+2} from `states`.
        
        psi_after_mixer = states[2 * k + 2]
        psi_before_mixer = states[2 * k + 1]
        
        # Compute H_mixer * psi_after_mixer
        h_psi_after = _apply_mixer_h(psi_after_mixer, n)
        
        # Inner product <chi_{2k+1} | H | psi_{2k+2}>
        # But we don't have chi_{2k+1} yet. We have chi_{2k+2}.
        # The formula uses chi_j where j is the index *before* the unitary U_j?
        # Let's re-verify:
        # dE/dtheta = 2 Im <chi_j | H_j | psi_{j+1}>
        # Where chi_j is the adjoint state *at the input of U_j*.
        # So for Mixer (step 2k+2), the input is psi_{2k+1}.
        # So we need chi_{2k+1}.
        
        # First, compute chi_{2k+1} = U_mixer^dagger chi_{2k+2}
        chi_before_mixer = _apply_mixer_dagger(chi, n, betas[k])
        
        # Now compute grad_beta_k
        # <chi_before_mixer | H_mixer | psi_after_mixer>
        inner = np.vdot(chi_before_mixer, h_psi_after)
        grads[2 * k + 1] = 2 * inner.imag
        
        # Now propagate chi to cost layer
        # Current chi is chi_{2k+1}.
        # We need to compute grad for gamma_k.
        # Step 2k+1 is Cost. Input psi_{2k}, Output psi_{2k+1}.
        # H = C.
        # grad_gamma_k = 2 Im <chi_{2k} | C | psi_{2k+1}>
        
        psi_after_cost = states[2 * k + 1]
        psi_before_cost = states[2 * k]
        
        # We need chi_{2k} = U_cost^dagger chi_{2k+1}
        chi_before_cost = _apply_cost_dagger(chi_before_mixer, cut_vals, gammas[k])
        
        # Compute C * psi_after_cost
        # C is diagonal, so this is just cut_vals * psi_after_cost
        c_psi_after = cut_vals * psi_after_cost
        
        # Inner product <chi_before_cost | C | psi_after_cost>
        inner_g = np.vdot(chi_before_cost, c_psi_after)
        grads[2 * k] = 2 * inner_g.imag
        
        # Update chi for next iteration (which will be the previous mixer)
        chi = chi_before_cost
        
    return exp_cut, grads

def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using L-BFGS-B with multiple restarts.
    """
    cut_vals = _precompute_cut_values(n, edges)
    
    def objective_and_grad(params):
        # params: [g0, b0, g1, b1]
        g0, b0, g1, b1 = params
        gammas = np.array([g0, g1])
        betas = np.array([b0, b1])
        try:
            exp_cut, grads = _qaoa_expectation_and_grad(n, edges, gammas, betas, cut_vals)
            # Minimize -exp_cut
            return -exp_cut, -grads
        except Exception:
            return 1e9, np.zeros(4)

    best_score = -np.inf
    best_params = None
    
    num_restarts = 50 # Fewer restarts needed with gradients
    
    np.random.seed(42)
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        init = np.random.uniform(0, np.pi, size=4)
        
        options = {
            'maxiter': 100,
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
