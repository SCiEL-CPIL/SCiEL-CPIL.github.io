# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: diagonal
    diag = np.zeros(dim)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        bits = np.arange(dim)
        bi = (bits >> i) & 1
        bj = (bits >> (i + 1)) & 1
        zval = (1 - 2 * bi) * (1 - 2 * bj)
        diag += -coupling * zval
    H[np.arange(dim), np.arange(dim)] = diag

    # X terms: off-diagonal
    for i in range(n):
        mask = 1 << i
        bits = np.arange(dim)
        idx0 = bits[(bits & mask) == 0]
        idx1 = idx0 | mask
        H[idx0, idx1] += -field
        H[idx1, idx0] += -field
    
    return H

def _circuit_state(n, layers, params, ry_indices, cz_indices):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_indices[q]
            
            # In-place update to avoid allocations
            # new_state[idx0] = c * state[idx0] - s * state[idx1]
            # new_state[idx1] = s * state[idx0] + c * state[idx1]
            # We need to store old values first because we are modifying state in-place
            # But idx0 and idx1 are disjoint, so we can do it carefully.
            old_idx0 = state[idx0].copy()
            old_idx1 = state[idx1].copy()
            
            state[idx0] = c * old_idx0 - s * old_idx1
            state[idx1] = s * old_idx0 + c * old_idx1
        
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                both_set = cz_indices[q]
                state[both_set] = -state[both_set]
    
    return state

def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _analytic_gradient(n, layers, params, H, ry_indices, cz_indices):
    """
    Compute gradient of energy with respect to params.
    params: (layers, n)
    Returns: (layers, n)
    """
    dim = 1 << n
    grad = np.zeros((layers, n), dtype=np.float64)
    
    # We need to compute the gradient for each parameter.
    # The energy is E = <psi| H |psi>.
    # dE/dtheta = 2 * Re( <psi| H | dpsi/dtheta > )
    # 
    # dpsi/dtheta is the state with the derivative of the gate applied at the specific position.
    # Since the circuit is a sequence of gates, we can compute the "forward" state up to the gate,
    # apply the derivative, and then "backward" propagate the rest.
    # However, doing this for every parameter is O(N_params * Circuit_Cost).
    # 
    # A more efficient way for small N is to just re-evaluate the circuit for each parameter 
    # with the gate replaced by its derivative. 
    # Derivative of Ry(theta) = [[-s/2, -c/2], [c/2, -s/2]] where c=cos(t/2), s=sin(t/2).
    # This is equivalent to applying (1/2) * Y * Ry(theta) ? No.
    # d/dtheta Ry(theta) = [[-sin(t/2)/2, -cos(t/2)/2], [cos(t/2)/2, -sin(t/2)/2]]
    # 
    # Let's just use the "forward-backward" or "Jacobian-vector product" logic manually.
    # Actually, for n=6, 18 params, and 120s, even O(N_params) circuit evals might be too slow 
    # if circuit eval is slow. But with optimized circuit eval, it should be fine.
    # Let's try to compute gradients by evaluating the circuit for each param perturbation 
    # using complex step differentiation? No, standard finite diff is too slow.
    # 
    # Let's implement the exact derivative by modifying the state evolution.
    # For each param (l, q):
    #   1. Evolve state to layer l, before q.
    #   2. Apply derivative gate.
    #   3. Evolve rest of circuit.
    #   4. Compute <psi_deriv | H | psi> + <psi | H | psi_deriv>
    #      Since H is Hermitian and real, and psi is complex, 
    #      dE = 2 Re( <psi| H | dpsi/dtheta > ).
    
    # To make this fast, we can precompute the "prefix" and "suffix" states?
    # That would require O(layers * n) states of size 2^n.
    # For n=6, 2^6=64 complex numbers. 3 layers * 6 qubits = 18 states.
    # 18 * 64 * 16 bytes ~ 18KB. Very small.
    # Let's do that.
    
    # prefix_states[l][q] = state after layer l, before qubit q is processed in layer l.
    # Wait, the order is: Ry(0), Ry(1), ..., Ry(n-1), then CZs.
    # So for layer l:
    #   State S_start
    #   Apply Ry(0) -> S_1
    #   Apply Ry(1) -> S_2
    #   ...
    #   Apply Ry(n-1) -> S_n
    #   Apply CZs -> S_end
    
    # We want d/dtheta_{l,q}.
    # We need S_before_q (state before applying Ry(l,q)) and S_after_rest (state after applying all subsequent gates).
    # Let S_{l,q} be the state just before Ry(l,q).
    # Let S_{l,q+1} be the state just after Ry(l,q).
    # S_{l,q+1} = Ry(q, theta) S_{l,q}
    # d S_{l,q+1} / d theta = dRy(q, theta) S_{l,q}
    
    # Then we apply the rest of the circuit (Ry(l, q+1...n-1) and CZs of layer l, and all subsequent layers).
    # Let U_rest be the unitary operator for the rest of the circuit.
    # |psi_deriv> = U_rest dRy S_{l,q}
    # |psi> = U_rest Ry S_{l,q}
    
    # We can compute S_{l,q} for all l,q by running the circuit forward and saving states.
    # Then we compute |psi_deriv> for each param.
    
    prefix_states = {} # key: (l, q) -> state before Ry(l, q)
    
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for l in range(layers):
        for q in range(n):
            prefix_states[(l, q)] = state.copy()
            
            # Apply Ry
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_indices[q]
            
            old_idx0 = state[idx0].copy()
            old_idx1 = state[idx1].copy()
            
            state[idx0] = c * old_idx0 - s * old_idx1
            state[idx1] = s * old_idx0 + c * old_idx1
            
        # Apply CZs
        if l < layers - 1:
            for q in range(n - 1):
                both_set = cz_indices[q]
                state[both_set] = -state[both_set]
                
    # Now for each param (l, q), we have S_before = prefix_states[(l, q)]
    # We need to compute S_after = U_rest S_before.
    # And S_deriv = U_rest dRy S_before.
    
    # To save time, we can compute the "suffix" unitaries?
    # Or just run the circuit from (l, q) to the end for each param?
    # That is O(N_params * Circuit_Cost).
    # With N_params=18, Circuit_Cost ~ 18 Ry + 9 CZ.
    # Total ~ 18 * 27 gate ops.
    # Each gate op is O(2^n) = 64 ops.
    # 18 * 27 * 64 ~ 30k ops. Very fast.
    
    final_state = state # This is the full circuit state |psi>
    
    for l in range(layers):
        for q in range(n):
            S_before = prefix_states[(l, q)]
            
            # Compute S_deriv_start = dRy S_before
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_indices[q]
            
            old_idx0 = S_before[idx0].copy()
            old_idx1 = S_before[idx1].copy()
            
            # dRy = [[-s/2, -c/2], [c/2, -s/2]]
            # new_idx0 = -s/2 * old_idx0 - c/2 * old_idx1
            # new_idx1 = c/2 * old_idx0 - s/2 * old_idx1
            S_deriv_start = S_before.copy() # We will modify this
            S_deriv_start[idx0] = -0.5 * s * old_idx0 - 0.5 * c * old_idx1
            S_deriv_start[idx1] = 0.5 * c * old_idx0 - 0.5 * s * old_idx1
            
            # Now evolve S_deriv_start through the rest of the circuit
            # Rest of circuit: Ry(l, q+1...n-1), CZs(l), Layers l+1...L-1
            
            # We need a helper to evolve a state from a specific point (l, q+1) to the end.
            # Let's just write a local function or inline it.
            
            S_deriv = S_deriv_start
            
            # Apply remaining Ry in layer l
            for q2 in range(q + 1, n):
                theta2 = params[l, q2]
                c2 = np.cos(theta2 / 2.0)
                s2 = np.sin(theta2 / 2.0)
                idx0_2, idx1_2 = ry_indices[q2]
                
                old0 = S_deriv[idx0_2].copy()
                old1 = S_deriv[idx1_2].copy()
                
                S_deriv[idx0_2] = c2 * old0 - s2 * old1
                S_deriv[idx1_2] = s2 * old0 + c2 * old1
            
            # Apply CZs in layer l
            if l < layers - 1:
                for q2 in range(n - 1):
                    both_set = cz_indices[q2]
                    S_deriv[both_set] = -S_deriv[both_set]
                    
            # Apply subsequent layers
            for l2 in range(l + 1, layers):
                for q2 in range(n):
                    theta2 = params[l2, q2]
                    c2 = np.cos(theta2 / 2.0)
                    s2 = np.sin(theta2 / 2.0)
                    idx0_2, idx1_2 = ry_indices[q2]
                    
                    old0 = S_deriv[idx0_2].copy()
                    old1 = S_deriv[idx1_2].copy()
                    
                    S_deriv[idx0_2] = c2 * old0 - s2 * old1
                    S_deriv[idx1_2] = s2 * old0 + c2 * old1
                    
                if l2 < layers - 1:
                    for q2 in range(n - 1):
                        both_set = cz_indices[q2]
                        S_deriv[both_set] = -S_deriv[both_set]
                        
            # Now S_deriv is |psi_deriv>
            # dE = 2 Re( <psi| H | psi_deriv> )
            # H | psi_deriv>
            H_deriv = H @ S_deriv
            val = np.vdot(final_state, H_deriv)
            grad[l, q] = 2.0 * np.real(val)
            
    return grad

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Precompute Hamiltonian and ground energy
    H = _hamiltonian(n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    
    # Precompute indices for Ry and CZ
    dim = 1 << n
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        bits = np.arange(dim)
        idx0 = bits[(bits & mask) == 0]
        idx1 = idx0 | mask
        ry_indices.append((idx0, idx1))
        
    cz_indices = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        bits = np.arange(dim)
        both_set = bits[(bits & mask1) != 0]
        both_set = both_set[(both_set & mask2) != 0]
        cz_indices.append(both_set)

    def obj(params):
        state = _circuit_state(n, layers, params, ry_indices, cz_indices)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n

    def gradient(params):
        return _analytic_gradient(n, layers, params, H, ry_indices, cz_indices)

    # Structured initial guesses
    ratio = field / coupling
    
    if ratio < 1.0:
        base_angle = 0.5 * np.arctan(ratio)
    elif ratio < 2.0:
        base_angle = np.pi / 4
    else:
        base_angle = 0.5 * (np.pi / 2 + np.arctan(1.0 / (ratio - 1)))
        
    np.random.seed(42)
    
    initial_guesses = []
    
    # Guess 1: All angles equal to base_angle
    initial_guesses.append(np.full((layers, n), base_angle))
    
    # Guess 2: Alternating pattern
    guess2 = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                guess2[l, q] = base_angle
            else:
                guess2[l, q] = -base_angle
    initial_guesses.append(guess2)
    
    # Guess 3: All zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # Guess 4: All pi/2
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    # Guess 5-12: Random perturbations around base_angle
    for i in range(8):
        guess = np.full((layers, n), base_angle) + np.random.normal(0, 0.3, (layers, n))
        initial_guesses.append(guess)
    
    # Also try random starting points
    for i in range(4):
        guess = np.random.uniform(-np.pi/2, np.pi/2, (layers, n))
        initial_guesses.append(guess)
        
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 15:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def grad_flat(xf):
                p = xf.reshape((layers, n))
                g = gradient(p)
                return g.flatten()
            
            # Try L-BFGS-B with analytical gradient
            try:
                res = minimize(obj_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            # Polish with Nelder-Mead (no gradient needed, robust)
            if time.time() > deadline - 10:
                break
            
            try:
                res2 = minimize(obj_flat, x_flat, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
            
            # Additional polish with tighter L-BFGS-B
            if time.time() > deadline - 5:
                break
            
            try:
                res3 = minimize(obj_flat, best_params.flatten(), jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        # Fallback: simple local optimization
        for guess in initial_guesses:
            if time.time() > deadline - 10:
                break
            
            params = guess.copy()
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                grad = gradient(params)
                lr = 0.01 / (1 + step * 0.1)
                params_new = params - lr * grad
                val_new = obj(params_new)
                if val_new < best_val:
                    best_val = val_new
                    best_params = params_new
                params = params_new
    
    # Final verification
    if time.time() < deadline - 1:
        try:
            final_val = obj(best_params)
            if final_val < best_val:
                best_val = final_val
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
