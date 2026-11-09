import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for b in range(dim):
            if (b & mask_i) and (b & mask_ip1):
                # Both 1: Z= -1, Z Z = 1 -> -coupling * 1
                H[b, b] -= coupling
            elif (b & mask_i) or (b & mask_ip1):
                # One 1, one 0: Z= 1, -1 -> Z Z = -1 -> -coupling * -1 = +coupling
                H[b, b] += coupling
            else:
                # Both 0: Z= 1, 1 -> Z Z = 1 -> -coupling * 1
                H[b, b] -= coupling
            
    # X terms: -field * sum X_i
    for i in range(n):
        mask = 1 << i
        for b in range(dim):
            b_flipped = b ^ mask
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _circuit_state(params, n, layers):
    """
    Simulate the circuit and return the final state vector.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            state_tensor = state.reshape([2] * n)
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
            
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                both_one = (indices & mask1) != 0
                both_one &= (indices & mask2) != 0
                state[both_one] *= -1
                
    return state

def _energy_and_grad(params, n, layers, H, E0):
    """
    Compute energy and analytic gradient via backpropagation.
    """
    dim = 1 << n
    
    # Forward pass, storing states
    states = [np.zeros(dim, dtype=complex)]
    states[0][0] = 1.0
    current_state = states[0]
    
    ops = []
    for l in range(layers):
        for q in range(n):
            ops.append(('ry', l, q))
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            st = current_state.reshape([2] * n)
            st = np.moveaxis(st, q, 0)
            
            new_st = np.empty_like(st)
            new_st[0] = c * st[0] - s * st[1]
            new_st[1] = s * st[0] + c * st[1]
            
            new_st = np.moveaxis(new_st, 0, q)
            current_state = new_st.reshape(dim)
            states.append(current_state.copy())
            
        if l < layers - 1:
            for q in range(n - 1):
                ops.append(('cz', q))
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                both_one = (indices & mask1) != 0
                both_one &= (indices & mask2) != 0
                current_state[both_one] *= -1
                states.append(current_state.copy())
    
    psi = states[-1]
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    
    # Backward pass
    v = Hpsi.copy()
    grad = np.zeros((layers, n))
    
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        psi_before = states[i]
        
        if op[0] == 'ry':
            l, q = op[1], op[2]
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # dU/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # (dU/dtheta)^dagger = 0.5 * [[-s, c], [-c, -s]]
            # We need g = 2 Re[ <psi_before| (dU/dtheta)^dagger |v> ]
            
            v_tensor = v.reshape([2] * n)
            v_tensor = np.moveaxis(v_tensor, q, 0)
            
            A_00 = -0.5 * s
            A_01 = 0.5 * c
            A_10 = -0.5 * c
            A_11 = -0.5 * s
            
            new_v = np.empty_like(v_tensor)
            new_v[0] = A_00 * v_tensor[0] + A_01 * v_tensor[1]
            new_v[1] = A_10 * v_tensor[0] + A_11 * v_tensor[1]
            
            new_v = np.moveaxis(new_v, 0, q)
            v_adj = new_v.reshape(dim)
            
            g = 2.0 * np.vdot(psi_before, v_adj).real
            grad[l, q] = g
            
            # Propagate v backwards: v_before = U^dagger v_after
            # U^dagger = [[c, s], [-s, c]]
            v_tensor_orig = v.reshape([2] * n)
            v_tensor_orig = np.moveaxis(v_tensor_orig, q, 0)
            
            new_v_prop = np.empty_like(v_tensor_orig)
            new_v_prop[0] = c * v_tensor_orig[0] + s * v_tensor_orig[1]
            new_v_prop[1] = -s * v_tensor_orig[0] + c * v_tensor_orig[1]
            
            v_tensor_orig = np.moveaxis(new_v_prop, 0, q)
            v = v_tensor_orig.reshape(dim)
            
        elif op[0] == 'cz':
            q = op[1]
            mask1 = 1 << q
            mask2 = 1 << (q + 1)
            indices = np.arange(dim)
            both_one = (indices & mask1) != 0
            both_one &= (indices & mask2) != 0
            v[both_one] *= -1
            
    return energy - E0, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    best_result = None
    best_energy = np.inf
    
    # Multi-start L-BFGS-B
    n_restarts = 25
    rng = np.random.default_rng(42)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial point
        init_params = rng.uniform(-np.pi, np.pi, (layers, n))
        
        def objective(flat_params):
            if time.time() > deadline:
                return 1e10
            params = flat_params.reshape((layers, n))
            try:
                e, _ = _energy_and_grad(params, n, layers, H, E0)
                return e
            except Exception:
                return 1e10
        
        def grad_func(flat_params):
            if time.time() > deadline:
                return np.zeros_like(flat_params)
            params = flat_params.reshape((layers, n))
            try:
                _, g = _energy_and_grad(params, n, layers, H, E0)
                return g.flatten()
            except Exception:
                return np.zeros_like(flat_params)
        
        try:
            res = minimize(
                objective,
                init_params.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_result = res.x
                
        except Exception:
            pass
        
        # If we found a very good solution, we might stop early or refine
        if best_energy < 1e-6:
            break

    if best_result is None:
        return np.zeros((layers, n))
        
    return best_result.reshape((layers, n))
