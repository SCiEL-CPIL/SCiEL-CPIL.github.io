import numpy as np
import time
from scipy.optimize import minimize

def _circuit_state(n, layers, params):
    """Compute state from circuit."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
            idx1 = idx0 | mask
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
                state[idx_both] = -state[idx_both]
    
    return state

def _hamiltonian(n, coupling, field):
    """Build Hamiltonian."""
    dim = 1 << n
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

def _closed_form_angle(coupling, field):
    """
    Closed-form optimal Ry angle for Ising model.
    
    For H = -J*ZZ - h*X on a single qubit (effective), the optimal rotation
    to minimize energy satisfies tan(theta/2) = h/J, so theta = 2*arctan(h/J).
    This is equivalent to arccos(J/sqrt(J^2+h^2)).
    """
    if coupling == 0:
        return np.pi / 2
    ratio = field / coupling
    # theta = 2 * arctan(ratio)
    # Clamp ratio to avoid numerical issues
    ratio = np.clip(ratio, -1e6, 1e6)
    return 2.0 * np.arctan(ratio)

def _energy_and_gradient(n, layers, params, H, e_ground):
    """Compute energy and analytical gradient using parameter-shift rule."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Precompute index arrays for efficiency
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_indices.append((idx0, idx1))
    
    cz_indices = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_indices.append(idx_both)
    
    def apply_circuit(params_arr):
        """Apply circuit to |0>^n and return final state."""
        st = np.zeros(dim, dtype=np.complex128)
        st[0] = 1.0
        
        for layer in range(layers):
            for q in range(n):
                theta = params_arr[layer, q]
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                idx0, idx1 = ry_indices[q]
                old0 = st[idx0].copy()
                old1 = st[idx1].copy()
                st[idx0] = c * old0 - s * old1
                st[idx1] = s * old0 + c * old1
            
            if layer < layers - 1:
                for q in range(n - 1):
                    idx_both = cz_indices[q]
                    st[idx_both] = -st[idx_both]
        
        return st
    
    # Compute state and energy
    state = apply_circuit(params)
    Hs = H @ state
    energy = np.vdot(state, Hs).real
    objective = (energy - e_ground) / n
    
    # Compute gradient using parameter-shift rule
    # For Ry(theta): d/dtheta <psi|H|psi> = 0.5 * (<psi(theta+pi/2)|H|psi(theta+pi/2)> - <psi(theta-pi/2)|H|psi(theta-pi/2)>)
    gradient = np.zeros((layers, n))
    
    for layer in range(layers):
        for q in range(n):
            # Shift +pi/2
            params_plus = params.copy()
            params_plus[layer, q] += np.pi / 2
            state_plus = apply_circuit(params_plus)
            energy_plus = np.vdot(state_plus, H @ state_plus).real
            
            # Shift -pi/2
            params_minus = params.copy()
            params_minus[layer, q] -= np.pi / 2
            state_minus = apply_circuit(params_minus)
            energy_minus = np.vdot(state_minus, H @ state_minus).real
            
            # Parameter-shift rule: gradient = 0.5 * (E(+) - E(-))
            gradient[layer, q] = 0.5 * (energy_plus - energy_minus) / n
    
    return objective, gradient.flatten()

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Compute closed-form optimal angle
    base_angle = _closed_form_angle(coupling, field)
    
    # Initialize all layers with the closed-form angle
    params = np.full((layers, n), base_angle)
    
    # Build Hamiltonian and compute ground state energy
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        p = xf.reshape((layers, n))
        val, grad = _energy_and_gradient(n, layers, p, H, e_ground)
        return val, grad
    
    # Evaluate the closed-form solution
    best_val = obj(params.flatten())
    best_params = params.copy()
    
    # Try a few variations: slightly different angles per layer
    # The CZ gates introduce phase correlations, so layer-dependent angles may help
    variations = []
    
    # Uniform variations
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        v = np.full((layers, n), base_angle * scale)
        variations.append(v)
    
    # Layer-decaying patterns
    for decay in [0.7, 0.85, 1.0]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (decay ** l)
        variations.append(v)
    
    # Layer-increasing patterns
    for growth in [1.2, 1.5]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (growth ** l)
        variations.append(v)
    
    # Evaluate all variations
    for v in variations:
        if time.time() > deadline - 10:
            break
        val = obj(v.flatten())
        if val < best_val:
            best_val = val
            best_params = v.copy()
    
    # Use L-BFGS-B with analytical gradients for fast, precise convergence
    if time.time() < deadline - 20:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 20})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final high-precision polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            res = minimize(obj_grad, best_params.flatten(), method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': 100, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # One more Nelder-Mead polish for any remaining non-smoothness
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
