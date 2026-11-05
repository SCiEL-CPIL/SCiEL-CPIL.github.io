"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # ZZ terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * z_i * z_ip1
    
    # X terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            if ((idx >> i) & 1) == 0:
                idx1 = idx | (1 << i)
                H[idx, idx1] += -field
                H[idx1, idx] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def _circuit_state_vectorized(thetas, n, layers):
    """
    Apply the circuit to |0...0> using vectorized operations.
    thetas: (layers, n) array of angles.
    Returns the final state vector of dimension 2**n.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Precompute bit masks for efficiency
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Create mask for indices where bit i is 0
            mask = np.array([(idx >> i) & 1 == 0 for idx in range(dim)], dtype=bool)
            idxs_0 = np.where(mask)[0]
            idxs_1 = idxs_0 | (1 << i)
            
            a = state[idxs_0].copy()
            b = state[idxs_1].copy()
            
            state[idxs_0] = c * a - s * b
            state[idxs_1] = s * a + c * b
        
        # Apply CZ on every adjacent pair in ascending order, except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                # Create mask for indices where both bits i and i+1 are 1
                mask = np.array([
                    ((idx >> i) & 1) == 1 and ((idx >> (i + 1)) & 1) == 1 
                    for idx in range(dim)
                ], dtype=bool)
                state[mask] *= -1.0
    
    return state


def _circuit_energy_and_grad(thetas, n, coupling, field, layers, H):
    """
    Compute circuit energy and gradient using analytical differentiation.
    thetas: (layers, n) array of angles.
    Returns (energy, gradient) where gradient has shape (layers, n).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Store intermediate states for backpropagation
    # We'll use a simpler approach: finite differences with small step
    # But for speed, let's use the chain rule properly
    
    # Actually, let's just compute energy and use numerical gradient with small step
    # This is simpler and the function is smooth
    
    # First compute the state
    for layer in range(layers):
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            mask = np.array([(idx >> i) & 1 == 0 for idx in range(dim)], dtype=bool)
            idxs_0 = np.where(mask)[0]
            idxs_1 = idxs_0 | (1 << i)
            
            a = state[idxs_0].copy()
            b = state[idxs_1].copy()
            
            state[idxs_0] = c * a - s * b
            state[idxs_1] = s * a + c * b
        
        if layer < layers - 1:
            for i in range(n - 1):
                mask = np.array([
                    ((idx >> i) & 1) == 1 and ((idx >> (i + 1)) & 1) == 1 
                    for idx in range(dim)
                ], dtype=bool)
                state[mask] *= -1.0
    
    # Compute energy: <psi|H|psi>
    Hpsi = H @ state
    energy = np.real(np.vdot(state, Hpsi))
    
    # Compute gradient using central differences (small step for accuracy)
    grad = np.zeros_like(thetas)
    eps = 1e-5
    
    for layer in range(layers):
        for i in range(n):
            # Forward
            thetas_p = thetas.copy()
            thetas_p[layer, i] += eps
            state_p = _circuit_state_vectorized(thetas_p, n, layers)
            Hpsi_p = H @ state_p
            energy_p = np.real(np.vdot(state_p, Hpsi_p))
            
            # Backward
            thetas_m = thetas.copy()
            thetas_m[layer, i] -= eps
            state_m = _circuit_state_vectorized(thetas_m, n, layers)
            Hpsi_m = H @ state_m
            energy_m = np.real(np.vdot(state_m, Hpsi_m))
            
            grad[layer, i] = (energy_p - energy_m) / (2 * eps)
    
    return energy, grad


def _circuit_energy_fast(thetas, n, coupling, field, layers, H):
    """Compute circuit energy using vectorized state evolution."""
    state = _circuit_state_vectorized(thetas, n, layers)
    Hpsi = H @ state
    return np.real(np.vdot(state, Hpsi))


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    # Precompute exact ground energy and Hamiltonian
    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        thetas = params.reshape(layers, n)
        e = _circuit_energy_fast(thetas, n, coupling, field, layers, H)
        return e
    
    def obj_grad(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10, np.zeros_like(params)
        thetas = params.reshape(layers, n)
        e, g = _circuit_energy_and_grad(thetas, n, coupling, field, layers, H)
        return e, g.reshape(-1)
    
    # Generate diverse initializations
    n_params = layers * n
    inits = []
    
    # Mean-field like initialization: pi/2 for all
    inits.append(np.full(n_params, np.pi / 2.0))
    
    # All zeros
    inits.append(np.zeros(n_params))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        np.random.seed(seed)
        inits.append(np.random.uniform(-np.pi, np.pi, size=n_params))
    
    # Symmetric inits: same angle for all qubits in a layer
    for seed in [505, 606, 707, 808]:
        np.random.seed(seed)
        layer_angles = np.random.uniform(-np.pi, np.pi, size=layers)
        inits.append(np.repeat(layer_angles, n))
    
    # Near-zero perturbations
    np.random.seed(999)
    inits.append(np.random.randn(n_params) * 0.5)
    inits.append(np.random.randn(n_params) * 0.1)
    
    # Sort by objective value
    init_objs = []
    for x in inits:
        init_objs.append((obj(x), x))
    init_objs.sort(key=lambda t: t[0])
    
    best_params = init_objs[0][1]
    best_val = init_objs[0][0]
    
    # Optimize from best initializations using L-BFGS-B with analytical gradient
    n_restarts = 10
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
        
        x_start = init_objs[r][1] if r < len(init_objs) else init_objs[0][1] + np.random.randn(n_params) * 0.3
        
        try:
            res = minimize(
                obj_grad,
                x_start,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            res = minimize(
                obj_grad,
                best_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
