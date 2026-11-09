"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # Z_i Z_{i+1} terms (diagonal)
    for i in range(n - 1):
        for k in range(dim):
            z_i = 1.0 if ((k >> i) & 1) == 0 else -1.0
            z_j = 1.0 if ((k >> (i + 1)) & 1) == 0 else -1.0
            H[k, k] += -coupling * z_i * z_j
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for k in range(dim):
            k_flip = k ^ (1 << i)
            H[k, k_flip] += -field
            H[k_flip, k] += -field
    
    return H


def _apply_ry_per_qubit(state, thetas, n):
    """Apply Ry(theta_i) to qubit i for each i, where thetas is an array of n angles."""
    for i in range(n):
        theta = thetas[i]
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        # Ry = [[c, -s], [s, c]] acting on qubit i
        # For states where qubit i is 0: new_0 = c*old_0 + s*old_1
        # For states where qubit i is 1: new_1 = -s*old_0 + c*old_1
        mask0 = np.array([((k >> i) & 1) == 0 for k in range(len(state))], dtype=bool)
        mask1 = ~mask0
        old0 = state[mask0].copy()
        old1 = state[mask1].copy()
        # Need to pair up: for each k with bit i = 0, k_flip = k ^ (1<<i) has bit i = 1
        new_vals = np.empty_like(state)
        for k in range(len(state)):
            if ((k >> i) & 1) == 0:
                k_flip = k ^ (1 << i)
                new_vals[k] = c * state[k] + s * state[k_flip]
                new_vals[k_flip] = -s * state[k] + c * state[k_flip]
        state = new_vals
    return state


def _apply_cz_adjacent(state, n):
    """Apply CZ on every adjacent pair in ascending order."""
    for i in range(n - 1):
        for k in range(len(state)):
            if ((k >> i) & 1) == 1 and ((k >> (i + 1)) & 1) == 1:
                state[k] *= -1.0
    return state


def _apply_circuit(params, n, layers):
    """Apply the full circuit to |0>^n and return final state.
    params is (layers, n) array of angles."""
    state = np.zeros(1 << n)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        state = _apply_ry_per_qubit(state, params[layer], n)
        if layer < layers - 1:
            state = _apply_cz_adjacent(state, n)
    
    return state


def _circuit_energy(params, n, coupling, field, H):
    """Compute circuit energy."""
    state = _apply_circuit(params, n, 3)
    return state @ H @ state


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate Ising ground state."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Build Hamiltonian and get exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E_ground = eigvals[0]
    
    dim = 1 << n
    
    def objective(params_flat):
        params = params_flat.reshape(layers, n)
        E = _circuit_energy(params, n, coupling, field, H)
        return (E - E_ground) / n
    
    best_params = np.zeros((layers, n))
    best_val = np.inf
    
    # Physics-based initialization: small field -> near |0>, large field -> near |+>
    # For transverse field Ising, ground state transitions from ferromagnetic to paramagnetic
    # Use a reasonable starting point based on field strength
    base_angle = np.arctan(field / coupling) if coupling > 0 else np.pi / 4
    best_params[0] = base_angle * 0.5
    best_params[1] = base_angle * 0.5
    best_params[2] = base_angle
    
    # Try the initialized point first
    try:
        val = objective(best_params.flatten())
        if val < best_val:
            best_val = val
    except Exception:
        pass
    
    # Multiple random restarts with varied initializations
    np.random.seed(42)
    n_restarts = 12
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_params.flatten()
        elif restart == 1:
            # All pi/2 (maximally mixed)
            x0 = np.full(layers * n, np.pi / 2)
        elif restart == 2:
            # All zero
            x0 = np.zeros(layers * n)
        elif restart == 3:
            # Physics-inspired: increasing angles
            x0 = np.linspace(0, np.pi, layers * n)
        else:
            x0 = np.random.uniform(-np.pi, np.pi, size=layers * n)
        
        try:
            result = scipy.optimize.minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final refinement with more iterations
    if time.time() < deadline - 10:
        try:
            result = scipy.optimize.minimize(
                objective, best_params.flatten(), method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final check: ensure finite
    if not np.all(np.isfinite(best_params)):
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
    
    return best_params
# EVOLVE-BLOCK-END
