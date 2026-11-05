"""Optimize 3-layer Ry/CZ circuit for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
_DEADLINE = 150.0
_START_TIME = None


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _check_deadline():
    return time.time() - _get_start_time() < _DEADLINE - 5.0  # 5s safety margin


def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _czz_matrix():
    """Return 2x2 CZ gate matrix (control-target)."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=np.complex128)


def _apply_ry_to_state(state, n, qubit, theta):
    """Apply Ry(theta) to qubit `qubit` of n-qubit state (little-endian, qubit i is bit i)."""
    # Reshape state into n qubits
    dim = 2 ** n
    state = state.reshape([2] * n)
    
    # Permute so the target qubit is last
    # We want to move qubit index `qubit` to position n-1
    perm = list(range(n))
    perm = perm[:qubit] + perm[qubit+1:] + [qubit]
    state = np.transpose(state, perm)
    
    # Apply Ry on the last qubit
    ry = _ry_matrix(theta)
    state = np.tensordot(ry, state, axes=([1], [n-1]))
    
    # Reshape back and unpermute
    state = state.reshape([2] * n)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    state = np.transpose(state, inv_perm)
    
    return state.reshape(-1)


def _apply_cz_to_state(state, n, q1, q2):
    """Apply CZ between qubits q1 and q2 of n-qubit state."""
    dim = 2 ** n
    state = state.reshape([2] * n)
    
    # Permute so q1 and q2 are the last two qubits
    perm = list(range(n))
    perm = [p for p in perm if p not in (q1, q2)] + [q1, q2]
    state = np.transpose(state, perm)
    
    # Apply CZ on last two qubits
    cz = _czz_matrix()
    state = state.reshape((-1, 4))
    state = cz @ state
    state = state.reshape([2] * n)
    
    # Unpermute
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    state = np.transpose(state, inv_perm)
    
    return state.reshape(-1)


def _simulate_circuit(n, layers, thetas):
    """Simulate the full circuit and return the final state vector.
    thetas: (layers, n) array of angles.
    """
    dim = 2 ** n
    # Start in |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            state = _apply_ry_to_state(state, n, q, thetas[layer, q])
        
        # Apply CZ between adjacent pairs (if not the last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_to_state(state, n, q, q + 1)
    
    return state


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z on qubit i and qubit i+1
        # Z = |0><0| - |1><1|
        for bit in range(dim):
            zi = 1 if (bit >> i) & 1 == 0 else -1
            zj = 1 if (bit >> (i+1)) & 1 == 0 else -1
            H[bit, bit] += -coupling * zi * zj
    
    # X_i terms
    for i in range(n):
        for bit in range(dim):
            # X flips bit i
            flipped = bit ^ (1 << i)
            H[bit, flipped] += -field
    
    return H


def _energy(state, H):
    """Compute <state|H|state>."""
    return np.real(np.vdot(state, H @ state))


def _objective(params, n, coupling, field, layers, H, e0):
    """Objective: mean energy gap per spin."""
    thetas = params.reshape((layers, n))
    state = _simulate_circuit(n, layers, thetas)
    e = _energy(state, H)
    return (e - e0) / n


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0], H


def angles(n, coupling, field, layers):
    """Main entry point. Returns (layers, n) array of angles in radians."""
    _get_start_time()
    
    # Get exact ground state energy and Hamiltonian
    e0, H = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: product |+> state
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    params0 = theta0.flatten()
    
    # Try multiple optimization strategies
    best_params = params0.copy()
    best_obj = _objective(params0, n, coupling, field, layers, H, e0)
    
    if not _check_deadline():
        return best_params.reshape((layers, n))
    
    # Strategy 1: L-BFGS-B from initial guess
    try:
        res = minimize(_objective, params0, args=(n, coupling, field, layers, H, e0),
                       method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    if not _check_deadline():
        return best_params.reshape((layers, n))
    
    # Strategy 2: Basin hopping with L-BFGS-B
    def _basin_hopping_step(x, stepsize):
        # Random perturbation
        perturbed = x + stepsize * np.random.randn(len(x))
        # Local minimize
        try:
            res = minimize(_objective, perturbed, args=(n, coupling, field, layers, H, e0),
                           method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            return res.fun, res.x, 1.0
        except Exception:
            return _objective(perturbed, n, coupling, field, layers, H, e0), perturbed, 1.0
    
    # Run several basin hopping iterations
    current_params = best_params.copy()
    current_obj = best_obj
    
    for i in range(50):
        if not _check_deadline():
            break
        if time.time() - _get_start_time() > _DEADLINE - 10:
            break
        
        new_obj, new_params, _ = _basin_hopping_step(current_params, 0.3)
        # Metropolis acceptance
        if new_obj < current_obj:
            current_params = new_params
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = current_params.copy()
        else:
            # Accept with some probability to escape local minima
            delta = new_obj - current_obj
            if delta < 0.01 and np.random.rand() < np.exp(-delta / 0.005):
                current_params = new_params
                current_obj = new_obj
    
    if not _check_deadline():
        return best_params.reshape((layers, n))
    
    # Strategy 3: More random restarts with L-BFGS-B
    for restart in range(20):
        if not _check_deadline():
            break
        if time.time() - _get_start_time() > _DEADLINE - 5:
            break
        
        # Random initial guess
        random_theta = np.random.uniform(-np.pi, np.pi, (layers, n))
        random_params = random_theta.flatten()
        
        try:
            res = minimize(_objective, random_params, args=(n, coupling, field, layers, H, e0),
                           method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tight tolerance
    if _check_deadline():
        try:
            res = minimize(_objective, best_params, args=(n, coupling, field, layers, H, e0),
                           method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
