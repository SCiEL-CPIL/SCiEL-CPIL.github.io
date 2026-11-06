"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of state (2^n vector). Qubit i is bit i (LSB)."""
    if abs(theta) < 1e-15:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_matrix(theta)
    state_moved = np.moveaxis(state_r, qubit, 0)
    state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
    state_moved_back = np.moveaxis(state_out, 0, qubit)
    return state_moved_back.reshape(-1)


def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2. Qubit i is bit i (LSB)."""
    if q1 == q2:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, [q1, q2], [-2, -1])
    factor = np.ones(state_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    state_moved = state_moved * factor
    state_moved_back = np.moveaxis(state_moved, [-2, -1], [q1, q2])
    return state_moved_back.reshape(-1)


def _circuit_state(params, n, layers):
    """Compute the state vector produced by the circuit."""
    thetas = params.reshape(layers, n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, thetas[layer, q], q, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
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


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _circuit_energy(params, n, coupling, field, layers, H=None):
    """Compute expectation value of H for given circuit parameters."""
    if H is None:
        H = _build_hamiltonian(n, coupling, field)
    state = _circuit_state(params, n, layers)
    energy = np.vdot(state, H @ state)
    return np.real(energy)


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    def objective(params):
        e_circ = _circuit_energy(params, n, coupling, field, layers, H)
        return (e_circ - e0) / n
    
    # Initial guesses
    x0 = np.zeros(layers * n)
    x0[-n:] = np.pi / 2  # last layer: |+>
    
    best_result = None
    best_val = np.inf
    
    np.random.seed(42)
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        if r == 0:
            x_init = x0.copy()
        elif r == 1:
            x_init = np.zeros(layers * n)
            x_init[:] = np.pi / 4
        elif r == 2:
            x_init = np.zeros(layers * n)
            x_init[:] = np.pi / 8
        elif r == 3:
            x_init = np.zeros(layers * n)
            x_init[:] = -np.pi / 4
        else:
            x_init = x0 + np.random.randn(layers * n) * 0.5
            x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(objective, x_init, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                       'adaptive': True})
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        best_result = x0
    
    # Try L-BFGS-B with the best result as starting point if time allows
    if time.time() - _start_time < _TIME_LIMIT - 20:
        try:
            res2 = opt.minimize(objective, best_result, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass
    
    theta = best_result.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
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
