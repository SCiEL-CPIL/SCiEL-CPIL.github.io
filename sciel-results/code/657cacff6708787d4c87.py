"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _circuit_state(params, n, layers):
    """Compute the state vector produced by the circuit (vectorized)."""
    thetas = params.reshape(layers, n)
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    shape = [2] * n

    for layer in range(layers):
        # Apply all Ry gates for this layer
        for q in range(n):
            theta = thetas[layer, q]
            if abs(theta) < 1e-15:
                continue
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            state_r = state.reshape(shape)
            # Move qubit axis to front
            state_moved = np.moveaxis(state_r, q, 0)
            # Apply: new[0] = c*old[0] - s*old[1], new[1] = s*old[0] + c*old[1]
            new_state = np.empty_like(state_moved)
            new_state[0] = c * state_moved[0] - s * state_moved[1]
            new_state[1] = s * state_moved[0] + c * state_moved[1]
            state = np.moveaxis(new_state, 0, q).reshape(-1)

        # Apply CZ gates between adjacent pairs
        if layer < layers - 1:
            state_r = state.reshape(shape)
            for q in range(n - 1):
                # CZ: flip sign of |11> component for qubits q, q+1
                # Create mask where both qubits are 1
                mask = np.ones(shape, dtype=np.float64)
                mask[(slice(None),) * q + (1,) + (slice(None),) * (n - q - 2) + (1,)] = -1.0
                # Need to apply mask correctly: select sub-array where qubit q=1 and qubit q+1=1
                # Use slicing: state_r[..., idx_q=1, ..., idx_q+1=1, ...]
                # Build index tuple
                idx = [slice(None)] * n
                idx[q] = 1
                idx[q + 1] = 1
                state_r[tuple(idx)] *= -1.0
            state = state_r.reshape(-1)

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
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() - _start_time > _TIME_LIMIT - 5:
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
        elif r == 4:
            x_init = np.zeros(layers * n)
            x_init[:n] = np.pi / 6
            x_init[n:2*n] = np.pi / 3
        elif r == 5:
            x_init = np.zeros(layers * n)
            x_init[:n] = -np.pi / 6
            x_init[n:2*n] = np.pi / 3
        else:
            x_init = x0 + np.random.randn(layers * n) * 0.5
            x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(objective, x_init, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                       'adaptive': True})
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        best_result = x0
    
    # Final high-precision polish pass with L-BFGS-B
    if time.time() - _start_time < _TIME_LIMIT - 3:
        try:
            res2 = opt.minimize(objective, best_result, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
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
