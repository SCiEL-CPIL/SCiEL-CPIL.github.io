"""Rewritten program: vectorized circuit simulation + multi-restart L-BFGS-B optimization."""
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


def _ry_derivative_matrix(theta):
    """Return 2x2 derivative of Ry rotation matrix w.r.t theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return 0.5 * np.array([[-s, -c], [c, -s]], dtype=np.complex128)


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


def _apply_ry_derivative(state, theta, qubit, n):
    """Apply dRy/dtheta to qubit of state (2^n vector)."""
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_derivative_matrix(theta)
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


def _circuit_state(params, n, layers, derivative_index=-1):
    """
    Compute the state vector produced by the circuit.
    If derivative_index >= 0, the gate at that index (flattened) is replaced by its derivative.
    """
    thetas = params.reshape(layers, n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0  # |0>^n

    idx = 0
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            if idx == derivative_index:
                state = _apply_ry_derivative(state, theta, q, n)
            else:
                state = _apply_ry(state, theta, q, n)
            idx += 1

        # Apply CZ between adjacent pairs (ascending order), if not last layer
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)

    return state


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)

    for idx in range(dim):
        # Diagonal: Z_i Z_{i+1} terms
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag

        # Off-diagonal: X_i terms
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field

    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    """
    Compute energy and gradient of (E_circuit - E0)/n.
    params: flattened array of size layers*n
    """
    # 1. Compute main state
    psi = _circuit_state(params, n, layers)

    # 2. Compute H |psi>
    Hpsi = H @ psi

    # 3. Energy
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n

    # 4. Gradient
    # dE/dtheta_k = 2 Re < psi_k' | H | psi >
    # where psi_k' is the state where the k-th Ry is replaced by dRy/dtheta
    grad = np.zeros_like(params)
    num_params = params.size

    for k in range(num_params):
        # Compute derivative state
        psi_k = _circuit_state(params, n, layers, derivative_index=k)
        # Inner product <psi_k | Hpsi>
        val = np.vdot(psi_k, Hpsi)
        grad[k] = 2.0 * val.real / n  # Divide by n for the objective function

    return obj_val, grad


def _initial_guesses(n, coupling, field, layers, num_guesses=15):
    """Generate diverse initial guesses for optimization."""
    np.random.seed(42)
    guesses = []

    # 1. All zeros (|0> state)
    guesses.append(np.zeros(layers * n))

    # 2. All pi/2 (|+> state)
    guesses.append(np.full(layers * n, np.pi / 2))

    # 3. All pi/4
    guesses.append(np.full(layers * n, np.pi / 4))

    # 4. All -pi/4
    guesses.append(np.full(layers * n, -np.pi / 4))

    # 5. Interpolate between |0> and |+> based on field/coupling ratio
    # For small field, |0> is better; for large field, |+> is better
    ratio = field / coupling if coupling != 0 else 0
    interp_angle = np.pi / 2 * min(1.0, ratio / 2.0)
    guesses.append(np.full(layers * n, interp_angle))

    # 6. Layer-dependent: first layer small, later layers larger
    for layer_scale in [0.3, 0.5, 0.7]:
        g = np.zeros(layers * n)
        for l in range(layers):
            g[l * n:(l + 1) * n] = layer_scale * np.pi / 2 * (l + 1) / layers
        guesses.append(g)

    # 7-12. Random perturbations around various base angles
    for base_angle in [0, np.pi / 4, np.pi / 2, -np.pi / 4]:
        noise = np.random.randn(layers * n) * 0.5
        guesses.append(np.clip(base_angle + noise, -np.pi, np.pi))

    # 13-15. Pure random
    for _ in range(3):
        guesses.append(np.clip(np.random.randn(layers * n) * 0.8, -np.pi, np.pi))

    return guesses[:num_guesses]


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    # Get exact ground energy and Hamiltonian
    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)

    def objective_only(params):
        val, _ = _energy_and_grad(params, n, coupling, field, layers, H, e0)
        return val

    def gradient_only(params):
        _, grad = _energy_and_grad(params, n, coupling, field, layers, H, e0)
        return grad

    # Generate initial guesses
    restarts = _initial_guesses(n, coupling, field, layers, num_guesses=15)

    best_result = None
    best_val = np.inf

    for r, x_init in enumerate(restarts):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break

        x_init = np.clip(x_init, -np.pi, np.pi)

        try:
            # L-BFGS-B with analytical gradient
            res = opt.minimize(
                objective_only,
                x_init,
                jac=gradient_only,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()

        except Exception:
            pass

    if best_result is None:
        best_result = np.zeros(layers * n)

    theta = best_result.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        # Return fallback: |+> state
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
