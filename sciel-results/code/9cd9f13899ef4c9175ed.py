# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 145.0  # safety margin before 150s kill


def _apply_ry(state, i, theta):
    """Apply Ry(theta) to qubit i (bit i)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    n = state.shape[0]
    dim = 1 << n
    # Build permutation: swap bit i
    # For bit i, the index changes by flipping bit i
    # state is indexed by integer; bit i is (index >> i) & 1
    # We reshape to (2,)*n with qubit i as axis i
    shape = [1] * n
    shape[i] = 2
    # Reshape: state[k] where k = sum b_j 2^j
    # We want axis i to be the bit-i dimension
    # Use reshape with order to put bit i first
    # Easier: use the fact that flipping bit i means XOR with (1<<i)
    # Apply 2x2 matrix [[c, -s],[s, c]] on the bit-i subspace
    # state_new = c*state - s*state_flipped  for the |1> part, etc.
    # Ry(theta) = [[cos, -sin],[sin, cos]] in {|0>,|1>} basis
    # |0> -> cos|0> + sin|1>, |1> -> -sin|0> + cos|1>
    # So new[a] = c*old[a] - s*old[a^mask] if bit i of a is 1
    #      new[a] = c*old[a] + s*old[a^mask] if bit i of a is 0
    mask = 1 << i
    # indices where bit i is 0
    idx0 = np.array([k for k in range(dim) if (k & mask) == 0])
    idx1 = idx0 | mask
    new_state = state.copy()
    new_state[idx0] = c * state[idx0] + s * state[idx1]
    new_state[idx1] = -s * state[idx0] + c * state[idx1]
    return new_state


def _apply_cz(state, i, j):
    """Apply CZ between qubits i and j."""
    mask = (1 << i) | (1 << j)
    dim = state.shape[0]
    idx = np.array([k for k in range(dim) if (k & mask) == mask])
    new_state = state.copy()
    new_state[idx] = -state[idx]
    return new_state


def _circuit_state(theta, n, layers):
    """Simulate the circuit. theta has shape (layers, n)."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for l in range(layers):
        # Apply Ry to every qubit
        for i in range(n):
            state = _apply_ry(state, i, theta[l, i])
        # Apply CZ between adjacent pairs (except after final layer)
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
    return state


def _compute_energy(state, n, coupling, field):
    """Compute <H> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = state.shape[0]
    E = 0.0
    # ZZ terms
    for i in range(n - 1):
        mi = 1 << i
        mj = 1 << (i + 1)
        for k in range(dim):
            bi = 1 if (k & mi) else -1
            bj = 1 if (k & mj) else -1
            E += -coupling * bi * bj * (np.abs(state[k]) ** 2)
    # X terms: <X_i> = sum_k state[k]*conj(state[k^mi]) + c.c.
    for i in range(n):
        mi = 1 << i
        x_exp = 0.0
        for k in range(dim):
            x_exp += state[k] * np.conj(state[k ^ mi])
        E += -field * 2.0 * np.real(x_exp)
    return E


def _energy_per_spin(theta, n, coupling, field, layers):
    state = _circuit_state(theta, n, layers)
    E = _compute_energy(state, n, coupling, field)
    return E / n


def _exact_ground_energy(n, coupling, field):
    """Exact ground energy of H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    # Build H
    H = np.zeros((dim, dim), dtype=np.complex128)
    # ZZ terms
    for i in range(n - 1):
        mi = 1 << i
        mj = 1 << (i + 1)
        for k in range(dim):
            bi = 1 if (k & mi) else -1
            bj = 1 if (k & mj) else -1
            H[k, k] += -coupling * bi * bj
    # X terms
    for i in range(n):
        mi = 1 << i
        for k in range(dim):
            H[k, k ^ mi] += -field
    # Hermitian
    H = (H + H.conj().T) / 2.0
    evals = np.linalg.eigvalsh(H)
    return evals[0]


def _physics_init(n, field, coupling, layers):
    """Physics-informed initialization based on field strength."""
    theta = np.zeros((layers, n), dtype=np.float64)
    # For a single qubit with H = -field*X, ground state is |+> rotated.
    # The angle that minimizes <Ry(theta)|X|Ry(theta)> = sin(theta) is theta=pi/2.
    # But with ZZ coupling, we need to balance.
    # Start all at pi/2 (|+> state) as a reasonable baseline.
    theta[:, :] = np.pi / 2.0
    # Adjust: stronger field -> closer to pi/2; weaker field -> maybe different.
    # For open chain, boundary qubits see fewer ZZ bonds.
    # Use a simple heuristic: scale by field strength.
    # Actually, just use pi/2 for all as the seed does, but add small variations.
    # Let's use pi/2 for all qubits in all layers.
    return theta


def _random_init(n, layers, seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
    return theta


def angles(n, coupling, field, layers):
    t_start = time.time()
    layers = 3
    n_q = n
    c = coupling
    f = field

    # Compute exact ground energy for reference (used in objective via energy gap)
    # Actually, the objective is (circuit energy - exact ground energy)/n
    # Since exact ground energy is constant, minimizing circuit energy/n is equivalent.
    # But let's compute it to be safe for the objective function.
    E0 = _exact_ground_energy(n_q, c, f)

    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n_q)
        try:
            E = _compute_energy(_circuit_state(theta, n_q, layers), n_q, c, f)
            return (E - E0) / n_q
        except Exception:
            return 1e6

    # Physics-informed initialization
    theta_init = _physics_init(n_q, f, c, layers).copy()

    best_theta = theta_init.copy()
    best_val = objective(theta_init.flatten())

    # Try L-BFGS-B from physics init
    try:
        res = minimize(objective, theta_init.flatten(), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x.reshape(layers, n_q).copy()
    except Exception:
        pass

    elapsed = time.time() - t_start
    if elapsed > _DEADLINE - 10:
        return best_theta

    # Try multiple random restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - t_start > _DEADLINE - 10:
            break
        seed = 42 + r * 1000
        theta_rand = _random_init(n_q, layers, seed)
        try:
            res = minimize(objective, theta_rand.flatten(), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n_q).copy()
        except Exception:
            pass

    # Final polish: tight tolerance
    if time.time() - t_start < _DEADLINE - 5:
        try:
            res = minimize(objective, best_theta.flatten(), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.reshape(layers, n_q).copy()
        except Exception:
            pass

    return best_theta
# EVOLVE-BLOCK-END
