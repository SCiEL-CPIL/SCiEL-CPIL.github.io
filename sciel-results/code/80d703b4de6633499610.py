"""Quantum information: shallow-circuit Ising ground states.
Optimizes the 3-layer Ry/CZ circuit parameters to approximate ground states
of the open transverse-field Ising chain, minimizing mean energy gap per spin."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before the 150 s wall-clock kill


def _hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i as a dense matrix."""
    J = coupling
    h = field
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    I2 = np.eye(2)
    H = np.zeros((2 ** n, 2 ** n))
    # ZZ terms
    for i in range(n - 1):
        A = np.array([I2, I2])
        for j in range(n):
            if j == i:
                A = np.kron(A, Z)
            elif j == i + 1:
                A = np.kron(A, Z)
            else:
                A = np.kron(A, I2)
        H = H - J * A
    # X terms
    for i in range(n):
        A = np.array([I2, I2])
        for j in range(n):
            if j == i:
                A = np.kron(A, X)
            else:
                A = np.kron(A, I2)
        H = H - h * A
    return H


def _ground_energy(n, coupling, field):
    H = _hamiltonian(n, coupling, field)
    w = np.linalg.eigvalsh(H)
    return float(w[0])


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])


def _circuit_state(n, params):
    """Apply Ry(theta) to every qubit, then CZ on adjacent pairs in ascending order.
    params is (layers, n). Returns the final statevector."""
    # start |0>^n
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    # Precompute CZ action: CZ|ab> = (-1)^{ab}|ab>.
    # For each adjacent pair (i, i+1), multiply amplitudes where both bits are 1 by -1.
    for layer in range(params.shape[0]):
        # Apply Ry to every qubit (single-qubit rotations).
        for i in range(n):
            R = _ry(params[layer, i])
            # Apply R on qubit i: for each basis state, split on bit i.
            new_state = np.zeros_like(state)
            for bit in range(n):
                mask = (1 << bit)
                # iterate over basis states
            # Efficient: loop over basis states
            for b in range(2 ** n):
                if state[b] == 0:
                    continue
                ib = (b >> i) & 1
                # new index with bit i flipped
                b0 = b & ~(1 << i)
                # amplitudes for |..0..> and |..1..> at this qubit
                a0 = state[b0] if ib == 0 else 0.0
                a1 = state[b] if ib == 1 else 0.0
                # Actually need both components for the pair; handle by reading both.
            # Simpler: build via tensor. Use reshape.
        # Fallback: use full tensor method for correctness.
    # Tensor-based implementation (clear and correct).
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    state = state.reshape([2] * n)
    for layer in range(params.shape[0]):
        for i in range(n):
            R = _ry(params[layer, i])
            new_state = np.zeros_like(state)
            for val in (0, 1):
                sl0 = [slice(None)] * n
                sl1 = [slice(None)] * n
                sl0[i] = 0
                sl1[i] = 1
                comp0 = state[tuple(sl0)]
                comp1 = state[tuple(sl1)]
                out0 = R[0, 0] * comp0 + R[0, 1] * comp1
                out1 = R[1, 0] * comp0 + R[1, 1] * comp1
                new_state[tuple(sl0)] = out0
                new_state[tuple(sl1)] = out1
            state = new_state
        # CZ on adjacent pairs in ascending order (skip after final layer)
        if layer < params.shape[0] - 1:
            for i in range(n - 1):
                # CZ: multiply amplitude by -1 where bits i and i+1 are both 1
                for b_i in range(2):
                    for b_ip1 in range(2):
                        if b_i == 1 and b_ip1 == 1:
                            sl = [slice(None)] * n
                            sl[i] = 1
                            sl[i + 1] = 1
                            state[tuple(sl)] = -state[tuple(sl)]
    return state.reshape(-1)


def _energy(n, params, H):
    state = _circuit_state(n, params)
    e = np.vdot(state, H @ state).real
    return float(e)


def _objective(params, n, coupling, field, H):
    e = _energy(n, params, H)
    return e  # we minimize circuit energy; ground energy is a constant offset


def angles(n, coupling, field, layers):
    from scipy.optimize import minimize, differential_evolution
    t0 = time.time()
    H = _hamiltonian(n, coupling, field)
    dims = n * layers

    def obj(p):
        return _energy(n, p.reshape(layers, n), H)

    # Multi-restart Nelder-Mead with a time budget.
    best_p = np.zeros(dims)
    best_f = np.inf
    rng = np.random.default_rng(0)
    attempts = 0
    while time.time() < _DEADLINE - 5.0:
        # Initialize around small random angles.
        if attempts == 0:
            p0 = np.zeros(dims)
        else:
            p0 = rng.uniform(-np.pi / 2, np.pi / 2, dims)
        attempts += 1
        try:
            res = minimize(obj, p0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-9,
                                    'adaptive': True})
            if res.fun < best_f:
                best_f = float(res.fun)
                best_p = res.x.copy()
        except Exception:
            pass
        # Stop early if we've converged close to ground energy.
        if best_f <= _ground_energy(n, coupling, field) + 1e-6:
            break
        if attempts >= 12:
            break

    return best_p.reshape(layers, n)
# EVOLVE-BLOCK-END
