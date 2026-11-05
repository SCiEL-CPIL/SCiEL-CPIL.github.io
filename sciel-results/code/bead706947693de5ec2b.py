"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_circuit(state, thetas, n, layers):
    """
    Apply the circuit to an initial |0...0> state.
    thetas: (layers, n) array of angles.
    Returns the final state vector of dimension 2**n.
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0...0>

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            ry = _ry_matrix(thetas[layer, i])
            # Apply Ry on qubit i
            new_state = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                # Determine if qubit i is 0 or 1 in this basis state
                bit = (idx >> i) & 1
                if bit == 0:
                    # qubit i is 0
                    idx1 = idx | (1 << i)  # flip bit i to 1
                    new_state[idx] += ry[0, 0] * state[idx] + ry[0, 1] * state[idx1]
                    new_state[idx1] += ry[1, 0] * state[idx] + ry[1, 1] * state[idx1]
                # We handle each pair (idx, idx1) exactly once by only processing when bit==0
            state = new_state

        # Apply CZ on every adjacent pair in ascending order, except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                # Apply CZ on qubits i and i+1
                # CZ is diagonal: |11> gets phase -1
                for idx in range(dim):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[idx] *= -1.0

    return state


def _energy(state, n, coupling, field):
    """
    Compute <H> for the state.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    e_zz = 0.0
    e_x = 0.0

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            # Z eigenvalue is +1 if bit=0, -1 if bit=1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz = z_i * z_ip1
            e_zz += zz * (np.abs(state[idx]) ** 2)

    # X_i terms: <X_i> = <0|_i X |1> + <1|_i X |0> = sum_{pairs} (c_0^* c_1 + c_1^* c_0)
    for i in range(n):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                idx1 = idx | (1 << i)
                e_x += 2.0 * np.real(np.conj(state[idx]) * state[idx1])

    return -coupling * e_zz - field * e_x


def _objective(params, n, coupling, field, layers):
    thetas = params.reshape(layers, n)
    state = _apply_circuit(np.zeros(2**n, dtype=np.complex128), thetas, n, layers)
    # Reset state[0] = 1 is done inside _apply_circuit, but we pass a zero array.
    # Wait, _apply_circuit sets state[0]=1.0 internally.
    e = _energy(state, n, coupling, field)
    return e


def _exact_ground_energy(n, coupling, field):
    """
    Compute exact ground state energy by diagonalizing H.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)

    # ZZ terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * z_i * z_ip1

    # X terms
    for i in range(n):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                idx1 = idx | (1 << i)
                H[idx, idx1] += -field
                H[idx1, idx] += -field

    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    # Precompute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)

    # Objective: minimize (E_circuit - E0) / n
    # We just minimize E_circuit, since E0 is constant.
    # But the score is based on mean gap. Minimizing E_circuit is equivalent for fixed E0.
    
    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # Penalty
        thetas = params.reshape(layers, n)
        state = _apply_circuit(np.zeros(2**n, dtype=np.complex128), thetas, n, layers)
        e = _energy(state, n, coupling, field)
        return e

    # Initial guess: all pi/2 (product |+>)
    x0 = np.full((layers, n), np.pi / 2.0)
    x0_flat = x0.flatten()

    best_params = x0_flat
    best_val = obj(best_params)

    # Try a few random restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
        
        if r == 0:
            x_start = x0_flat
        else:
            x_start = x0_flat + np.random.randn(layers * n) * 0.5

        try:
            res = minimize(
                obj,
                x_start,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
