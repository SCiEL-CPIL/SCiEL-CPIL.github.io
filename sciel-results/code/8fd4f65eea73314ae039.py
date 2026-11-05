# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import diags, kron
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build sparse Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # Build H term by term using kron
    H = np.zeros((dim, dim), dtype=complex)

    # ZZ terms: -coupling * Z_i ⊗ Z_{i+1}
    for i in range(n - 1):
        # Z on qubit i, Z on qubit i+1, I elsewhere
        # qubit 0 is rightmost (LSB), qubit n-1 is leftmost (MSB)
        # In kron order: qubit n-1 is first (leftmost), qubit 0 is last (rightmost)
        # So for qubit i (0-indexed from right), its position in kron is (n-1-i)
        # Actually let's use: kron(A, B) where A acts on leftmost qubit
        # Qubit i (bit i) corresponds to position (n-1-i) in the kron chain
        ops = []
        for q in range(n - 1, -1, -1):
            if q == i or q == i + 1:
                ops.append(Z)
            else:
                ops.append(I)
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += -coupling * term

    # X terms: -field * X_i
    for i in range(n):
        ops = []
        for q in range(n - 1, -1, -1):
            if q == i:
                ops.append(X)
            else:
                ops.append(I)
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += -field * term

    return H

def _apply_ry(state, qubit, theta, n):
    """Apply Ry(theta) to qubit in state vector (qubit 0 = LSB)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]]
    # For qubit q (0-indexed from right), we need to reshape and apply
    # State is indexed by |q_{n-1} ... q_1 q_0> where q_0 is LSB
    # Reshape to (2,)*n, then apply on axis (n-1-q)
    new_state = np.zeros_like(state)
    for idx in range(1 << n):
        bits = [(idx >> b) & 1 for b in range(n)]
        # Apply Ry to qubit `qubit`
        b = bits[qubit]
        # Ry matrix element: <0| = c, <1| = s for input |0>; <0| = -s, <1| = c for input |1>
        if b == 0:
            # Output bit 0: amplitude c
            idx0 = idx
            new_state[idx0] += state[idx] * c
            # Output bit 1: amplitude s
            idx1 = idx | (1 << qubit)
            new_state[idx1] += state[idx] * s
        else:
            # Output bit 0: amplitude -s
            idx0 = idx & ~(1 << qubit)
            new_state[idx0] += state[idx] * (-s)
            # Output bit 1: amplitude c
            idx1 = idx
            new_state[idx1] += state[idx] * c
    return new_state

def _apply_cz(state, q1, q2, n):
    """Apply CZ to qubits q1 and q2 (0-indexed from right)."""
    # CZ = diag(1,1,1,-1) in basis |00>,|01>,|10>,|11>
    # Flip sign when both bits are 1
    new_state = state.copy()
    mask = (1 << q1) | (1 << q2)
    for idx in range(1 << n):
        if (idx & mask) == mask:
            new_state[idx] = -state[idx]
    return new_state

def _circuit_state(params, n, layers):
    """Simulate the circuit: start |0>^n, apply layers of Ry then CZ."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0  # |0>^n

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            theta = params[layer, q]
            state = _apply_ry(state, q, theta, n)
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)

    return state

def _energy(params, n, coupling, field, H, layers):
    """Compute energy expectation value of circuit state."""
    state = _circuit_state(params, n, layers)
    # <psi|H|psi>
    Hpsi = H @ state
    energy = np.vdot(state, Hpsi).real
    return energy

def _objective(params_flat, n, coupling, field, H, ground_energy, layers):
    """Objective: mean energy gap per spin = (E_circuit - E_ground) / n."""
    params = params_flat.reshape(layers, n)
    e = _energy(params, n, coupling, field, H, layers)
    return (e - ground_energy) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)

    # Get exact ground energy
    dim = 1 << n
    if dim <= 64:
        # Use dense eigensolver for small systems
        eigs = np.linalg.eigvalsh(H)
        ground_energy = eigs[0].real
    else:
        ground_energy = eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]

    # Initial guess: try a few random starts and pick the best
    best_params = np.zeros((layers, n))
    best_obj = np.inf

    # Start with a reasonable initial guess
    # For transverse field Ising, the ground state is a superposition
    # Try theta = pi/2 for all (maximally mixed in X basis)
    initial_guesses = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.full((layers, n), np.pi / 4),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
    ]

    for guess in initial_guesses:
        if time.time() > deadline:
            break
        obj_val = _objective(guess.ravel(), n, coupling, field, H, ground_energy, layers)
        if obj_val < best_obj:
            best_obj = obj_val
            best_params = guess.copy()

    # Optimize from the best initial guess
    result = minimize(
        _objective,
        best_params.ravel(),
        args=(n, coupling, field, H, ground_energy, layers),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )

    if result.fun < best_obj:
        best_params = result.x.reshape(layers, n)
        best_obj = result.fun

    # Try a second optimization from a different random start if time allows
    if time.time() < deadline - 5.0:
        guess2 = np.random.uniform(-np.pi, np.pi, (layers, n))
        obj2 = _objective(guess2.ravel(), n, coupling, field, H, ground_energy, layers)
        if obj2 < best_obj:
            result2 = minimize(
                _objective,
                guess2.ravel(),
                args=(n, coupling, field, H, ground_energy, layers),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result2.fun < best_obj:
                best_params = result2.x.reshape(layers, n)
                best_obj = result2.fun

    # Ensure output is finite
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=np.pi, neginf=-np.pi)

    return best_params
# EVOLVE-BLOCK-END
