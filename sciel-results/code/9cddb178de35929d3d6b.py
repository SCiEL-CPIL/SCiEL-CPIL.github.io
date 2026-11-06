"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE_FRACTION = 0.92
_MAX_GATE_COUNT = 400


def _now():
    return time.time()


def _target(problem):
    return (
        np.array(problem["real"], dtype=complex)
        + 1j * np.array(problem["imaginary"], dtype=complex)
    )


def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


def _kron_chain_single(qubit_idx, single, qubits):
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, single if k == qubit_idx else eye)
    return op


def _apply_u3(matrix, qubit_idx, theta, phi, lam, qubits):
    single = _u3_matrix(theta, phi, lam)
    op = _kron_chain_single(qubit_idx, single, qubits)
    return op @ matrix


def _apply_cnot(matrix, control, target, qubits):
    n = 2 ** qubits
    idx = np.arange(n)
    ctrl_bit = (idx >> (qubits - 1 - control)) & 1
    mask = ctrl_bit == 1
    perm = idx.copy()
    swap_indices = idx[mask]
    new_indices = (swap_indices & ~(1 << (qubits - 1 - target))) | (
        (swap_indices ^ (1 << (qubits - 1 - target)))
    )
    perm[mask] = new_indices
    return matrix[perm, :]


def _build(gates, qubits):
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            matrix = _apply_cnot(matrix, control, target, qubits)
    return matrix


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass, efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.zeros_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _match_target(target, circuit, qubits, tol=1e-8):
    dist = _build_fast(circuit, qubits, target)
    return dist < tol


def _try_known_circuits(target, qubits, budget):
    """Try hardcoded exact circuits for standard gates."""
    if qubits == 2:
        # CNOT
        if budget >= 1:
            for ctrl, tgt in [(0, 1), (1, 0)]:
                circuit = [_cx(ctrl, tgt)]
                if _match_target(target, circuit, 2):
                    return circuit

        # SWAP
        if budget >= 3:
            circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
            if _match_target(target, circuit, 2):
                return circuit

        # iSWAP
        if budget >= 2:
            circuit = [
                _u3(0, math.pi/2, 0, math.pi),
                _u3(1, math.pi/2, 0, math.pi),
                _cx(0, 1),
                _u3(0, 0, 0, -math.pi/2),
                _u3(1, 0, 0, math.pi/2),
                _cx(1, 0),
                _u3(0, math.pi/2, 0, math.pi),
                _u3(1, math.pi/2, 0, math.pi),
            ]
            if _match_target(target, circuit, 2):
                return circuit

    if qubits == 3:
        # Toffoli: try all 6 permutations of (ctrl1, ctrl2, target)
        if budget >= 6:
            perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            for c1, c2, tgt in perms:
                circuit = [
                    _u3(tgt, 0, 0, math.pi/2),
                    _cx(c2, tgt),
                    _u3(tgt, 0, 0, -math.pi/4),
                    _cx(c1, tgt),
                    _u3(tgt, 0, 0, math.pi/4),
                    _cx(c2, tgt),
                    _u3(tgt, 0, 0, -math.pi/4),
                    _cx(c1, tgt),
                    _u3(tgt, 0, 0, math.pi/4),
                    _u3(tgt, 0, 0, -math.pi/2),
                    _u3(c2, 0, 0, math.pi/2),
                    _cx(c1, c2),
                    _u3(c2, 0, 0, -math.pi/4),
                    _cx(c1, c2),
                    _u3(c2, 0, 0, math.pi/4),
                    _u3(c2, 0, 0, -math.pi/2),
                ]
                if _match_target(target, circuit, 3):
                    return circuit

    return None


def _generic_synthesize(problem, deadline):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits

    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]

    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3

    rng = random.Random(42 + qubits * 100 + budget)

    def circuit_from_params(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    best_params = [0.0] * n_params
    best_dist = _build_fast(circuit_from_params(best_params), qubits, target)

    # Coordinate descent with Nelder-Mead refinement on stuck parameters
    sweep = 0
    max_sweeps = 50
    while sweep < max_sweeps:
        if _now() > deadline:
            break
        step = 2.0 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if _now() > deadline:
                break
            best_i = best_params[i]
            best_val = best_dist
            for trial in (
                best_i + step,
                best_i - step,
                best_i + rng.uniform(-math.pi, math.pi),
                best_i + rng.uniform(-math.pi, math.pi),
            ):
                params = best_params[:]
                params[i] = trial
                dist = _build_fast(circuit_from_params(params), qubits, target)
                if dist < best_val:
                    best_val = dist
                    best_i = trial
            if best_i != best_params[i]:
                improved = True
            best_params[i] = best_i
            best_dist = best_val
        sweep += 1
        if not improved and sweep > 5:
            # Try a few random restarts with small perturbations
            for _ in range(3):
                if _now() > deadline:
                    break
                perturbed = [p + rng.gauss(0, 0.1) for p in best_params]
                dist = _build_fast(circuit_from_params(perturbed), qubits, target)
                if dist < best_dist:
                    best_params = perturbed
                    best_dist = dist

    return circuit_from_params(best_params)


def synthesize(problem):
    start_time = _now()
    deadline = start_time + 270 * _DEADLINE_FRACTION

    exact = _try_known_circuits(_target(problem), problem["qubits"], problem["cnot_budget"])
    if exact is not None and len(exact) <= _MAX_GATE_COUNT:
        cnot_count = sum(1 for g in exact if g[0] == "cx")
        if cnot_count <= problem["cnot_budget"]:
            return exact

    return _generic_synthesize(problem, deadline)
# EVOLVE-BLOCK-END
