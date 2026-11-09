"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 250.0


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _cnot_op(control, target, qubits):
    size = 2 ** qubits
    op = np.eye(size, dtype=complex)
    for i in range(size):
        ctrl_bit = (i >> (qubits - 1 - control)) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << (qubits - 1 - target))
            op[i, i] = 0
            op[i, j] = 1
    return op


def _identify_known_gate(target, qubits):
    size = 2 ** qubits
    if np.allclose(target, np.eye(size), atol=1e-10):
        return []
    if qubits == 2:
        for ctrl in range(2):
            for tgt in range(2):
                if ctrl == tgt:
                    continue
                cnot = _cnot_op(ctrl, tgt, 2)
                overlap = np.trace(target.conj().T @ cnot) / size
                if abs(abs(overlap) - 1.0) < 1e-8:
                    return [["cx", ctrl, tgt]]
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        overlap = np.trace(target.conj().T @ swap) / size
        if abs(abs(overlap) - 1.0) < 1e-8:
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        overlap = np.trace(target.conj().T @ iswap) / size
        if abs(abs(overlap) - 1.0) < 1e-8:
            return [
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, -math.pi/2],
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["u3", 0, math.pi/2, 0, -math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2]
            ]
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        overlap = np.trace(target.conj().T @ toffoli) / 8
        if abs(abs(overlap) - 1.0) < 1e-8:
            return [
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, -math.pi/2],
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, -math.pi/2]
            ]
    return None


def _synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    known = _identify_known_gate(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            return known

    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    best = [0.0] * (3 * sum(1 for g in layout if g[0] == "u3"))
    best_distance = _distance(target, _build(circuit(best), qubits))

    # Try a few random restarts to find a better starting point
    for restart in range(3):
        trial = [rng.uniform(-math.pi, math.pi) for _ in best]
        d = _distance(target, _build(circuit(trial), qubits))
        if d < best_distance:
            best, best_distance = trial, d

    for sweep in range(15):
        if _remaining_time() < 5.0:
            break
        step = 1.0 / (1 + sweep)
        for i in range(len(best)):
            if _remaining_time() < 5.0:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    return circuit(best)


def synthesize(problem):
    return _synthesize(problem)
# EVOLVE-BLOCK-END
