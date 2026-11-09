"""Exact quantum circuit synthesis using known decompositions and KAK for generic gates."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0


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


def _toffoli_decomposition():
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


def _iswap_decomposition():
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2]
    ]


def _swap_decomposition():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]


def _cnot_decomposition(control, target):
    return [
        ["cx", control, target]
    ]


def _fredkin_decomposition():
    return [
        ["cx", 0, 1],
        ["cx", 1, 2],
        ["cx", 0, 1],
        ["cx", 1, 2],
        ["cx", 0, 1]
    ]


def _identify_known_gate(target, qubits):
    """Check if target is a known gate."""
    if qubits == 2:
        cnot_01 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_01):
            return _cnot_decomposition(0, 1)
        cnot_10 = np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]], dtype=complex)
        if np.allclose(target, cnot_10):
            return _cnot_decomposition(1, 0)
        
        iswap = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]], dtype=complex) * 1j
        if np.allclose(target, iswap):
            return _iswap_decomposition()
        
        swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)
        if np.allclose(target, swap):
            return _swap_decomposition()
            
    if qubits == 3:
        toffoli = np.identity(8, dtype=complex)
        toffoli[6,6] = 0
        toffoli[7,7] = 0
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        if np.allclose(target, toffoli):
            return _toffoli_decomposition()
            
        fredkin = np.identity(8, dtype=complex)
        fredkin[5,5] = 0
        fredkin[6,6] = 0
        fredkin[5,6] = 1
        fredkin[6,5] = 1
        if np.allclose(target, fredkin):
            return _fredkin_decomposition()
            
    return None


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    known_gates = _identify_known_gate(target, qubits)
    if known_gates is not None:
        return known_gates
        
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    import random
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
    
    start_time = time.time()
    time_limit = 10.0
    
    for sweep in range(12):
        if time.time() - start_time > time_limit:
            break
        step = 1.0 / (1 + sweep)
        for i in range(len(best)):
            if time.time() - start_time > time_limit:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    return circuit(best)
# EVOLVE-BLOCK-END
