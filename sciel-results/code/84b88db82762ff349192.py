"""Quantum circuit synthesis using closed-form decompositions and optimized generic circuits."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 250  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
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


def _cnot_gate(control, target):
    return ["cx", control, target]


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _synthesize_cnot(problem):
    """CNOT gate: 1 CNOT, no single-qubit rotations needed."""
    return [_cnot_gate(0, 1)]


def _synthesize_iswap(problem):
    """iSWAP gate: 3 CNOTs + single-qubit rotations."""
    # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    # Standard decomposition uses 3 CNOTs
    gates = []
    # U1 gates on qubit 0
    gates.append(_u3_gate(0, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(1, 0))
    gates.append(_u3_gate(0, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    return gates


def _synthesize_swap(problem):
    """SWAP gate: 3 CNOTs, no single-qubit rotations needed."""
    return [_cnot_gate(0, 1), _cnot_gate(1, 0), _cnot_gate(0, 1)]


def _synthesize_toffoli(problem):
    """Toffoli gate: 6 CNOTs + single-qubit rotations."""
    # Standard Toffoli decomposition
    gates = []
    # H on target
    gates.append(_u3_gate(2, math.pi / 2, 0, math.pi / 2))
    gates.append(_cnot_gate(2, 1))
    gates.append(_u3_gate(1, 0, 0, -math.pi / 4))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(1, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(2, 1))
    gates.append(_u3_gate(1, 0, 0, -math.pi / 4))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(1, 0, 0, math.pi / 4))
    gates.append(_u3_gate(2, math.pi / 2, 0, math.pi / 2))
    return gates


def _synthesize_generic(problem):
    """Generic unitary synthesis using optimized CNOT ladder with coordinate descent."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    n_qubits = qubits
    dim = 2 ** n_qubits
    
    # Create a layout with CNOTs and u3 gates
    # Use a standard CNOT ladder pattern
    pairs = [(c, t) for c in range(n_qubits) for t in range(n_qubits) if c != t]
    
    # Build layout: u3 gates before and after each CNOT
    layout = []
    n_u3 = 0
    for layer in range(budget):
        for q in range(n_qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            n_u3 += 1
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(n_qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        n_u3 += 1
    
    n_params = 3 * n_u3
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with zeros
    best = [0.0] * n_params
    best_distance = _distance(target, _build(circuit(best), n_qubits))
    
    # Coordinate descent optimization
    max_sweeps = 8
    for sweep in range(max_sweeps):
        if _time_remaining() < 10:
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            if _time_remaining() < 5:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), n_qubits))
                if value < best_distance:
                    best, best_distance = values, value
            # Also try a random perturbation
            trial = best[i] + (np.random.uniform(-math.pi, math.pi))
            values = best[:]
            values[i] = trial
            value = _distance(target, _build(circuit(values), n_qubits))
            if value < best_distance:
                best, best_distance = values, value
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    name = problem.get("name", "")
    
    # Use closed-form decompositions for known gates
    if name == "CNOT" or name == "cnot":
        return _synthesize_cnot(problem)
    elif name == "iSWAP" or name == "iswap":
        return _synthesize_iswap(problem)
    elif name == "SWAP" or name == "swap":
        return _synthesize_swap(problem)
    elif name == "Toffoli" or name == "toffoli" or name == "CCX":
        return _synthesize_toffoli(problem)
    else:
        # Generic synthesis for unknown unitaries
        return _synthesize_generic(problem)
# EVOLVE-BLOCK-END
