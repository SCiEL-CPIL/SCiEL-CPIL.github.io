"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Global time tracking
_START_TIME = time.time()
TIME_LIMIT = 240  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator by kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(name):
    """Check if the target is a known gate we can synthesize exactly."""
    return name in ("CNOT", "iSWAP", "SWAP", "Toffoli")


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate: 1 CNOT + single qubit rotations."""
    # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    # This can be done with 1 CNOT and single-qubit rotations
    # Standard decomposition:
    # iSWAP = (H on both) * (CNOT) * (S on both) * (CNOT) * (H on both) -- no, that's SWAP
    # Actually iSWAP needs 2 CNOTs in general, but let's check the budget
    # For 2 qubits, iSWAP can be done with 2 CNOTs
    # Let me use a standard decomposition
    # iSWAP = exp(i*pi/4 * (X⊗X + Y⊗Y))
    # One known decomposition uses 2 CNOTs
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, math.pi/2, 0],
        ["u3", 1, math.pi/2, math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
    # Toffoli |cc> -> |t>
    # Standard decomposition uses 6 CNOTs
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]


def _generic_synthesis(problem):
    """Generic synthesis using coordinate descent with better initialization."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Build a layout with CNOTs and single-qubit gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
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
    
    # Initialize with zeros
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with multiple sweeps
    max_sweeps = 30
    for sweep in range(max_sweeps):
        if _time_remaining() < 10:
            break
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if _time_remaining() < 10:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved:
            break
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Try to use known exact syntheses first
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        # Generic synthesis
        gates = _generic_synthesis(problem)
    
    # Verify the result
    target = _target(problem)
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    # If the known synthesis didn't work well, fall back to generic
    if dist > 0.01:
        gates = _generic_synthesis(problem)
    
    return gates
# EVOLVE-BLOCK-END
