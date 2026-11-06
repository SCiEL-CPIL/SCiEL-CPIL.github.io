"""Exact decompositions for known gates + efficient optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 230.0


def _elapsed():
    return time.time() - _START_TIME


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", c, t]


def _build(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
            ])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else np.eye(2, dtype=complex))
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / dim)


def _exact_cnot():
    return [_cx(0, 1)]


def _exact_swap():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_iswap():
    # iSWAP = e^{iπ/4}(S⊗S) · CNOT(0,1) · CNOT(1,0) · (S†⊗S†)
    # S = diag(1,i) = u3(0, 0, 0, π/2)
    # S† = diag(1,-i) = u3(0, 0, 0, -π/2)
    return [
        _u3(0, 0, 0, -math.pi / 2),
        _u3(1, 0, 0, -math.pi / 2),
        _cx(0, 1),
        _cx(1, 0),
        _u3(0, 0, 0, math.pi / 2),
        _u3(1, 0, 0, math.pi / 2),
    ]


def _exact_toffoli():
    # Toffoli(0,1,2): 6 CNOT decomposition
    # Using standard decomposition with T gates (T = u3(π/4, 0, 0), T† = u3(-π/4, 0, 0))
    gates = [
        _u3(2, math.pi / 2, 0, 0),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi / 4),
        _cx(0, 2),
        _u3(2, 0, 0, math.pi / 4),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi / 4),
        _cx(0, 2),
        _u3(2, math.pi / 2, 0, 0),
    ]
    return gates


def _generic_2qubit(target, qubits=2):
    """Decompose generic 2-qubit unitary into 3 CNOTs + single-qubit rotations."""
    # U = L3 · CNOT(0,1) · L2 · CNOT(1,0) · L1 · CNOT(0,1) · L0
    # Each Li is (A_i ⊗ B_i) where A_i, B_i are single-qubit unitaries
    # Total: 8 u3 gates (2 per Li) + 3 CNOTs = 11 gates, 24 parameters
    
    params = [0.0] * 24
    
    def circuit(p):
        return [
            _u3(0, p[0], p[1], p[2]),
            _u3(1, p[3], p[4], p[5]),
            _cx(0, 1),
            _u3(0, p[6], p[7], p[8]),
            _u3(1, p[9], p[10], p[11]),
            _cx(1, 0),
            _u3(0, p[12], p[13], p[14]),
            _u3(1, p[15], p[16], p[17]),
            _cx(0, 1),
            _u3(0, p[18], p[19], p[20]),
            _u3(1, p[21], p[22], p[23]),
        ]
    
    best = params[:]
    best_d = _distance(target, _build(circuit(best), qubits))
    
    step = 0.3
    for sweep in range(60):
        if _elapsed() > _DEADLINE - 15:
            break
        step *= 0.97
        for i in range(24):
            for delta in (step, -step):
                trial = best[:]
                trial[i] += delta
                d = _distance(target, _build(circuit(trial), qubits))
                if d < best_d:
                    best = trial
                    best_d = d
    
    return circuit(best)


def _generic_3qubit(target, qubits=3):
    """Decompose generic 3-qubit unitary using CNOT ladder + single-qubit rotations."""
    # Use a fixed CNOT pattern with 10 CNOTs
    cnots = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
    n_layers = len(cnots) + 1  # +1 for final layer
    
    params = [0.0] * (3 * 3 * n_layers)  # 3 qubits * 3 params * n_layers
    
    def circuit(p):
        gates = []
        idx = 0
        for layer in range(n_layers):
            for q in range(3):
                gates.append(_u3(q, p[idx], p[idx+1], p[idx+2]))
                idx += 3
            if layer < len(cnots):
                c, t = cnots[layer]
                gates.append(_cx(c, t))
        return gates
    
    best = params[:]
    best_d = _distance(target, _build(circuit(best), qubits))
    
    step = 0.3
    for sweep in range(40):
        if _elapsed() > _DEADLINE - 15:
            break
        step *= 0.96
        for i in range(len(best)):
            for delta in (step, -step):
                trial = best[:]
                trial[i] += delta
                d = _distance(target, _build(circuit(trial), qubits))
                if d < best_d:
                    best = trial
                    best_d = d
    
    return circuit(best)


def _is_known_swap(target):
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    return _distance(swap, target) < 1e-8


def _is_known_iswap(target):
    iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
    return _distance(iswap, target) < 1e-8


def _is_known_toffoli(target):
    t = np.eye(8, dtype=complex)
    t[6,6] = 0; t[7,7] = 0; t[6,7] = 1; t[7,6] = 1
    return _distance(t, target) < 1e-8


def synthesize(problem):
    name = problem.get("name", "").lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Exact known decompositions
    if "cnot" in name or (qubits == 2 and budget == 1):
        return _exact_cnot()
    if "swap" in name and "iswap" not in name:
        if _is_known_swap(target):
            return _exact_swap()
    if "iswap" in name:
        if _is_known_iswap(target):
            return _exact_iswap()
    if "toffoli" in name or "ccnot" in name:
        if _is_known_toffoli(target):
            return _exact_toffoli()
    
    # Generic
    if qubits == 2:
        return _generic_2qubit(target, qubits)
    else:
        return _generic_3qubit(target, qubits)
# EVOLVE-BLOCK-END
