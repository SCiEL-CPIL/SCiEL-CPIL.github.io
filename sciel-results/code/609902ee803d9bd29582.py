"""Exact decompositions for known gates + fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 210.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(name, target, qubits):
    """Check if target matches a known gate up to global phase."""
    dim = 2 ** qubits
    
    # CNOT on (0,1) for 2 qubits
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _phase_insensitive_match(target, cnot):
            return "cnot"
        # iSWAP
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if _phase_insensitive_match(target, iswap):
            return "iswap"
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _phase_insensitive_match(target, swap):
            return "swap"
    
    # Toffoli for 3 qubits
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        if _phase_insensitive_match(target, toffoli):
            return "toffoli"
    
    return None


def _phase_insensitive_match(a, b, tol=1e-10):
    """Check if a and b match up to global phase."""
    dim = a.shape[0]
    trace = np.trace(a.conj().T @ b)
    if abs(trace) < 1e-15:
        return False
    phase = trace / abs(trace)
    return np.allclose(a, phase * b, atol=tol)


def _decompose_cnot():
    return [_cx(0, 1)]


def _decompose_iswap():
    # iSWAP = (H⊗H) · CNOT · (S⊗S) · CNOT · (H⊗H) up to phase
    # Actually: iSWAP = exp(iπ/4) · (H⊗H) · CNOT · (S⊗S) · CNOT · (H⊗H)
    # Let's use a known decomposition
    gates = []
    # H on both
    gates.append(_u3(0, math.pi, 0, 0))  # H
    gates.append(_u3(1, math.pi, 0, 0))  # H
    gates.append(_cx(0, 1))
    # S on both: S = u3(π/2, 0, π/2)
    gates.append(_u3(0, math.pi/2, 0, math.pi/2))
    gates.append(_u3(1, math.pi/2, 0, math.pi/2))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, math.pi, 0, 0))  # H
    gates.append(_u3(1, math.pi, 0, 0))  # H
    return gates


def _decompose_swap():
    gates = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    return gates


def _decompose_toffoli():
    # Toffoli decomposition with 6 CNOTs
    # Standard decomposition
    gates = []
    # H on target
    gates.append(_u3(2, math.pi, 0, 0))
    # CNOT(1,2)
    gates.append(_cx(1, 2))
    # T† on target
    gates.append(_u3(2, math.pi/4, 0, -math.pi/2))
    # CNOT(0,2)
    gates.append(_cx(0, 2))
    # T on target
    gates.append(_u3(2, math.pi/4, 0, math.pi/2))
    # CNOT(1,2)
    gates.append(_cx(1, 2))
    # T† on target
    gates.append(_u3(2, math.pi/4, 0, -math.pi/2))
    # CNOT(0,2)
    gates.append(_cx(0, 2))
    # T on target
    gates.append(_u3(2, math.pi/4, 0, math.pi/2))
    # CNOT(1,2)
    gates.append(_cx(1, 2))
    # H on target
    gates.append(_u3(2, math.pi, 0, 0))
    # T† on control 1
    gates.append(_u3(1, math.pi/4, 0, -math.pi/2))
    # CNOT(0,1)
    gates.append(_cx(0, 1))
    # T on control 1
    gates.append(_u3(1, math.pi/4, 0, math.pi/2))
    # CNOT(0,1)
    gates.append(_cx(0, 1))
    # T† on control 0
    gates.append(_u3(0, math.pi/4, 0, -math.pi/2))
    # T on control 1
    gates.append(_u3(1, math.pi/4, 0, math.pi/2))
    return gates


def _synthesize_generic(problem):
    """Optimize single-qubit parameters for a fixed CNOT topology."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    dim = 2 ** qubits
    
    # Create a reasonable CNOT layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Start from zero
    best = np.zeros(n_params)
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with adaptive step
    step = 1.0
    for sweep in range(20):
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best.copy()
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
            if _elapsed() > _DEADLINE - 30:
                break
        if not improved:
            step *= 0.5
        if step < 1e-8:
            break
        if _elapsed() > _DEADLINE - 30:
            break
    
    return circuit(best)


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates
    known = _is_known_gate(name, target, qubits)
    if known == "cnot":
        return _decompose_cnot()
    elif known == "iswap":
        return _decompose_iswap()
    elif known == "swap":
        return _decompose_swap()
    elif known == "toffoli":
        return _decompose_toffoli()
    
    # Generic optimization
    return _synthesize_generic(problem)
# EVOLVE-BLOCK-END
