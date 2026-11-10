"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0  # seconds safety margin before 270s kill


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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


def _known_decompositions(problem):
    """Return exact gate lists for known gates, or None."""
    name = problem["name"]
    qubits = problem["qubits"]
    target = _target(problem)
    
    # CNOT gate
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    
    # iSWAP gate
    if name == "iSWAP" and qubits == 2:
        # iSWAP = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]] * i on off-diagonals
        # Standard decomposition:
        # q0: H --×-- S† --×-- H
        # q1: H --×-- S  --×-- H
        # S = diag(1, i) = u3(0, 0, pi/2)
        # S† = diag(1, -i) = u3(0, 0, -pi/2)
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
            ["cx", 0, 1],
            ["u3", 0, 0.0, 0.0, -math.pi / 2],  # S† on q0
            ["u3", 1, 0.0, 0.0, math.pi / 2],   # S on q1
            ["cx", 1, 0],
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
        ]
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    # SWAP gate
    if name == "SWAP" and qubits == 2:
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    # Toffoli gate
    if name == "Toffoli" and qubits == 3:
        # Standard Toffoli decomposition with 6 CNOTs
        # q2 (MSB) -- control, q1 -- control, q0 -- target
        # Standard: 
        # q2: H --×------×-- H
        # q1: H --×--×--×-- H  
        # q0: H --×--×--×-- H  ... no that's not right
        # The standard 6-CNOT Toffoli:
        # Let me use the known decomposition:
        # q2: --×--×--×--
        # q1: --×--×--×--  
        # q0: --×--×--×--
        # With single-qubit gates in between
        # Actually the standard is:
        # q2: H --×------×-- H
        # q1: H --×--×--×-- H
        # q0: H --×--×--×-- H
        # No, let me just use a known 6-CNOT decomposition
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H on q0
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, math.pi / 4],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 0, 1],
            ["u3", 0, math.pi, 0.0, 0.0],  # H on q0
        ]
        built = _build(gates, 3)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    return None


def _build_fast(params, qubits, cnot_ctrls, cnot_tgts, u3_qubits):
    """Fast circuit builder for optimization. params is flat [theta,phi,lam]*n_u3."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    n_u3 = len(u3_qubits)
    n_cnot = len(cnot_ctrls)
    
    # Pre-compute CNOT permutation indices
    cnot_perms = []
    for ctrl, tgt in zip(cnot_ctrls, cnot_tgts):
        idx = np.arange(n)
        bit = (idx >> (qubits - 1 - ctrl)) & 1
        perm = idx ^ (bit << (qubits - 1 - tgt))
        cnot_perms.append(perm)
    
    # Pre-compute single-qubit operator bases for each qubit position
    # For each qubit q, we need the operator that acts on qubit q
    # We'll build it on the fly using kron
    
    gate_idx = 0
    cnot_idx = 0
    u3_idx = 0
    
    # Build the circuit sequence: we need to know the order of gates
    # Layout: for each layer, u3 on all qubits, then one CNOT
    # After all layers, u3 on all qubits
    
    total_gates = n_u3 + n_cnot
    # Build gate order: list of ('u3', qubit_idx) or ('cx', cnot_idx)
    gate_order = []
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    for layer in range(n_cnot):
        for q in range(qubits):
            gate_order.append(('u3', u3_idx))
            u3_idx += 1
        gate_order.append(('cx', cnot_idx))
        cnot_idx += 1
    for q in range(qubits):
        gate_order.append(('u3', u3_idx))
        u3_idx += 1
    
    for gtype, gidx in gate_order:
        if gtype == 'u3':
            q = u3_qubits[gidx]
            theta, phi, lam = params[3*gidx], params[3*gidx+1], params[3*gidx+2]
            c, s = math.cos(theta/2), math.sin(theta/2)
            single = np.array([[c, -np.exp(1j*lam)*s],
                               [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]])
            # Build operator
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            mat = op @ mat
        else:
            mat = mat[cnot_perms[gidx], :]
    
    return mat


def _optimize_generic(problem, max_time=25.0):
    """Optimize a generic unitary using gradient-free optimization."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Build CNOT layout - use a pattern that covers all pairs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_ctrls = [pairs[i % len(pairs)][0] for i in range(budget)]
    cnot_tgts = [pairs[i % len(pairs)][1] for i in range(budget)]
    
    # U3 slots: before each CNOT layer (all qubits) and after last layer
    u3_qubits = []
    for layer in range(budget):
        for q in range(qubits):
            u3_qubits.append(q)
    for q in range(qubits):
        u3_qubits.append(q)
    
    n_u3 = len(u3_qubits)
    n_params = 3 * n_u3
    
    # Initialize with zeros
    best = np.zeros(n_params)
    best_mat = _build_fast(best, qubits, cnot_ctrls, cnot_tgts, u3_qubits)
    best_dist = _distance(target, best_mat)
    
    start = time.time()
    sweep = 0
    step = 0.5
    
    while time.time() - start < max_time and _time_remaining() > 5.0:
        improved = False
        for i in range(n_params):
            if time.time() - start > max_time or _time_remaining() < 5.0:
                break
            for sign in (1, -1):
                trial = best.copy()
                trial[i] += sign * step
                trial_mat = _build_fast(trial, qubits, cnot_ctrls, cnot_tgts, u3_qubits)
                val = _distance(target, trial_mat)
                if val < best_dist:
                    best, best_dist, best_mat = trial, val, trial_mat
                    improved = True
        sweep += 1
        step *= 0.7
        if not improved and step < 0.01:
            break
    
    # Convert to gate list
    gates = []
    k = 0
    for layer in range(budget):
        for q in range(qubits):
            gates.append(["u3", q, best[k], best[k+1], best[k+2]])
            k += 3
        gates.append(["cx", cnot_ctrls[layer], cnot_tgts[layer]])
    for q in range(qubits):
        gates.append(["u3", q, best[k], best[k+1], best[k+2]])
        k += 3
    
    return gates


def synthesize(problem):
    """Synthesize a quantum circuit within the CNOT budget."""
    # First try known exact decompositions
    known = _known_decompositions(problem)
    if known is not None:
        return known
    
    # For generic unitaries, use optimization
    return _optimize_generic(problem)
# EVOLVE-BLOCK-END
