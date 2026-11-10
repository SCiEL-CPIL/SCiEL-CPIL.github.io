"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = None
_TIME_MARGIN = 35.0  # seconds safety margin before 270s kill


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
        gates = [["cx", 0, 1]]
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None

    # iSWAP gate
    if name == "iSWAP" and qubits == 2:
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],
            ["u3", 1, math.pi, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, 0.0, 0.0, -math.pi / 2],
            ["u3", 1, 0.0, 0.0, math.pi / 2],
            ["cx", 1, 0],
            ["u3", 0, math.pi, 0.0, 0.0],
            ["u3", 1, math.pi, 0.0, 0.0],
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
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, math.pi / 4],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 0, 1],
            ["u3", 0, math.pi, 0.0, 0.0],
        ]
        built = _build(gates, 3)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None

    return None


def _build_from_params(params, qubits, cnot_ctrls, cnot_tgts, u3_qubits):
    """Build circuit from flat parameter vector."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    n_u3 = len(u3_qubits)
    n_cnot = len(cnot_ctrls)

    cnot_perms = []
    for ctrl, tgt in zip(cnot_ctrls, cnot_tgts):
        idx = np.arange(n)
        bit = (idx >> (qubits - 1 - ctrl)) & 1
        perm = idx ^ (bit << (qubits - 1 - tgt))
        cnot_perms.append(perm)

    gate_order = []
    u3_idx = 0
    cnot_idx = 0
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
            theta, phi, lam = params[3 * gidx], params[3 * gidx + 1], params[3 * gidx + 2]
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            mat = op @ mat
        else:
            mat = mat[cnot_perms[gidx], :]

    return mat


def _objective(params, target, qubits, cnot_ctrls, cnot_tgts, u3_qubits):
    """Objective function for optimization: phase-insensitive distance."""
    mat = _build_from_params(params, qubits, cnot_ctrls, cnot_tgts, u3_qubits)
    return _distance(target, mat)


def _optimize_generic(problem, max_time=20.0):
    """Optimize a generic unitary using scipy L-BFGS-B."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits

    # Build CNOT layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_ctrls = [pairs[i % len(pairs)][0] for i in range(budget)]
    cnot_tgts = [pairs[i % len(pairs)][1] for i in range(budget)]

    # U3 slots
    u3_qubits = []
    for layer in range(budget):
        for q in range(qubits):
            u3_qubits.append(q)
    for q in range(qubits):
        u3_qubits.append(q)

    n_u3 = len(u3_qubits)
    n_params = 3 * n_u3

    # Multiple random initializations
    best_params = None
    best_dist = float('inf')
    start = time.time()

    n_restarts = min(5, max(1, int((max_time - (time.time() - start)) / 4.0)))

    for restart in range(n_restarts):
        if time.time() - start > max_time or _time_remaining() < 10.0:
            break

        if restart == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-math.pi, math.pi, size=n_params)

        def obj(p):
            return _objective(p, target, qubits, cnot_ctrls, cnot_tgts, u3_qubits)

        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros(n_params)

    # Convert to gate list
    gates = []
    k = 0
    for layer in range(budget):
        for q in range(qubits):
            gates.append(["u3", q, float(best_params[k]), float(best_params[k + 1]), float(best_params[k + 2])])
            k += 3
        gates.append(["cx", cnot_ctrls[layer], cnot_tgts[layer]])
    for q in range(qubits):
        gates.append(["u3", q, float(best_params[k]), float(best_params[k + 1]), float(best_params[k + 2])])
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
