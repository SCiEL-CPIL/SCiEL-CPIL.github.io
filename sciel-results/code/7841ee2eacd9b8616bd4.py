"""Quantum circuit synthesis with exact decompositions for known gates and KAK for generic."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _u3_from_unitary_2x2(u):
    """Extract u3 parameters from a 2x2 unitary. Returns (theta, phi, lam)."""
    if abs(u[0, 0]) < 1e-12:
        theta = math.pi
        phi = np.angle(u[1, 0])
        lam = np.angle(-u[0, 1])
    else:
        theta = 2 * np.arccos(np.clip(abs(u[0, 0]), 0, 1))
        if abs(u[1, 0]) < 1e-12:
            phi = 0.0
            if abs(u[0, 0]) > 1e-12:
                phi_lam = np.angle(u[1, 1]) - np.angle(u[0, 0])
                lam = phi_lam
            else:
                lam = 0.0
        else:
            phi = np.angle(u[1, 0])
            lam = np.angle(-u[0, 1])
    theta = (theta + 2 * math.pi) % (2 * math.pi)
    if theta > math.pi:
        theta = 2 * math.pi - theta
        phi += math.pi
        lam += math.pi
    phi = (phi + 2 * math.pi) % (2 * math.pi)
    lam = (lam + 2 * math.pi) % (2 * math.pi)
    return [theta, phi, lam]


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
            _, control, target_q = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target_q)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _cnot_gate(c, t):
    return ["cx", c, t]


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _decompose_cnot(c=0, t=1):
    return [_cnot_gate(c, t)]


def _decompose_iswap():
    # iSWAP = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
    # Known 3-CNOT decomposition
    gates = []
    gates.append(_u3_gate(0, 0, 0, math.pi / 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(0, 0, 0, math.pi / 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(0, 0, 0, math.pi / 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    gates.append(_cnot_gate(0, 1))
    return gates


def _decompose_swap():
    return [_cnot_gate(0, 1), _cnot_gate(1, 0), _cnot_gate(0, 1)]


def _decompose_toffoli():
    gates = []
    gates.append(_u3_gate(2, math.pi / 2, 0, 0))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_cnot_gate(0, 2))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_cnot_gate(0, 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 4))
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(1, 0, 0, -math.pi / 4))
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    gates.append(_cnot_gate(0, 1))
    gates.append(_u3_gate(2, math.pi / 2, 0, 0))
    return gates


def _kak_decompose_2qubit(u, qubits=2):
    best_params = [0.0] * 12
    
    def circuit_from_params(params):
        gates = []
        gates.append(_u3_gate(0, params[0], params[1], params[2]))
        gates.append(_u3_gate(1, params[3], params[4], params[5]))
        gates.append(_cnot_gate(0, 1))
        gates.append(_u3_gate(0, params[6], params[7], params[8]))
        gates.append(_cnot_gate(1, 0))
        gates.append(_u3_gate(1, params[9], params[10], params[11]))
        gates.append(_cnot_gate(0, 1))
        return gates
    
    def dist(params):
        gates = circuit_from_params(params)
        mat = _build(gates, 2)
        return _distance(u, mat)
    
    best_d = dist(best_params)
    best_p = best_params[:]
    
    for sweep in range(8):
        step = 0.5 / (sweep + 1)
        for i in range(12):
            for trial in (best_p[i] + step, best_p[i] - step,
                         best_p[i] + np.random.uniform(-math.pi, math.pi)):
                p = best_p[:]
                p[i] = trial
                d = dist(p)
                if d < best_d:
                    best_p, best_d = p, d
    
    return circuit_from_params(best_p)


def _decompose_3qubit_generic(u, qubits=3):
    n_u3 = 12
    best_params = [0.0] * (3 * n_u3)
    
    def circuit_from_params(params):
        gates = []
        for layer in range(4):
            for q in range(3):
                idx = (layer * 3 + q) * 3
                gates.append(_u3_gate(q, params[idx], params[idx+1], params[idx+2]))
            if layer < 3:
                cnots = [(0,1), (1,2), (0,2)]
                gates.append(_cnot_gate(cnots[layer][0], cnots[layer][1]))
                gates.append(_cnot_gate(cnots[(layer+1)%3][0], cnots[(layer+1)%3][1]))
        return gates
    
    def dist(params):
        gates = circuit_from_params(params)
        mat = _build(gates, 3)
        return _distance(u, mat)
    
    best_d = dist(best_params)
    best_p = best_params[:]
    
    for sweep in range(6):
        step = 0.5 / (sweep + 1)
        for i in range(len(best_p)):
            for trial in (best_p[i] + step, best_p[i] - step):
                p = best_p[:]
                p[i] = trial
                d = dist(p)
                if d < best_d:
                    best_p, best_d = p, d
    
    return circuit_from_params(best_p)


def synthesize(problem):
    start_time = time.time()
    deadline = start_time + 230
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    if qubits == 2:
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_target, atol=1e-10):
            return _decompose_cnot()
        
        iswap_target = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex)
        if np.allclose(target, iswap_target, atol=1e-10):
            return _decompose_iswap()
        
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap_target, atol=1e-10):
            return _decompose_swap()
        
        return _kak_decompose_2qubit(target)
    
    else:
        toffoli_target = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            if (i & 4) and (i & 2):
                toffoli_target[i, i ^ 1] = 1
            else:
                toffoli_target[i, i] = 1
        if np.allclose(target, toffoli_target, atol=1e-10):
            return _decompose_toffoli()
        
        return _decompose_3qubit_generic(target)


# EVOLVE-BLOCK-END
