"""Quantum circuit synthesis with exact decompositions for known gates and fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = 255.0  # Safety margin before 270s kill
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            e_lam = np.exp(1j * lam)
            e_phi = np.exp(1j * phi)
            single = np.array([[c, -e_lam * s], [e_phi * s, e_phi * e_lam * c]])
            
            # Build operator using kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target bit when control is 1
            # In basis ordering where qubit 0 is MSB
            n = dim
            new_matrix = matrix.copy()
            for i in range(n):
                ctrl_bit = (i >> (qubits - 1 - control)) & 1
                if ctrl_bit:
                    tgt_bit = (i >> (qubits - 1 - target)) & 1
                    if tgt_bit == 0:
                        j = i | (1 << (qubits - 1 - target))
                        new_matrix[i] = matrix[j]
                        new_matrix[j] = matrix[i]
            matrix = new_matrix
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / dim"""
    dim = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / dim
    return 1.0 - min(1.0, val)


def _exact_cnot():
    """CNOT gate: 1 CNOT"""
    return [["cx", 0, 1]]


def _exact_swap():
    """SWAP gate: 3 CNOTs"""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]


def _exact_iswap():
    """iSWAP gate: 3 CNOTs with single-qubit rotations"""
    gates = []
    h = (math.pi / 2, 0, math.pi)
    s = (0, 0, math.pi / 2)
    sdag = (0, 0, -math.pi / 2)
    
    gates.append(["u3", 0, h[0], h[1], h[2]])
    gates.append(["u3", 1, h[0], h[1], h[2]])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, sdag[0], sdag[1], sdag[2]])
    gates.append(["u3", 1, sdag[0], sdag[1], sdag[2]])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, s[0], s[1], s[2]])
    gates.append(["u3", 1, s[0], s[1], s[2]])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, h[0], h[1], h[2]])
    gates.append(["u3", 1, h[0], h[1], h[2]])
    
    return gates


def _exact_toffoli():
    """Toffoli gate: 6 CNOTs standard decomposition"""
    t = (0, 0, math.pi / 4)
    tdag = (0, 0, -math.pi / 4)
    h = (math.pi / 2, 0, math.pi)
    
    gates = [
        ["u3", 2, h[0], h[1], h[2]],
        ["cx", 1, 2],
        ["u3", 2, tdag[0], tdag[1], tdag[2]],
        ["cx", 0, 2],
        ["u3", 2, t[0], t[1], t[2]],
        ["cx", 1, 2],
        ["u3", 2, tdag[0], tdag[1], tdag[2]],
        ["u3", 1, tdag[0], tdag[1], tdag[2]],
        ["cx", 0, 1],
        ["u3", 1, t[0], t[1], t[2]],
        ["u3", 2, t[0], t[1], t[2]],
        ["cx", 0, 2],
        ["u3", 1, tdag[0], tdag[1], tdag[2]],
        ["u3", 2, t[0], t[1], t[2]],
        ["u3", 2, h[0], h[1], h[2]],
    ]
    
    return gates


def _optimize_generic(problem, qubits, budget):
    """Optimize single-qubit parameters for a generic unitary with given CNOT layout."""
    global _start_time, _DEADLINE
    
    target = _target(problem)
    dim = 2 ** qubits
    
    # Choose a CNOT layout
    if qubits == 2:
        # Use 4 CNOTs for 2-qubit generic
        pairs = [(0, 1), (1, 0), (0, 1), (1, 0)]
        cnot_count = min(budget, 4)
    elif qubits == 3:
        # Use up to budget CNOTs for 3-qubit
        all_pairs = [(0,1), (1,2), (0,2), (2,1), (1,0), (2,0)]
        cnot_count = min(budget, 15)
    else:
        cnot_count = min(budget, 4)
        pairs = [(0, 1)] * cnot_count
    
    # Build layout: u3 layers around CNOTs
    layout = []
    num_u3_layers = cnot_count + 1
    u3_params = []
    
    param_index = 0
    for layer in range(num_u3_layers):
        for q in range(qubits):
            layout.append(["u3", q, param_index])
            param_index += 3
        if layer < cnot_count:
            if qubits == 2:
                layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
            else:
                layout.append(["cx", all_pairs[layer % len(all_pairs)][0], all_pairs[layer % len(all_pairs)][1]])
    
    total_params = param_index
    max_gates = 400
    if len(layout) > max_gates:
        layout = layout[:max_gates]
        total_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def build_circuit(params):
        gates = []
        pi = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(g))
        return gates
    
    # Initialize parameters
    rng = random.Random(12345 + qubits)
    best_params = [0.0] * total_params
    best_dist = _distance(target, _build_fast(build_circuit(best_params), qubits))
    
    # Coordinate descent with deadline guard
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
    
    for sweep in range(len(step_sizes)):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = step_sizes[sweep]
        for i in range(total_params):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial_offset in (step, -step, rng.uniform(-step*2, step*2)):
                trial_params = best_params[:]
                trial_params[i] = trial_offset + best_params[i]
                trial_dist = _distance(target, _build_fast(build_circuit(trial_params), qubits))
                if trial_dist < best_dist:
                    best_params = trial_params
                    best_dist = trial_dist
    
    # Final polish
    for sweep in range(3):
        if time.time() - _start_time > _DEADLINE - 5:
            break
        step = 0.001 / (1 + sweep)
        for i in range(total_params):
            if time.time() - _start_time > _DEADLINE - 5:
                break
            for trial_offset in (step, -step):
                trial_params = best_params[:]
                trial_params[i] = best_params[i] + trial_offset
                trial_dist = _distance(target, _build_fast(build_circuit(trial_params), qubits))
                if trial_dist < best_dist:
                    best_params = trial_params
                    best_dist = trial_dist
    
    return build_circuit(best_params)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    global _start_time, _DEADLINE
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE - 5:
        return []
    
    # Exact decompositions for known gates
    if name == "CNOT" or name == "cnot":
        gates = _exact_cnot()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots > budget:
            return _optimize_generic(problem, qubits, budget)
        return gates
    
    elif name == "SWAP" or name == "swap":
        gates = _exact_swap()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots > budget:
            return _optimize_generic(problem, qubits, budget)
        return gates
    
    elif name == "iSWAP" or name == "iswap" or name == "ISWAP":
        gates = _exact_iswap()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots > budget:
            return _optimize_generic(problem, qubits, budget)
        return gates
    
    elif name == "Toffoli" or name == "toffoli" or name == "TOFFOLI":
        gates = _exact_toffoli()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots > budget:
            return _optimize_generic(problem, qubits, budget)
        return gates
    
    else:
        # Generic unitary: use optimization
        return _optimize_generic(problem, qubits, budget)
# EVOLVE-BLOCK-END
