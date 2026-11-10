"""Quantum circuit synthesis with CNOT budget constraints."""
import math
import time
import random
import itertools
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
_DEADLINE = None
_TIME_MARGIN = 10.0  # seconds of safety margin


def _elapsed():
    """Return seconds elapsed since the module was loaded."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return time.time() - _DEADLINE


def _time_remaining():
    """Return seconds remaining before the deadline."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    """Rebuild the target unitary from real and imaginary parts."""
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_u3(theta, phi, lam):
    """Build a single-qubit u3 rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _precompute_cnot(qubits):
    """Precompute CNOT matrices for all pairs."""
    cnots = {}
    dim = 2 ** qubits
    for c in range(qubits):
        for t in range(qubits):
            if c == t:
                continue
            cx_mat = np.eye(dim, dtype=complex)
            c_bit = qubits - 1 - c
            t_bit = qubits - 1 - t
            for i in range(dim):
                c_val = (i >> c_bit) & 1
                if c_val:
                    new_i = i ^ (1 << t_bit)
                    cx_mat[i, new_i] = 1.0
                else:
                    cx_mat[i, i] = 1.0
            cnots[(c, t)] = cx_mat
    return cnots


def _build_matrix(gates, qubits, cnot_cache):
    """Build the full unitary matrix from a gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    dim = 2 ** qubits
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _build_u3(theta, phi, lam)
            # Build the operator for this single-qubit gate
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        elif gate[0] == "cx":
            _, control, target = gate
            matrix = cnot_cache[(control, target)] @ matrix
            
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target† @ matrix)| / dim."""
    dim = target.shape[0]
    tr_val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr_val) / dim)


def _u3_gate(q, theta, phi, lam):
    """Create a u3 gate list entry."""
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx_gate(c, t):
    """Create a CNOT gate list entry."""
    return ["cx", int(c), int(t)]


def _optimize_angles(gates_template, target, qubits, cnot_cache, max_time=30.0):
    """
    Optimize single-qubit rotation angles using Nelder-Mead or Powell.
    """
    start_time = time.time()
    
    # Identify which gates are u3 and their parameter indices
    u3_indices = []
    param_map = {}  # param_idx -> (gate_idx, which_param)
    pi = 0
    for gi, gate in enumerate(gates_template):
        if gate[0] == "u3":
            u3_indices.append(gi)
            param_map[pi] = (gi, 0)
            param_map[pi + 1] = (gi, 1)
            param_map[pi + 2] = (gi, 2)
            pi += 3
    
    n_params = pi
    if n_params == 0:
        return gates_template[:]
    
    # Initialize with zero angles
    best = [0.0] * n_params
    best_gates = []
    for gate in gates_template:
        if gate[0] == "u3":
            best_gates.append(["u3", gate[1], 0.0, 0.0, 0.0])
        else:
            best_gates.append(list(gate))
    
    best_dist = _distance(target, _build_matrix(best_gates, qubits, cnot_cache))
    
    if best_dist < 1e-10:
        return best_gates
    
    def objective(params):
        if time.time() - start_time > max_time:
            return 1.0 # Penalty for timeout
        trial_gates = []
        tp = 0
        for gate in gates_template:
            if gate[0] == "u3":
                trial_gates.append(["u3", gate[1], float(params[tp]), float(params[tp+1]), float(params[tp+2])])
                tp += 3
            else:
                trial_gates.append(list(gate))
        dist = _distance(target, _build_matrix(trial_gates, qubits, cnot_cache))
        return dist
    
    # Choose optimizer based on dimension
    if n_params <= 30:
        method = 'Nelder-Mead'
    else:
        method = 'Powell'
        
    try:
        result = scipy.optimize.minimize(
            objective, 
            best, 
            method=method,
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
        )
        if result.fun < best_dist:
            best = result.x
            best_dist = result.fun
    except Exception:
        pass
    
    # Build final gates
    final_gates = []
    tp = 0
    for gate in gates_template:
        if gate[0] == "u3":
            final_gates.append(["u3", gate[1], float(best[tp]), float(best[tp + 1]), float(best[tp + 2])])
            tp += 3
        else:
            final_gates.append(list(gate))
    
    return final_gates


def _known_decompositions(name, qubits, cnot_budget, target):
    """
    Return known optimal decompositions for specific gates.
    """
    if qubits == 2:
        if name == "cnot" or name == "CNOT":
            gates = [_cx_gate(0, 1)]
            return gates
        elif name == "iswap" or name == "iSWAP":
            # iSWAP decomposition with 3 CNOTs
            gates_template = [
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(0, 1),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(1, 0),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(0, 1),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
            ]
            return gates_template
        elif name == "swap" or name == "SWAP":
            gates = [_cx_gate(0, 1), _cx_gate(1, 0), _cx_gate(0, 1)]
            return gates
        elif name == "cz" or name == "CZ":
            gates = [
                _u3_gate(0, math.pi/2, 0, math.pi),  # H on q0
                _cx_gate(0, 1),
                _u3_gate(0, math.pi/2, 0, math.pi),  # H on q0
            ]
            return gates
    
    elif qubits == 3:
        if name == "toffoli" or name == "Toffoli" or name == "CCNOT":
            # Toffoli decomposition
            gates_template = [
                _u3_gate(2, 0, 0, 0),  # H on c
                _cx_gate(0, 2),
                _u3_gate(2, 0, 0, 0),  # T on c
                _cx_gate(1, 2),
                _u3_gate(2, 0, 0, 0),  # T† on c
                _u3_gate(1, 0, 0, 0),  # T† on b
                _cx_gate(0, 2),
                _u3_gate(1, 0, 0, 0),  # T on b
                _u3_gate(2, 0, 0, 0),  # T on c
                _cx_gate(1, 2),
                _u3_gate(2, 0, 0, 0),  # H on c
                _u3_gate(0, 0, 0, 0),  # T on a
                _u3_gate(0, 0, 0, 0),  # S† on a
                _u3_gate(1, 0, 0, 0),  # S on b
            ]
            return gates_template
        elif name == "ccz" or name == "CCZ":
            # CCZ decomposition (similar to Toffoli with H on target)
            gates_template = [
                _u3_gate(2, math.pi/2, 0, math.pi), # H on c
                _u3_gate(2, 0, 0, 0),  # placeholder for T
                _cx_gate(0, 2),
                _u3_gate(2, 0, 0, 0),  # T
                _cx_gate(1, 2),
                _u3_gate(2, 0, 0, 0),  # T†
                _u3_gate(1, 0, 0, 0),  # T†
                _cx_gate(0, 2),
                _u3_gate(1, 0, 0, 0),  # T
                _u3_gate(2, 0, 0, 0),  # T
                _cx_gate(1, 2),
                _u3_gate(2, 0, 0, 0),  # H
                _u3_gate(2, math.pi/2, 0, math.pi), # H on c
                _u3_gate(0, 0, 0, 0),  # T
                _u3_gate(0, 0, 0, 0),  # S†
                _u3_gate(1, 0, 0, 0),  # S
            ]
            return gates_template

    return None


def _synthesize_2qubit_generic(target, cnot_budget, qubits=2, cnot_cache=None):
    """Synthesize a generic 2-qubit gate using Nelder-Mead."""
    if cnot_budget >= 3:
        cnots = 3
    elif cnot_budget >= 2:
        cnots = 2
    elif cnot_budget >= 1:
        cnots = 1
    else:
        cnots = 0
    
    gates_template = []
    for i in range(cnots):
        for q in range(2):
            gates_template.append(_u3_gate(q, 0, 0, 0))
        if i % 2 == 0:
            gates_template.append(_cx_gate(0, 1))
        else:
            gates_template.append(_cx_gate(1, 0))
    for q in range(2):
        gates_template.append(_u3_gate(q, 0, 0, 0))
        
    result = _optimize_angles(gates_template, target, qubits, cnot_cache, max_time=15.0)
    return result


def _synthesize_3qubit_generic(target, cnot_budget, qubits=3, cnot_cache=None):
    """Synthesize a generic 3-qubit gate using multiple templates."""
    best_gates = []
    best_dist = 1.0
    
    # Template 1: Toffoli-like (6 CNOTs)
    if cnot_budget >= 6:
        gates_template = [
            _u3_gate(2, 0, 0, 0),
            _cx_gate(0, 2),
            _u3_gate(2, 0, 0, 0),
            _cx_gate(1, 2),
            _u3_gate(2, 0, 0, 0),
            _u3_gate(1, 0, 0, 0),
            _cx_gate(0, 2),
            _u3_gate(1, 0, 0, 0),
            _u3_gate(2, 0, 0, 0),
            _cx_gate(1, 2),
            _u3_gate(2, 0, 0, 0),
            _u3_gate(0, 0, 0, 0),
            _u3_gate(0, 0, 0, 0),
            _u3_gate(1, 0, 0, 0),
        ]
        result = _optimize_angles(gates_template, target, qubits, cnot_cache, max_time=20.0)
        dist = _distance(target, _build_matrix(result, qubits, cnot_cache))
        if dist < best_dist:
            best_dist = dist
            best_gates = result
            
    # Template 2: 14-CNOT Ladder
    if cnot_budget >= 14 and _time_remaining() > 10.0:
        pairs = [(0,1), (1,0), (0,2), (2,0), (1,2), (2,1)]
        gates_template = []
        for i in range(14):
            for q in range(3):
                gates_template.append(_u3_gate(q, 0, 0, 0))
            c, t = pairs[i % len(pairs)]
            gates_template.append(_cx_gate(c, t))
        for q in range(3):
            gates_template.append(_u3_gate(q, 0, 0, 0))
            
        result = _optimize_angles(gates_template, target, qubits, cnot_cache, max_time=30.0)
        dist = _distance(target, _build_matrix(result, qubits, cnot_cache))
        if dist < best_dist:
            best_dist = dist
            best_gates = result

    if not best_gates:
        # Fallback: Identity
        best_gates = []
        
    return best_gates


def synthesize(problem):
    """
    Synthesize a quantum circuit for the given target unitary.
    Returns a list of gates.
    """
    name = problem.get("name", "")
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    global _DEADLINE
    _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    
    cnot_cache = _precompute_cnot(qubits)
    
    # Try known decompositions first
    known = _known_decompositions(name, qubits, cnot_budget, target)
    if known is not None:
        # Verify and optimize if needed
        if len(known) <= 400:
            cnot_count = sum(1 for g in known if g[0] == "cx")
            if cnot_count <= cnot_budget:
                has_u3 = any(g[0] == "u3" for g in known)
                if has_u3 and _time_remaining() > 5.0:
                    result = _optimize_angles(known, target, qubits, cnot_cache, max_time=10.0)
                else:
                    result = known
                return result
    
    # Generic synthesis
    if qubits == 2:
        result = _synthesize_2qubit_generic(target, cnot_budget, qubits, cnot_cache)
    elif qubits == 3:
        result = _synthesize_3qubit_generic(target, cnot_budget, qubits, cnot_cache)
    else:
        result = []
    
    # Verify CNOT count
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > cnot_budget:
        cnots_seen = 0
        trimmed = []
        for g in result:
            if g[0] == "cx":
                if cnots_seen < cnot_budget:
                    trimmed.append(g)
                    cnots_seen += 1
            else:
                trimmed.append(g)
        result = trimmed
    
    # Verify gate count
    if len(result) > 400:
        result = result[:400]
    
    return result
# EVOLVE-BLOCK-END
