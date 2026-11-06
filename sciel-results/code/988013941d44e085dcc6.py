"""Combined approach: exact algebraic decompositions for known gates + fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before 270s wall-clock kill


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]])


def _build_fast(gates, qubits, target):
    """Build circuit unitary and compute phase-insensitive distance efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1 = np.exp(1j * lam)
            e2 = np.exp(1j * phi)
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.zeros_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _exact_cnot():
    """CNOT gate: |00><00| + |01><01| + |10><11| + |11><10|"""
    return [_cx(0, 1)]


def _exact_iswap():
    """iSWAP gate decomposition with 2 CNOTs."""
    # iSWAP = [[0,1,0,0],[1,0,0,0],[0,0,0,i],[0,0,i,0]]
    # Decomposition: (H⊗H) CNOT (S†⊗S†) CNOT (H⊗H) with phase corrections
    # H = u3(π/2, 0, π)
    # S† = u3(0, 0, -π/2)
    gates = [
        _u3(0, math.pi / 2, 0, math.pi),   # H on q0
        _u3(1, math.pi / 2, 0, math.pi),   # H on q1
        _cx(0, 1),
        _u3(0, 0, 0, -math.pi / 2),        # S† on q0
        _u3(1, 0, 0, -math.pi / 2),        # S† on q1
        _cx(0, 1),
        _u3(0, math.pi / 2, 0, math.pi),   # H on q0
        _u3(1, math.pi / 2, 0, math.pi),   # H on q1
    ]
    return gates


def _exact_swap():
    """SWAP gate: 3 CNOTs"""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)"""
    # H = u3(π/2, 0, π), T = u3(0, 0, π/4), T† = u3(0, 0, -π/4)
    gates = [
        _u3(2, math.pi / 2, 0, math.pi),   # H on q2
        _u3(2, 0, 0, math.pi / 4),         # T on q2
        _cx(1, 2),
        _u3(1, 0, 0, -math.pi / 4),        # T† on q1
        _u3(2, 0, 0, math.pi / 4),         # T on q2
        _cx(0, 2),
        _u3(0, 0, 0, -math.pi / 4),        # T† on q0
        _u3(1, 0, 0, math.pi / 4),         # T on q1
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi / 4),        # T† on q2
        _u3(0, 0, 0, math.pi / 4),         # T on q0
        _u3(1, 0, 0, -math.pi / 4),        # T† on q1
        _cx(0, 2),
        _u3(2, math.pi / 2, 0, math.pi),   # H on q2
    ]
    return gates


def _try_known_gates(problem, target, qubits, budget):
    """Try to match target to known standard gates and return exact circuit if found."""
    name = problem.get("name", "")
    
    if qubits == 2:
        # CNOT
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_target, atol=1e-8):
            return _exact_cnot()
        
        # iSWAP: [[0,1,0,0],[1,0,0,0],[0,0,0,i],[0,0,i,0]]
        iswap_target = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1j],[0,0,1j,0]], dtype=complex)
        if np.allclose(target, iswap_target, atol=1e-8):
            return _exact_iswap()
        
        # SWAP
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap_target, atol=1e-8):
            return _exact_swap()
    
    if qubits == 3:
        # Toffoli: |110> -> |111>, |111> -> |110>
        toffoli_target = np.eye(8, dtype=complex)
        toffoli_target[6,6] = 0
        toffoli_target[7,7] = 0
        toffoli_target[6,7] = 1
        toffoli_target[7,6] = 1
        if np.allclose(target, toffoli_target, atol=1e-8):
            return _exact_toffoli()
    
    return None


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations for generic unitaries using coordinate descent."""
    # Define CNOT layout - use all possible pairs for better coverage
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout: u3 gates before and after each CNOT
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * num_u3
    
    param_map = []
    k = 0
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0))
            param_map.append((i, 1))
            param_map.append((i, 2))
            k += 3
    
    def get_circuit(params):
        gates = [list(g) for g in layout]
        for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
            gates[gate_idx][2 + angle_idx] = params[p_idx]
        return gates
    
    def objective(params):
        if _check_deadline():
            return 1.0
        gates = get_circuit(params)
        return _build_fast(gates, qubits, target)
    
    best_params = [0.0] * n_params
    best_dist = objective(best_params)
    
    rng = random.Random(4242 + qubits + budget)
    
    # Adaptive coordinate descent with multiple restarts
    max_sweeps = 25
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        
        step = 1.0 / (1 + sweep)
        improved = False
        
        for i in range(n_params):
            if _check_deadline():
                break
            
            current_val = best_params[i]
            candidates = [
                current_val + step,
                current_val - step,
                rng.uniform(-math.pi, math.pi)
            ]
            
            for trial in candidates:
                trial = (trial + math.pi) % (2 * math.pi) - math.pi
                
                new_params = best_params[:]
                new_params[i] = trial
                
                val = objective(new_params)
                
                if val < best_dist:
                    best_params = new_params
                    best_dist = val
                    improved = True
                    break
        
        if not improved and sweep > 10 and step < 1e-5:
            break
    
    # If still not close, try a few random restarts
    if best_dist > 0.01 and not _check_deadline():
        for restart in range(3):
            if _check_deadline():
                break
            
            trial_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            trial_dist = objective(trial_params)
            
            if trial_dist < best_dist:
                best_params = trial_params
                best_dist = trial_dist
                
                # Quick local refinement
                for _ in range(5):
                    if _check_deadline():
                        break
                    step = 0.5 / (1 + _)
                    for i in range(n_params):
                        if _check_deadline():
                            break
                        for trial in (best_params[i] + step, best_params[i] - step):
                            trial = (trial + math.pi) % (2 * math.pi) - math.pi
                            new_params = best_params[:]
                            new_params[i] = trial
                            val = objective(new_params)
                            if val < best_dist:
                                best_params = new_params
                                best_dist = val
                                break
    
    return get_circuit(best_params)


def synthesize(problem):
    """
    Synthesize a quantum circuit: use exact decompositions for known gates,
    optimize for generic unitaries.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate decompositions first
    known_circuit = _try_known_gates(problem, target, qubits, budget)
    if known_circuit is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known_circuit if g[0] == "cx")
        if cnot_count <= budget and len(known_circuit) <= 400:
            return known_circuit
    
    # Fall back to optimization for generic unitaries
    return _optimize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
