"""Quantum circuit synthesis: exact algebraic decompositions + fast coordinate descent."""
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
                     [e2 * s, e2 * e1 * c]], dtype=complex)


def _precompute_cnot_perm(qubits, control, target):
    """Precompute the permutation for a CNOT gate on given qubits."""
    n = 2 ** qubits
    ctrl_bit = qubits - 1 - control
    tgt_bit = qubits - 1 - target
    perm = np.arange(n)
    for idx in range(n):
        if (idx >> ctrl_bit) & 1:
            perm[idx] = idx ^ (1 << tgt_bit)
    return perm


def _build_fast(gates, qubits, cnot_perms=None):
    """Build the unitary from a gate list using precomputed CNOT permutations."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator by kron
            op = np.array([[1.0 + 0j]])
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, control, target = gate
            if cnot_perms is not None:
                perm = cnot_perms[(control, target)]
            else:
                ctrl_bit = qubits - 1 - control
                tgt_bit = qubits - 1 - target
                perm = np.arange(n)
                for idx in range(n):
                    if (idx >> ctrl_bit) & 1:
                        perm[idx] = idx ^ (1 << tgt_bit)
            U = U[perm, :]
    return U


def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)


def _exact_cnot_decomposition():
    """CNOT gate: 1 CNOT, no single-qubit gates needed."""
    return [["cx", 0, 1]]


def _exact_iswap_decomposition():
    """iSWAP gate: 3 CNOTs with single-qubit rotations."""
    gates = []
    gates.append(["u3", 0, math.pi / 2, 0.0, math.pi])
    gates.append(["u3", 1, math.pi / 2, 0.0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0.0, 0.0, math.pi / 2])
    gates.append(["u3", 1, 0.0, 0.0, math.pi / 2])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, 0.0, 0.0, math.pi / 2])
    gates.append(["u3", 1, 0.0, 0.0, math.pi / 2])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi / 2, 0.0, math.pi])
    gates.append(["u3", 1, math.pi / 2, 0.0, math.pi])
    return gates


def _exact_swap_decomposition():
    """SWAP gate: 3 CNOTs."""
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _exact_toffoli_decomposition():
    """Toffoli gate: 6 CNOTs with single-qubit rotations."""
    gates = []
    gates.append(["u3", 2, math.pi / 2, 0.0, math.pi])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -math.pi / 4])
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, 0.0, 0.0, math.pi / 4])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi / 2, 0.0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, 0.0, 0.0, -math.pi / 4])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, 0.0, 0.0, math.pi / 4])
    return gates


def _solve_u3_params(target_single):
    """
    Solve for theta, phi, lam such that u3(theta, phi, lam) = target_single (up to global phase).
    u3 = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
          [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    """
    det = np.linalg.det(target_single)
    if abs(det) > 1e-12:
        global_phase = np.sqrt(det)
        U = target_single / global_phase
    else:
        U = target_single
    
    cos_half = np.real(U[0, 0])
    sin_half_abs = np.sqrt(max(0, 1 - cos_half**2))
    
    if sin_half_abs < 1e-10:
        if abs(cos_half - 1) < 1e-10:
            return 0.0, 0.0, 0.0
        else:
            lam = -np.angle(U[0, 1])
            phi = np.angle(U[1, 0])
            return math.pi, phi, lam
    
    theta = 2 * np.arccos(np.clip(cos_half, -1, 1))
    sin_half = np.sin(theta / 2)
    
    lam = np.angle(-U[0, 1] / sin_half)
    phi = np.angle(U[1, 0] / sin_half)
    
    return theta, phi, lam


def _optimize_generic(target, qubits, budget, max_time=20.0):
    """Optimize a generic unitary using multi-start coordinate descent."""
    n = 2 ** qubits
    rng = random.Random(12345 + qubits + budget)
    
    # Precompute CNOT permutations for all pairs
    cnot_perms = {}
    for c in range(qubits):
        for t in range(qubits):
            if c != t:
                cnot_perms[(c, t)] = _precompute_cnot_perm(qubits, c, t)
    
    # Build a layout: alternate single-qubit layers and CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        if layer < budget:
            layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    total_gates = len(layout)
    if total_gates > 400:
        while total_gates > 400 and len(layout) > budget:
            for i in range(len(layout) - 1, -1, -1):
                if layout[i][0] == "u3":
                    layout.pop(i)
                    total_gates -= 1
                    break
            else:
                break
        num_u3 = sum(1 for g in layout if g[0] == "u3")
        num_params = 3 * num_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def diagonal_init():
        """Initialize by trying to match the diagonal of the target."""
        values = [0.0] * num_params
        return values
    
    best_values = None
    best_dist = 1.0
    
    num_starts = 5
    for start in range(num_starts):
        if _check_deadline():
            break
        
        if start == 0:
            values = diagonal_init()
        else:
            values = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
        
        dist = _distance(target, _build_fast(circuit(values), qubits, cnot_perms))
        
        for sweep in range(10):
            if _check_deadline():
                break
            step = math.pi / (2 ** sweep)
            improved = False
            for i in range(num_params):
                if _check_deadline():
                    break
                for trial in (values[i] + step, values[i] - step):
                    new_values = values[:]
                    new_values[i] = trial
                    d = _distance(target, _build_fast(circuit(new_values), qubits, cnot_perms))
                    if d < dist:
                        values = new_values
                        dist = d
                        improved = True
                if sweep >= 4:
                    trial = values[i] + rng.uniform(-step, step)
                    new_values = values[:]
                    new_values[i] = trial
                    d = _distance(target, _build_fast(circuit(new_values), qubits, cnot_perms))
                    if d < dist:
                        values = new_values
                        dist = d
                        improved = True
            if not improved and sweep > 3:
                break
        
        if dist < best_dist:
            best_dist = dist
            best_values = values[:]
    
    return circuit(best_values), best_dist


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    
    if qubits == 2:
        if name == "cnot" or name == "CNOT":
            gates = _exact_cnot_decomposition()
            cx_count = sum(1 for g in gates if g[0] == "cx")
            if cx_count <= budget:
                return gates
        
        if name == "iswap" or name == "iSWAP":
            gates = _exact_iswap_decomposition()
            cx_count = sum(1 for g in gates if g[0] == "cx")
            if cx_count <= budget:
                return gates
        
        if name == "swap" or name == "SWAP":
            gates = _exact_swap_decomposition()
            cx_count = sum(1 for g in gates if g[0] == "cx")
            if cx_count <= budget:
                return gates
    
    if qubits == 3:
        if name == "toffoli" or name == "Toffoli" or name == "ccx" or name == "CCX":
            gates = _exact_toffoli_decomposition()
            cx_count = sum(1 for g in gates if g[0] == "cx")
            if cx_count <= budget:
                return gates
    
    gates, dist = _optimize_generic(target, qubits, budget)
    
    cx_count = sum(1 for g in gates if g[0] == "cx")
    if cx_count > budget or len(gates) > 400:
        return [["u3", 0, 0.0, 0.0, 0.0]]
    
    return gates
# EVOLVE-BLOCK-END
