"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _u3_params_from_matrix(mat):
    """Extract (theta, phi, lam) from a 2x2 unitary matching the u3 convention."""
    c = mat[0, 0]
    s1 = mat[1, 0]
    s2 = mat[0, 1]
    c2 = mat[1, 1]
    # c = cos(theta/2)
    # s = sin(theta/2)
    # mat[0,0] = c
    # mat[0,1] = -e^{i*lam} * s
    # mat[1,0] = e^{i*phi} * s
    # mat[1,1] = e^{i*(phi+lam)} * c
    theta = 2.0 * np.angle(c)
    # c should be real and non-negative; if it's complex, adjust
    if abs(c) < 1e-12:
        theta = math.pi
    else:
        # c = cos(theta/2), so theta = 2*acos(Re(c)/|c|)
        # But c should be real. Let's use:
        cos_half = np.abs(c)
        sin_half = np.abs(s1)
        theta = 2.0 * np.arctan2(sin_half, cos_half)
    
    if sin_half < 1e-12:
        phi = 0.0
        lam = 0.0
    else:
        # mat[1,0] = e^{i*phi} * s => phi = angle(mat[1,0])
        phi = np.angle(s1)
        # mat[0,1] = -e^{i*lam} * s => e^{i*lam} = -mat[0,1]/s
        lam = np.angle(-s2 / sin_half)
    
    return float(theta), float(phi), float(lam)


def _known_decomposition(name, qubits, target):
    """
    Return exact gate list for known gates if the target matches.
    Returns None if not a known gate or doesn't match.
    """
    # CNOT: |00><00| + |01><01| + |10><11| + |11><10|
    cnot_target = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if name == "CNOT" or (qubits == 2 and np.allclose(target, cnot_target)):
        return [["cx", 0, 1]]

    # SWAP: |00><00| + |01><10| + |10><01| + |11><11|
    swap_target = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if name == "SWAP" or (qubits == 2 and np.allclose(target, swap_target)):
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1]
        ]

    # iSWAP: |00><00| + i|01><10| + i|10><01| + |11><11|
    iswap_target = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if name == "iSWAP" or (qubits == 2 and np.allclose(target, iswap_target)):
        # 2-CNOT decomposition of iSWAP
        # iSWAP = (H on q0)(H on q1) * CNOT(0->1) * (U3 on q0)(U3 on q1) * CNOT(1->0) * (U3 on q0)(U3 on q1) * (H on q0)(H on q1)
        # Let me use the standard decomposition:
        # q0: U3(pi/2, 0, pi)  -> H
        # q1: U3(pi/2, 0, pi)  -> H
        # CNOT 0->1
        # q0: U3(pi/4, 0, -pi/2)
        # q1: U3(pi/4, pi/2, 0)
        # CNOT 1->0
        # q0: U3(-pi/4, 0, -pi/2)
        # q1: U3(-pi/4, pi/2, 0)
        # q0: U3(pi/2, 0, pi)  -> H
        # q1: U3(pi/2, 0, pi)  -> H
        return [
            ["u3", 0, math.pi / 2, 0, math.pi],
            ["u3", 1, math.pi / 2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 0, math.pi / 4, 0, -math.pi / 2],
            ["u3", 1, math.pi / 4, math.pi / 2, 0],
            ["cx", 1, 0],
            ["u3", 0, -math.pi / 4, 0, -math.pi / 2],
            ["u3", 1, -math.pi / 4, math.pi / 2, 0],
            ["u3", 0, math.pi / 2, 0, math.pi],
            ["u3", 1, math.pi / 2, 0, math.pi]
        ]

    # Toffoli: 3 qubits, 6 CNOTs
    toffoli_target = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=complex)
    if name == "Toffoli" or (qubits == 3 and np.allclose(target, toffoli_target)):
        # Standard 6-CNOT Toffoli decomposition
        # Using the circuit:
        #   q2: U3(pi/4, 0, -pi/2)
        #   CNOT q2 -> q1
        #   q1: U3(-pi/4, 0, 0)
        #   CNOT q1 -> q0
        #   q0: U3(pi/4, 0, 0)
        #   q1: U3(pi/4, 0, 0)
        #   CNOT q1 -> q0
        #   q0: U3(-pi/4, 0, 0)
        #   q1: U3(pi/4, 0, 0)
        #   CNOT q2 -> q1
        #   q1: U3(-pi/4, 0, 0)
        #   q2: U3(pi/4, 0, 0)
        #   CNOT q1 -> q2
        #   q2: U3(-pi/4, 0, 0)
        #   q1: U3(pi/4, 0, 0)
        #   CNOT q2 -> q1
        #   q2: U3(pi/4, 0, 0)
        #   q0: U3(pi/2, 0, 0)
        #   q1: U3(pi/2, 0, 0)
        return [
            ["u3", 2, math.pi / 4, 0, -math.pi / 2],
            ["cx", 2, 1],
            ["u3", 1, -math.pi / 4, 0, 0],
            ["cx", 1, 0],
            ["u3", 0, math.pi / 4, 0, 0],
            ["u3", 1, math.pi / 4, 0, 0],
            ["cx", 1, 0],
            ["u3", 0, -math.pi / 4, 0, 0],
            ["u3", 1, math.pi / 4, 0, 0],
            ["cx", 2, 1],
            ["u3", 1, -math.pi / 4, 0, 0],
            ["u3", 2, math.pi / 4, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi / 4, 0, 0],
            ["u3", 1, math.pi / 4, 0, 0],
            ["cx", 2, 1],
            ["u3", 2, math.pi / 4, 0, 0],
            ["u3", 0, math.pi / 2, 0, 0],
            ["u3", 1, math.pi / 2, 0, 0]
        ]

    return None


def _kak_decompose_2qubit(target):
    """
    Decompose a 2-qubit unitary into single-qubit rotations and CNOTs.
    Uses the standard KAK/Cartan decomposition.
    Returns (gates, num_cnots) or (None, 0) if it fails.
    """
    # For 2-qubit gates, we can use the following approach:
    # Any 2-qubit unitary U can be written as:
    # U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F)
    # where A,B,C,D,E,F are single-qubit unitaries.
    
    # This is equivalent to the KAK decomposition:
    # U = (L1 ⊗ L2) * exp(-i * theta * (sigma_x/2 ⊗ sigma_x/2 + sigma_y/2 ⊗ sigma_y/2)) * (R1 ⊗ R2)
    
    # For a general 2-qubit gate, we need at most 3 CNOTs.
    # Let's use a simpler approach: use the fact that any 2-qubit gate can be
    # decomposed using the standard circuit synthesis.
    
    # Actually, let me use a more direct approach.
    # For 2-qubit gates with budget >= 3, we can use the standard decomposition.
    # For budget = 2, we need to check if the gate is in the "2-CNOT class".
    
    # Let me implement a general 3-CNOT decomposition.
    # U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F)
    
    # This requires solving for 6 single-qubit unitaries, which is non-trivial.
    # Instead, let me use the approach of matching the target with a parametrized circuit.
    
    # For now, return None and let the optimizer handle it.
    return None, 0


def _optimize_2qubit(target, budget, rng):
    """
    Optimize a 2-qubit circuit with the given CNOT budget.
    Uses coordinate descent on the U3 parameters.
    """
    n = 2 ** 2  # 4
    
    # Fixed CNOT layout
    if budget >= 1:
        layout = [("cx", 0, 1)] * budget
    else:
        return [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
    
    # Number of U3 slots: (budget + 1) * 2
    num_slots = (budget + 1) * 2
    values = [0.0] * (num_slots * 3)
    
    def build_circuit(vals):
        g = []
        k = 0
        # Initial U3s
        for q in range(2):
            g.append(["u3", q, vals[k], vals[k + 1], vals[k + 2]])
            k += 3
        for i in range(budget):
            g.append(["cx", 0, 1] if i % 2 == 0 else ["cx", 1, 0])
            for q in range(2):
                g.append(["u3", q, vals[k], vals[k + 1], vals[k + 2]])
                k += 3
        return g
    
    def calc_dist(vals):
        g = build_circuit(vals)
        u = _build(g, 2)
        return _distance(target, u)
    
    best_vals = values[:]
    best_dist = calc_dist(best_vals)
    
    # Coordinate descent
    max_sweeps = 8
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 10:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(len(best_vals)):
            if time.time() > _DEADLINE - 10:
                break
            # Try perturbations
            for delta in (step, -step, rng.uniform(-math.pi, math.pi)):
                v = best_vals[:]
                v[i] = best_vals[i] + delta
                d = calc_dist(v)
                if d < best_dist:
                    best_vals = v
                    best_dist = d
            if best_dist < 1e-12:
                break
        if best_dist < 1e-12:
            break
    
    return build_circuit(best_vals)


def _optimize_3qubit(target, budget, rng):
    """
    Optimize a 3-qubit circuit with the given CNOT budget.
    Uses coordinate descent on the U3 parameters.
    """
    n = 2 ** 3  # 8
    
    # Fixed CNOT layout: ladder pattern
    pairs = [(0, 1), (1, 2), (0, 1), (1, 2)]
    layout = []
    for i in range(budget):
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
    
    # Number of U3 slots: (budget + 1) * 3
    num_slots = (budget + 1) * 3
    values = [0.0] * (num_slots * 3)
    
    def build_circuit(vals):
        g = []
        k = 0
        # Initial U3s
        for q in range(3):
            g.append(["u3", q, vals[k], vals[k + 1], vals[k + 2]])
            k += 3
        for i in range(budget):
            g.append(layout[i])
            for q in range(3):
                g.append(["u3", q, vals[k], vals[k + 1], vals[k + 2]])
                k += 3
        return g
    
    def calc_dist(vals):
        g = build_circuit(vals)
        u = _build(g, 3)
        return _distance(target, u)
    
    best_vals = values[:]
    best_dist = calc_dist(best_vals)
    
    # Coordinate descent
    max_sweeps = 5
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 10:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(len(best_vals)):
            if time.time() > _DEADLINE - 10:
                break
            # Try perturbations
            for delta in (step, -step, rng.uniform(-math.pi, math.pi)):
                v = best_vals[:]
                v[i] = best_vals[i] + delta
                d = calc_dist(v)
                if d < best_dist:
                    best_vals = v
                    best_dist = d
            if best_dist < 1e-12:
                break
        if best_dist < 1e-12:
            break
    
    return build_circuit(best_vals)


def synthesize(problem):
    name = problem["name"]
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known decompositions first
    known_gates = _known_decomposition(name, qubits, target)
    if known_gates is not None:
        # Verify it's actually close
        u = _build(known_gates, qubits)
        if _distance(target, u) < 1e-9:
            return known_gates
    
    # Fallback to optimization
    rng = random.Random(4242 + qubits + hash(name) % 10000)
    
    if qubits == 2:
        gates = _optimize_2qubit(target, budget, rng)
    else:
        gates = _optimize_3qubit(target, budget, rng)
    
    return gates
# EVOLVE-BLOCK-END
