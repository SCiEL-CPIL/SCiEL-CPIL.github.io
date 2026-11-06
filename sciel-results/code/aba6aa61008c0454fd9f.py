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
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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

def _is_close(A, B, tol=1e-9):
    """Check if two unitary matrices are equivalent up to global phase."""
    if A.shape != B.shape:
        return False
    # Compute overlap
    overlap = np.trace(A.conj().T @ B)
    norm = np.abs(overlap)
    n = A.shape[0]
    if norm < 1e-9:
        return False
    phase = overlap / norm
    # Check if A == phase * B
    return np.allclose(A, phase * B, atol=tol)

def _known_decomposition(name, qubits, target):
    """
    Return exact gate list for known gates if the target matches.
    Returns None if not a known gate or doesn't match.
    """
    # CNOT: |00><00| + |01><01| + |10><11| + |11><10|
    if name == "CNOT" or (qubits == 2 and np.allclose(target, np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex))):
        return [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0], ["cx", 0, 1], ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
    
    # SWAP: |00><00| + |01><10| + |10><01| + |11><11|
    if name == "SWAP" or (qubits == 2 and np.allclose(target, np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex))):
        # SWAP = 3 CNOTs
        return [
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 1, 0],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]
        ]
        
    # iSWAP: |00><00| + i|01><10| + i|10><01| + |11><11|
    if name == "iSWAP" or (qubits == 2 and np.allclose(target, np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex))):
        # iSWAP can be decomposed into 2 CNOTs and single qubit rotations
        # iSWAP = (H on both) * (CNOT 0->1) * (rotations) * (CNOT 1->0) * (H on both) ... 
        # Actually, standard decomposition:
        # iSWAP = 
        #   U3(pi/4, 0, -pi/2) on 0
        #   U3(pi/4, pi/2, 0) on 1
        #   CNOT 0->1
        #   U3(-pi/4, 0, -pi/2) on 0
        #   U3(-pi/4, pi/2, 0) on 1
        #   CNOT 1->0
        #   U3(pi/2, -pi/2, 0) on 0  (H gate equivalent? No, H is U3(pi/2, 0, pi))
        # Let's use a known exact decomposition.
        # iSWAP = 
        #   0: U3(pi/2, 0, pi) (H)
        #   1: U3(pi/2, 0, pi) (H)
        #   CNOT 0->1
        #   0: U3(pi/4, 0, -pi/2)
        #   1: U3(pi/4, pi/2, 0)
        #   CNOT 1->0
        #   0: U3(-pi/4, 0, -pi/2)
        #   1: U3(-pi/4, pi/2, 0)
        #   CNOT 0->1
        #   0: U3(pi/2, 0, pi) (H)
        #   1: U3(pi/2, 0, pi) (H)
        # This uses 3 CNOTs. Budget for iSWAP is usually 2.
        # Let's check the 2-CNOT decomposition.
        # iSWAP = 
        #   0: U3(pi/4, 0, -pi/2)
        #   1: U3(pi/4, pi/2, 0)
        #   CNOT 0->1
        #   0: U3(-pi/4, 0, -pi/2)
        #   1: U3(-pi/4, pi/2, 0)
        #   CNOT 1->0
        #   0: U3(pi/2, -pi/2, 0)
        #   1: U3(pi/2, pi/2, 0)
        # Let's just return None and let the optimizer handle it if it's not exact in my head, 
        # OR I can use a simpler known one.
        # Actually, for iSWAP, a 2-CNOT circuit is:
        # 0: U3(pi/2, 0, pi)
        # 1: U3(pi/2, 0, pi)
        # CNOT 0->1
        # 0: U3(pi/4, 0, -pi/2)
        # 1: U3(pi/4, pi/2, 0)
        # CNOT 1->0
        # 0: U3(-pi/4, 0, -pi/2)
        # 1: U3(-pi/4, pi/2, 0)
        # 0: U3(pi/2, 0, pi)
        # 1: U3(pi/2, 0, pi)
        # This is 2 CNOTs.
        return [
            ["u3", 0, math.pi/2, 0, math.pi],
            ["u3", 1, math.pi/2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 0, math.pi/4, 0, -math.pi/2],
            ["u3", 1, math.pi/4, math.pi/2, 0],
            ["cx", 1, 0],
            ["u3", 0, -math.pi/4, 0, -math.pi/2],
            ["u3", 1, -math.pi/4, math.pi/2, 0],
            ["u3", 0, math.pi/2, 0, math.pi],
            ["u3", 1, math.pi/2, 0, math.pi]
        ]

    # Toffoli: 3 qubits, 6 CNOTs
    if name == "Toffoli" or (qubits == 3 and np.allclose(target, np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0]
    ], dtype=complex))):
        # Standard 6-CNOT Toffoli decomposition
        # 2: U3(pi/4, 0, -pi/2)
        # CNOT 2->1
        # 1: U3(-pi/4, 0, 0)
        # CNOT 1->0
        # 0: U3(pi/4, 0, 0)
        # 1: U3(pi/4, 0, 0)
        # CNOT 1->0
        # 0: U3(-pi/4, 0, 0)
        # 1: U3(pi/4, 0, 0)
        # CNOT 2->1
        # 1: U3(-pi/4, 0, 0)
        # 2: U3(pi/4, 0, 0)
        # CNOT 1->2
        # 2: U3(-pi/4, 0, 0)
        # 1: U3(pi/4, 0, 0)
        # CNOT 2->1
        # 2: U3(pi/4, 0, 0)
        # 0: U3(pi/2, 0, 0)
        # 1: U3(pi/2, 0, 0)
        # Let's use a more standard one.
        # Toffoli(2,1,0)
        return [
            ["u3", 2, math.pi/4, 0, -math.pi/2],
            ["cx", 2, 1],
            ["u3", 1, -math.pi/4, 0, 0],
            ["cx", 1, 0],
            ["u3", 0, math.pi/4, 0, 0],
            ["u3", 1, math.pi/4, 0, 0],
            ["cx", 1, 0],
            ["u3", 0, -math.pi/4, 0, 0],
            ["u3", 1, math.pi/4, 0, 0],
            ["cx", 2, 1],
            ["u3", 1, -math.pi/4, 0, 0],
            ["u3", 2, math.pi/4, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/4, 0, 0],
            ["u3", 1, math.pi/4, 0, 0],
            ["cx", 2, 1],
            ["u3", 2, math.pi/4, 0, 0],
            ["u3", 0, math.pi/2, 0, 0],
            ["u3", 1, math.pi/2, 0, 0]
        ]

    return None

def _greedy_decompose(target, qubits, budget, rng):
    """
    Greedy layer-by-layer decomposition.
    """
    n = 2 ** qubits
    remaining = target.copy()
    gates = []
    
    # Pre-compute single-qubit operators
    # We will search for the best single-qubit gate on each qubit to apply after each CNOT
    
    # Layout: CNOTs are fixed in a ladder pattern
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout.append(["cx", *pairs[layer % len(pairs)]])
        
    # We will insert single-qubit gates before and after each CNOT
    # Structure: U3s -> CNOT -> U3s -> CNOT -> ... -> U3s
    
    # Initialize all U3s to identity
    num_u3_slots = (budget + 1) * qubits
    values = [0.0] * (num_u3_slots * 3)
    
    def build_circuit(vals):
        g = []
        k = 0
        # Initial U3s
        for q in range(qubits):
            g.append(["u3", q, vals[k], vals[k+1], vals[k+2]])
            k += 3
        for i, gate in enumerate(layout):
            g.append(gate)
            # U3s after CNOT (except last one, handled at end)
            for q in range(qubits):
                g.append(["u3", q, vals[k], vals[k+1], vals[k+2]])
                k += 3
        return g

    def calc_dist(vals):
        g = build_circuit(vals)
        u = _build(g, qubits)
        return _distance(target, u)

    best_vals = values[:]
    best_dist = calc_dist(best_vals)
    
    # Coordinate descent
    for sweep in range(5):
        if time.time() > _DEADLINE - 5:
            break
        step = 1.0 / (1 + sweep)
        for i in range(len(best_vals)):
            if time.time() > _DEADLINE - 5:
                break
            for trial in (best_vals[i] + step, best_vals[i] - step):
                v = best_vals[:]
                v[i] = trial
                d = calc_dist(v)
                if d < best_dist:
                    best_vals = v
                    best_dist = d
            # Random kick
            trial = best_vals[i] + rng.uniform(-math.pi, math.pi)
            v = best_vals[:]
            v[i] = trial
            d = calc_dist(v)
            if d < best_dist:
                best_vals = v
                best_dist = d

    return build_circuit(best_vals)

def synthesize(problem):
    name = problem["name"]
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known decompositions first
    known_gates = _known_decomposition(name, qubits, target)
    if known_gates is not None:
        # Verify it's actually close (just in case of name mismatch or different phase)
        u = _build(known_gates, qubits)
        if _distance(target, u) < 1e-9:
            return known_gates
        # If not close, maybe phase mismatch? The distance function handles phase.
        # If distance is small but not 0, we might still return it if it's the best we have?
        # No, better to optimize if it's not exact.
        # But for known gates, they should be exact.
    
    # Fallback to optimization
    rng = random.Random(4242 + qubits + hash(name) % 1000)
    return _greedy_decompose(target, qubits, budget, rng)
# EVOLVE-BLOCK-END
