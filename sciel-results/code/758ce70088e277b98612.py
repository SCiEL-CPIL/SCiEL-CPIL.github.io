"""Rewritten program: exact algebraic synthesis for known gates, coordinate descent for generics."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START = time.time()
_DEADLINE = 225.0  # safety margin before 240s kill


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _build_fast(gates, qubits):
    """Build unitary from gate list. Qubit 0 is MSB."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator via Kronecker products
            op = np.array([[1.0 + 0j]])
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            # In our basis, qubit 0 is MSB
            # control bit position from MSB: qubits - 1 - control
            # target bit position from MSB: qubits - 1 - target
            ctrl_shift = qubits - 1 - control
            tgt_shift = qubits - 1 - target
            indices = np.arange(n)
            ctrl_bit = (indices >> ctrl_shift) & 1
            # For rows where control is 1, swap with the row where target is flipped
            # We need to permute rows: for each index, if control bit is 1, flip target bit
            new_indices = indices ^ (ctrl_bit << tgt_shift)
            matrix = matrix[new_indices, :]
    
    return matrix


def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)


def _known_decompositions(name, qubits):
    """Return exact gate lists for known gates, or None if not recognized."""
    if qubits == 2:
        if name == "CNOT":
            return [["cx", 0, 1]]
        elif name == "iSWAP":
            # iSWAP = diag(0,1,1,0) * [[0,1],[1,0]] on the middle subspace... 
            # Actually iSWAP maps |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
            # Standard decomposition: 
            # |00> -> |00>
            # |01> -> i|10>
            # |10> -> i|01>
            # |11> -> |11>
            # One known decomposition:
            # H on q1, CNOT(0,1), H on q1, then phase gates
            # Actually let's use: 
            # iSWAP = (I ⊗ H) · CNOT(0,1) · (I ⊗ H) · (S ⊗ S) · (I ⊗ H) · CNOT(1,0) · (I ⊗ H)
            # Hmm, let me think more carefully.
            # iSWAP matrix in basis |00>,|01>,|10>,|11>:
            # [[1, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 1]]
            # 
            # Known 2-CNOT decomposition:
            # Step 1: Apply H to qubit 1
            # Step 2: CNOT(0,1)
            # Step 3: Apply H to qubit 1
            # Step 4: Apply S (phase) to qubit 0 and qubit 1
            # Step 5: Apply H to qubit 1
            # Step 6: CNOT(1,0)
            # Step 7: Apply H to qubit 1
            #
            # Let me verify with a simpler approach. The iSWAP can be decomposed as:
            # iSWAP = (S⊗S) · (H⊗H) · CNOT(0,1) · (H⊗H) · CNOT(1,0) · (H⊗H)
            # Hmm, this is getting complex. Let me use a verified decomposition.
            #
            # Actually, a well-known 2-CNOT iSWAP decomposition:
            gates = [
                ["u3", 0, 0.0, 0.0, math.pi / 2],  # S on q0
                ["u3", 1, 0.0, 0.0, math.pi / 2],  # S on q1
                ["u3", 1, math.pi / 2, 0.0, 0.0],  # H-like on q1 (actually this is not H)
            ]
            # Let me just use the standard result. iSWAP requires 2 CNOTs minimum.
            # A known decomposition:
            # 1. u3(1, 0, pi/2) on q1  -> this is... let me be more careful
            # 
            # Let me use a different approach: just return None and let coordinate descent handle it.
            # Actually, for safety, let me return None for iSWAP and SWAP and let the optimizer work.
            return None
        elif name == "SWAP":
            # SWAP = CNOT(0,1) · CNOT(1,0) · CNOT(0,1)
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif qubits == 3:
        if name == "Toffoli":
            # Toffoli (CCNOT) with control q0,q1 and target q2
            # Known 6-CNOT decomposition (or 7 depending on source)
            # A standard 6-CNOT Toffoli decomposition:
            gates = [
                ["u3", 2, math.pi / 4, 0.0, 0.0],
                ["cx", 1, 2],
                ["u3", 2, -math.pi / 4, 0.0, 0.0],
                ["cx", 0, 2],
                ["u3", 2, math.pi / 4, 0.0, 0.0],
                ["cx", 1, 2],
                ["u3", 2, -math.pi / 4, 0.0, 0.0],
                ["u3", 2, 0.0, 0.0, -math.pi / 2],
                ["cx", 0, 1],
                ["u3", 1, 0.0, 0.0, -math.pi / 2],
                ["u3", 1, math.pi / 4, 0.0, 0.0],
                ["cx", 2, 1],
                ["u3", 1, -math.pi / 4, 0.0, 0.0],
                ["cx", 0, 1],
                ["u3", 1, math.pi / 2, 0.0, 0.0],
                ["u3", 2, math.pi / 2, 0.0, 0.0],
            ]
            # This is a known decomposition but let me count CNOTs: 6 CNOTs
            # Actually, let me just return None and let the optimizer handle Toffoli too,
            # since getting the exact decomposition right is tricky.
            return None
    return None


def _optimize_generic(target, qubits, budget, max_time):
    """Coordinate descent optimization for a generic unitary."""
    n_params_per_u3 = 3
    # Layout: alternating u3 layers and CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    
    # Build layout: for each CNOT in budget, we have a layer of u3s before it
    layout = []
    num_u3_layers = budget + 1  # u3 layer before each CNOT and one after
    for layer in range(num_u3_layers):
        layout.append(("u3_layer", list(range(qubits))))
        if layer < budget:
            layout.append(("cx", pairs[layer % len(pairs)]))
    
    # Count total u3 parameters
    total_u3 = num_u3_layers * qubits
    total_params = total_u3 * n_params_per_u3
    
    # Initialize with zeros
    best = [0.0] * total_params
    best_dist = _distance(target, _build_fast(_make_gates(layout, best, qubits), qubits))
    
    if best_dist < 1e-10:
        return _make_gates(layout, best, qubits)
    
    # Coordinate descent with adaptive step
    step = 0.5
    no_improve_count = 0
    
    while _remaining() > 5.0:
        improved = False
        for i in range(total_params):
            if _remaining() < 5.0:
                break
            # Try several perturbations
            for trial_val in (best[i] + step, best[i] - step,
                              best[i] + rng.uniform(-step * 2, step * 2)):
                trial_val = (trial_val + math.pi) % (2 * math.pi) - math.pi
                values = best[:]
                values[i] = trial_val
                gates = _make_gates(layout, values, qubits)
                dist = _distance(target, _build_fast(gates, qubits))
                if dist < best_dist - 1e-12:
                    best, best_dist = values, dist
                    improved = True
                    no_improve_count = 0
                    break
        if not improved:
            no_improve_count += 1
            step *= 0.8
            if step < 1e-6:
                break
        else:
            no_improve_count = 0
    
    return _make_gates(layout, best, qubits)


def _make_gates(layout, params, qubits):
    """Convert layout and parameter values into a gate list."""
    gates = []
    k = 0
    for item in layout:
        if item[0] == "u3_layer":
            for q in item[1]:
                theta = params[k]
                phi = params[k + 1]
                lam = params[k + 2]
                gates.append(["u3", q, theta, phi, lam])
                k += 3
        else:
            gates.append(["cx", item[1], item[2]])
    return gates


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decomposition for known gates
    known = _known_decompositions(name, qubits)
    if known is not None:
        # Verify the known decomposition
        built = _build_fast(known, qubits)
        dist = _distance(target, built)
        if dist < 1e-8:
            return known
        # If the known decomposition doesn't match, fall through to optimization
    
    # For generic unitaries or failed known decompositions, use coordinate descent
    gates = _optimize_generic(target, qubits, budget, max_time=_remaining())
    return gates
# EVOLVE-BLOCK-END
