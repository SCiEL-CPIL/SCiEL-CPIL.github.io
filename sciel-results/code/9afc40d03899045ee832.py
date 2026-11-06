"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass, efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
            # u3 = [[c, -e1*s], [e2*s, e2*e1*c]]
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


def _match_target(target, circuit, qubits, tol=1e-8):
    """Check if target matches the unitary of circuit (up to global phase)."""
    dist = _build_fast(circuit, qubits, target)
    return dist < tol


def _try_known_circuits(target, qubits, budget):
    """Try hardcoded exact circuits for standard gates."""
    # 2-qubit gates
    if qubits == 2:
        # CNOT: |00><00| + |01><01| + |10><11| + |11><10|
        if budget >= 1:
            for ctrl, tgt in [(0, 1), (1, 0)]:
                circuit = [_cx(ctrl, tgt)]
                if _match_target(target, circuit, 2):
                    return circuit
        
        # SWAP: |00><00| + |01><10| + |10><01| + |11><11|
        if budget >= 3:
            circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
            if _match_target(target, circuit, 2):
                return circuit
        
        # iSWAP: |00><00| + 0.5i*|01><10| + 0.5i*|10><01| + |11><11|
        # Known decomposition using H, S, and CNOT
        if budget >= 2:
            # iSWAP = (H ⊗ H) · CX · (S ⊗ I) · CX · (H ⊗ H) with phase corrections
            # More precisely: iSWAP = diag(1, i, i, 1) · SWAP
            # SWAP = CX(0,1) · CX(1,0) · CX(0,1)
            # We need to add iSWAP = e^{iπ/4} · (H⊗H) · CX · (S⊗I) · CX · (H⊗H)
            # Let's use the standard decomposition:
            # iSWAP = (Rz(π/4) ⊗ Rz(-π/4)) · CX · (S ⊗ I) · CX · (Rz(-π/4) ⊗ Rz(π/4))
            # Actually, let me use a simpler approach:
            # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
            # This can be decomposed as:
            # H(0) · H(1) · CX(0,1) · S(0) · CX(0,1) · H(0) · H(1)
            # But we need to check the exact phases.
            
            # Standard iSWAP decomposition:
            # iSWAP = (H ⊗ H) · CX(0,1) · (S ⊗ I) · CX(0,1) · (H ⊗ H)
            # where S = diag(1, i)
            # Let me construct this:
            # H = u3(π/2, 0, π)
            # S = u3(0, 0, π/2)
            
            h_gate = _u3(0, math.pi/2, 0, math.pi)
            h_gate1 = _u3(1, math.pi/2, 0, math.pi)
            s_gate = _u3(0, 0, 0, math.pi/2)
            
            circuit = [
                h_gate, h_gate1,
                _cx(0, 1),
                s_gate,
                _cx(0, 1),
                h_gate, h_gate1
            ]
            if _match_target(target, circuit, 2):
                return circuit
    
    # 3-qubit gates
    if qubits == 3:
        # Toffoli: |000><000| + |001><001| + |010><010| + |011><011| + |100><100| + |101><101| + |110><110| + |111><110|
        # Standard Toffoli decomposition using 6 CNOTs
        if budget >= 6:
            # Toffoli(0,1,2) decomposition:
            # H(2) · CX(0,2) · T(2) · CX(1,2) · T†(2) · T(1) · CX(0,2) · T(2) · T†(1) · CX(1,2) · T†(2) · H(2)
            # where T = diag(1, e^{iπ/4}) = u3(0, 0, π/4)
            # T† = diag(1, e^{-iπ/4}) = u3(0, 0, -π/4)
            
            h_gate = _u3(2, math.pi/2, 0, math.pi)
            t_gate = _u3(2, 0, 0, math.pi/4)
            t_dag_gate = _u3(2, 0, 0, -math.pi/4)
            t_gate1 = _u3(1, 0, 0, math.pi/4)
            t_dag_gate1 = _u3(1, 0, 0, -math.pi/4)
            
            circuit = [
                h_gate,
                _cx(0, 2),
                t_gate,
                _cx(1, 2),
                t_dag_gate,
                t_gate1,
                _cx(0, 2),
                t_gate,
                t_dag_gate1,
                _cx(1, 2),
                t_dag_gate,
                h_gate
            ]
            if _match_target(target, circuit, 3):
                return circuit
            
            # Try Toffoli with different control/target ordering
            # Toffoli(1,0,2)
            h_gate2 = _u3(2, math.pi/2, 0, math.pi)
            circuit2 = [
                h_gate2,
                _cx(1, 2),
                t_gate,
                _cx(0, 2),
                t_dag_gate,
                t_gate1,
                _cx(1, 2),
                t_gate,
                t_dag_gate1,
                _cx(0, 2),
                t_dag_gate,
                h_gate2
            ]
            if _match_target(target, circuit2, 3):
                return circuit2
    
    return None

def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    rng = random.Random(4242 + qubits + budget)

    # 1. Try hardcoded exact circuits for common gates
    known_circuit = _try_known_circuits(target, qubits, budget)
    if known_circuit is not None:
        return known_circuit

    # 2. Use numerical optimizer with improved layout and more sweeps
    # Create a better initial layout: alternate CNOT patterns for better mixing
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # For 2 qubits: use CX(0,1), CX(1,0) alternating
    # For 3 qubits: use a more complete pattern
    if qubits == 2:
        cnot_pattern = [(0, 1), (1, 0)]
    elif qubits == 3:
        cnot_pattern = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    else:
        cnot_pattern = pairs
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *cnot_pattern[layer % len(cnot_pattern)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _build_fast(circuit(best), qubits, target)

    # Coordinate descent with adaptive step sizes and deadline guard
    # Increase sweeps since we're spending less time on known gates
    sweeps = 30
    for sweep in range(sweeps):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 1.0 / (1 + sweep * 0.3)  # Slower decay for more exploration
        for i in range(n_params):
            if time.time() - _START_TIME > _DEADLINE:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
            # Also try a random perturbation occasionally
            if rng.random() < 0.2:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value

    return circuit(best)
# EVOLVE-BLOCK-END
