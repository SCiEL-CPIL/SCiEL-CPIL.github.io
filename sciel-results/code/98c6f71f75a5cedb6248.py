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
        # iSWAP = exp(i*pi/4 * (X⊗X + Y⊗Y)/2)
        # Known decomposition: H⊗H · CX · (S⊗I) · (Rz(π/4)⊗Rz(-π/4)) · CX · H⊗H
        # Let's try a standard decomposition
        if budget >= 2:
            # iSWAP decomposition using 2 CNOTs
            # Apply to qubits 0,1
            circuit = [
                _u3(0, 0, 0, math.pi/4),
                _u3(1, 0, 0, -math.pi/4),
                _cx(0, 1),
                _u3(0, math.pi/2, 0, 0),
                _u3(1, math.pi/2, 0, 0),
                _cx(1, 0),
                _u3(0, 0, 0, -math.pi/4),
                _u3(1, 0, 0, math.pi/4),
            ]
            if _match_target(target, circuit, 2):
                return circuit
    
    # 3-qubit gates
    if qubits == 3:
        # Toffoli: |000><000| + |001><001| + |010><010| + |011><011| + |100><100| + |101><101| + |110><111| + |111><110|
        # Standard Toffoli decomposition with 6 CNOTs
        if budget >= 6:
            # Toffoli(0,1,2) - controls 0,1, target 2
            circuit = [
                _u3(2, 0, 0, math.pi/2),
                _cx(1, 2),
                _u3(2, 0, 0, -math.pi/4),
                _cx(0, 2),
                _u3(2, 0, 0, math.pi/4),
                _cx(1, 2),
                _u3(2, 0, 0, -math.pi/4),
                _cx(0, 2),
                _u3(2, 0, 0, math.pi/4),
                _u3(2, 0, 0, -math.pi/2),
                _u3(1, 0, 0, math.pi/2),
                _cx(0, 1),
                _u3(1, 0, 0, -math.pi/4),
                _cx(0, 1),
                _u3(1, 0, 0, math.pi/4),
                _u3(1, 0, 0, -math.pi/2),
            ]
            if _match_target(target, circuit, 3):
                return circuit
            
            # Toffoli(0,2,1) - controls 0,2, target 1
            circuit2 = [
                _u3(1, 0, 0, math.pi/2),
                _cx(2, 1),
                _u3(1, 0, 0, -math.pi/4),
                _cx(0, 1),
                _u3(1, 0, 0, math.pi/4),
                _cx(2, 1),
                _u3(1, 0, 0, -math.pi/4),
                _cx(0, 1),
                _u3(1, 0, 0, math.pi/4),
                _u3(1, 0, 0, -math.pi/2),
                _u3(2, 0, 0, math.pi/2),
                _cx(0, 2),
                _u3(2, 0, 0, -math.pi/4),
                _cx(0, 2),
                _u3(2, 0, 0, math.pi/4),
                _u3(2, 0, 0, -math.pi/2),
            ]
            if _match_target(target, circuit2, 3):
                return circuit2
            
            # Toffoli(1,0,2) - controls 1,0, target 2
            circuit3 = [
                _u3(2, 0, 0, math.pi/2),
                _cx(0, 2),
                _u3(2, 0, 0, -math.pi/4),
                _cx(1, 2),
                _u3(2, 0, 0, math.pi/4),
                _cx(0, 2),
                _u3(2, 0, 0, -math.pi/4),
                _cx(1, 2),
                _u3(2, 0, 0, math.pi/4),
                _u3(2, 0, 0, -math.pi/2),
                _u3(0, 0, 0, math.pi/2),
                _cx(1, 0),
                _u3(0, 0, 0, -math.pi/4),
                _cx(1, 0),
                _u3(0, 0, 0, math.pi/4),
                _u3(0, 0, 0, -math.pi/2),
            ]
            if _match_target(target, circuit3, 3):
                return circuit3
    
    return None


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    rng = random.Random(4242 + qubits + budget)

    # 1. Try hardcoded exact circuits for common gates
    known = _try_known_circuits(target, qubits, budget)
    if known is not None:
        return known

    # 2. Use numerical optimizer with coordinate descent
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
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
    sweeps = 20
    for sweep in range(sweeps):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 1.0 / (1 + sweep * 0.5)
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
            if rng.random() < 0.3:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value

    # 3. High-precision polish pass if we have time
    if time.time() - _START_TIME < _DEADLINE - 30:
        # Use smaller steps for fine-tuning
        polish_sweeps = 10
        for sweep in range(polish_sweeps):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            step = 0.1 / (1 + sweep * 0.5)
            for i in range(n_params):
                if time.time() - _START_TIME > _DEADLINE - 10:
                    break
                for trial in (best[i] + step, best[i] - step):
                    values = best[:]
                    values[i] = trial
                    value = _build_fast(circuit(values), qubits, target)
                    if value < best_distance:
                        best, best_distance = values, value
                # Very fine steps
                for trial in (best[i] + step/10, best[i] - step/10):
                    values = best[:]
                    values[i] = trial
                    value = _build_fast(circuit(values), qubits, target)
                    if value < best_distance:
                        best, best_distance = values, value

    return circuit(best)
# EVOLVE-BLOCK-END
