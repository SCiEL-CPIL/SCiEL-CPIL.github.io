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
        
        # iSWAP: |00>→|00>, |01>→i|10>, |10>→i|01>, |11>→|11>
        # (H⊗H) · CX(0,1) · (S†⊗S) · CX(1,0) · (H⊗H)
        if budget >= 2:
            # iSWAP(0,1)
            circuit = [
                _u3(0, math.pi/2, 0, math.pi),
                _u3(1, math.pi/2, 0, math.pi),
                _cx(0, 1),
                _u3(0, 0, 0, -math.pi/2),
                _u3(1, 0, 0, math.pi/2),
                _cx(1, 0),
                _u3(0, math.pi/2, 0, math.pi),
                _u3(1, math.pi/2, 0, math.pi),
            ]
            if _match_target(target, circuit, 2):
                return circuit
            
            # iSWAP(1,0) - reversed
            circuit2 = [
                _u3(1, math.pi/2, 0, math.pi),
                _u3(0, math.pi/2, 0, math.pi),
                _cx(1, 0),
                _u3(1, 0, 0, -math.pi/2),
                _u3(0, 0, 0, math.pi/2),
                _cx(0, 1),
                _u3(1, math.pi/2, 0, math.pi),
                _u3(0, math.pi/2, 0, math.pi),
            ]
            if _match_target(target, circuit2, 2):
                return circuit2
    
    # 3-qubit gates
    if qubits == 3:
        # Toffoli with 6 CNOTs - try all 6 permutations of (c1, c2, t)
        if budget >= 6:
            perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
            for c1, c2, t in perms:
                circuit = [
                    _u3(t, 0, 0, math.pi/2),
                    _cx(c2, t),
                    _u3(t, 0, 0, -math.pi/4),
                    _cx(c1, t),
                    _u3(t, 0, 0, math.pi/4),
                    _cx(c2, t),
                    _u3(t, 0, 0, -math.pi/4),
                    _cx(c1, t),
                    _u3(t, 0, 0, math.pi/4),
                    _u3(t, 0, 0, -math.pi/2),
                    _u3(c2, 0, 0, math.pi/2),
                    _cx(c1, c2),
                    _u3(c2, 0, 0, -math.pi/4),
                    _cx(c1, c2),
                    _u3(c2, 0, 0, math.pi/4),
                    _u3(c2, 0, 0, -math.pi/2),
                ]
                if _match_target(target, circuit, 3):
                    return circuit
    
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

    # 2. Use numerical optimizer
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    def _make_layout(num_cx):
        layout = []
        for layer in range(num_cx):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            layout.append(["cx", *pairs[layer % len(pairs)]])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        return layout

    def _circuit_from_layout(layout, values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def _optimize(layout, target, qubits, rng, max_time=30.0):
        n_params = 3 * sum(1 for g in layout if g[0] == "u3")
        best = [0.0] * n_params
        best_distance = _build_fast(_circuit_from_layout(layout, best), qubits, target)
        
        elapsed_start = time.time()
        max_elapsed = max_time
        
        # Multi-scale coordinate descent
        scales = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        for scale in scales:
            if time.time() - elapsed_start > max_elapsed:
                break
            # Multiple passes at this scale
            for pass_num in range(3):
                if time.time() - elapsed_start > max_elapsed:
                    break
                for i in range(n_params):
                    if time.time() - elapsed_start > max_elapsed:
                        break
                    for trial in (best[i] + scale, best[i] - scale):
                        values = best[:]
                        values[i] = trial
                        value = _build_fast(_circuit_from_layout(layout, values), qubits, target)
                        if value < best_distance:
                            best, best_distance = values, value
                    # Random kick
                    if rng.random() < 0.2:
                        trial = best[i] + rng.uniform(-2*scale, 2*scale)
                        values = best[:]
                        values[i] = trial
                        value = _build_fast(_circuit_from_layout(layout, values), qubits, target)
                        if value < best_distance:
                            best, best_distance = values, value
        
        return best, best_distance

    # Try multiple layouts with different CNOT counts (up to budget)
    best_overall_params = None
    best_overall_dist = 1.0
    best_overall_layout = None
    
    time_budget_per_layout = 25.0  # seconds per layout attempt
    num_layouts = min(budget, 6)  # Try up to 6 different CNOT counts
    
    for num_cx in range(1, num_layouts + 1):
        if time.time() - _START_TIME > _DEADLINE - 20:
            break
        
        layout = _make_layout(num_cx)
        best, dist = _optimize(layout, target, qubits, rng, max_time=time_budget_per_layout)
        
        if dist < best_overall_dist:
            best_overall_dist = dist
            best_overall_params = best
            best_overall_layout = layout
    
    if best_overall_params is None:
        # Fallback: use full budget
        best_overall_layout = _make_layout(budget)
        best_overall_params, best_overall_dist = _optimize(best_overall_layout, target, qubits, rng, max_time=20.0)
    
    return _circuit_from_layout(best_overall_layout, best_overall_params)
# EVOLVE-BLOCK-END
