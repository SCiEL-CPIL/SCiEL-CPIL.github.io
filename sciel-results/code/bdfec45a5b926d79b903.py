"""Quantum circuit synthesis with known gate decompositions and optimized generic unitaries."""
# EVOLVE-BLOCK-START
import math
import random

import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
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
    """Phase-insensitive distance."""
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0, math.pi)


def _s(q):
    return _u3(q, 0, 0, math.pi / 2)


def _t(q):
    return _u3(q, 0, 0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0, 0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0, 0, -math.pi / 2)


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]
    
    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    
    if name == "iSWAP" and qubits == 2:
        # iSWAP = 0.5 * [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, -1], [0, 1, -1, 0]] (up to phase)
        # Standard decomposition: H on both, then CNOT(0,1), then S on target, CNOT(0,1), S^dag on target, H on both
        # Actually iSWAP = (H⊗H) · CNOT(0,1) · (I⊗S) · CNOT(0,1) · (I⊗S†) · (H⊗H) up to global phase
        # Let me use a cleaner decomposition
        # iSWAP = diag(1, -1) on control/target with CNOT structure
        # Standard: CNOT(1,0), then S on q0, S† on q1, CNOT(1,0)
        # But let me verify with a known decomposition
        # iSWAP = (1/sqrt(2)) * [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, -1], [0, 1, -1, 0]]
        # Using: H0 H1 CNOT(0,1) S1 CNOT(0,1) S†1 H0 H1 gives a related gate
        # Let me use: CNOT(1,0), S0, S†1, CNOT(1,0) and check
        # Actually, the standard iSWAP decomposition with 2 CNOTs:
        # H(0), H(1), CNOT(0,1), S(1), CNOT(0,1), S†(1), H(0), H(1)
        # This gives iSWAP up to global phase
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates
    
    if name == "Toffoli" and qubits == 3:
        # Toffoli (CCNOT) with 6 CNOTs
        # Standard decomposition:
        # T(2), CNOT(1,2), T†(2), CNOT(0,2), T(1), T(2), CNOT(1,2), T†(1), T†(2), CNOT(0,2), T(0), T(2), CNOT(0,1), T†(0), T(1), CNOT(1,2), T†(2)
        # That's 6 CNOTs
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates
    
    return None


def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using coordinate descent with better initialization."""
    # Create a layout with CNOTs and single-qubit gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(rng_seed)
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Try multiple random initializations
    best_overall = None
    best_dist_overall = 1.0
    
    for init_trial in range(3):
        # Initialize with random values
        best = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        best_distance = _distance(target, _build(circuit(best), qubits))
        
        # Coordinate descent with multiple sweeps
        for sweep in range(12):
            step = 1.0 / (1 + sweep * 0.5)
            improved = False
            for i in range(n_params):
                # Try several perturbations around current value
                for trial in (best[i] + step, best[i] - step,
                             best[i] + step * 0.5, best[i] - step * 0.5,
                             best[i] + rng.uniform(-math.pi, math.pi) * 0.05):
                    values = best[:]
                    values[i] = trial
                    value = _distance(target, _build(circuit(values), qubits))
                    if value < best_distance:
                        best, best_distance = values, value
                        improved = True
            # If we're close enough, break
            if best_distance < 1e-10:
                break
            # If no improvement in this sweep, reduce step size
            if not improved:
                step *= 0.5
        
        if best_distance < best_dist_overall:
            best_dist_overall = best_distance
            best_overall = best[:]
        
        if best_dist_overall < 1e-10:
            break
    
    if best_overall is None:
        best_overall = [0.0] * n_params
    
    return circuit(best_overall)


def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        # Verify the known circuit is within budget
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            # Check if it's actually correct (up to phase)
            built = _build(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # For generic or incorrect known circuits, use optimization
    return _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100 + budget)
# EVOLVE-BLOCK-END
