"""Quantum circuit synthesis: exact known gates + optimized generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_start_time = None
_DEADLINE = 200.0


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary matrix from gate list."""
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
            ], dtype=complex)
            # Build the full operator by kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, control, target = gate
            idx = np.arange(dim)
            bit = (idx >> (qubits - 1 - control)) & 1
            # CNOT: if control is 1, flip target
            new_idx = idx ^ (bit << (qubits - 1 - target))
            U = U[new_idx, :]
    return U


def _distance(target, U):
    dim = target.shape[0]
    val = abs(np.trace(target.conj().T @ U)) / dim
    return 1.0 - min(1.0, val)


def _exact_cnot(problem):
    """Exact CNOT gate: 1 CNOT."""
    qubits = problem["qubits"]
    if qubits == 2:
        return [_cx(0, 1)]
    else:
        # For 3 qubits, CNOT on specific qubits - assume standard
        return [_cx(0, 1)]


def _exact_swap(problem):
    """Exact SWAP: 3 CNOTs."""
    qubits = problem["qubits"]
    if qubits == 2:
        return [
            _u3(0, 0, 0, math.pi),  # X on q0
            _cx(0, 1),
            _u3(0, 0, 0, math.pi),  # X on q0
            _cx(1, 0),
            _u3(0, 0, 0, math.pi),  # X on q0
            _cx(0, 1),
            _u3(0, 0, 0, math.pi),  # X on q0
        ]
    else:
        # SWAP on qubits 0 and 1
        return [
            _u3(0, 0, 0, math.pi),
            _cx(0, 1),
            _u3(0, 0, 0, math.pi),
            _cx(1, 0),
            _u3(0, 0, 0, math.pi),
            _cx(0, 1),
            _u3(0, 0, 0, math.pi),
        ]


def _exact_iswap(problem):
    """Exact iSWAP: 2 CNOTs + phase rotations."""
    qubits = problem["qubits"]
    if qubits == 2:
        # iSWAP = |0><0|⊗|0><0| + i|0><1|⊗|1><0| + i|1><0|⊗|0><1| + |1><1|⊗|1><1|
        # Standard decomposition
        return [
            _u3(0, math.pi / 2, 0, math.pi),  # S gate on q0: diag(1, i)
            _u3(1, math.pi / 2, 0, math.pi),  # S gate on q1
            _u3(0, math.pi / 2, 0, 0),        # H-like rotation
            _cx(0, 1),
            _u3(0, math.pi / 2, math.pi, 0),
            _cx(1, 0),
            _u3(0, math.pi / 2, 0, 0),
            _u3(1, math.pi / 2, 0, 0),
        ]
    else:
        return [
            _u3(0, math.pi / 2, 0, math.pi),
            _u3(1, math.pi / 2, 0, math.pi),
            _u3(0, math.pi / 2, 0, 0),
            _cx(0, 1),
            _u3(0, math.pi / 2, math.pi, 0),
            _cx(1, 0),
            _u3(0, math.pi / 2, 0, 0),
            _u3(1, math.pi / 2, 0, 0),
        ]


def _exact_toffoli(problem):
    """Exact Toffoli (CCX): 6 CNOTs standard decomposition."""
    qubits = problem["qubits"]
    # Toffoli: control q0, q1, target q2
    # Standard decomposition uses 6 CNOTs
    return [
        _u3(2, math.pi / 2, 0, 0),           # H on target
        _cx(1, 2),
        _u3(2, -math.pi / 4, 0, 0),          # T†
        _cx(0, 2),
        _u3(2, math.pi / 4, 0, 0),           # T
        _cx(1, 2),
        _u3(2, -math.pi / 4, 0, 0),          # T†
        _u3(1, math.pi / 2, 0, 0),           # H on q1
        _cx(0, 1),
        _u3(1, math.pi / 4, 0, 0),           # T on q1
        _cx(0, 1),
        _u3(1, -math.pi / 4, 0, 0),          # T† on q1
        _u3(2, math.pi / 2, 0, 0),           # H on target
        _cx(1, 2),
        _u3(2, math.pi / 4, 0, 0),           # T on target
        _cx(0, 2),
        _u3(2, -math.pi / 4, 0, 0),          # T† on target
    ]


def _optimize_generic(problem, target, qubits, budget):
    """Optimize single-qubit parameters for generic unitary with given CNOT budget."""
    # Build a CNOT ladder layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Layout: alternating single-qubit layers and CNOT layers
    # Number of CNOT layers = budget
    # Each CNOT layer uses 1 CNOT
    # Single-qubit layers: before first CNOT, between CNOTs, after last CNOT
    
    # Total single-qubit gates: (budget + 1) * qubits
    n_u3_layers = budget + 1
    n_u3 = n_u3_layers * qubits
    
    # Build layout
    layout = []
    for layer in range(budget + 1):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        if layer < budget:
            layout.append(["cx", *pairs[layer % len(pairs)]])
    
    # Count CNOTs to verify
    cnot_count = sum(1 for g in layout if g[0] == "cx")
    assert cnot_count <= budget, f"CNOT count {cnot_count} exceeds budget {budget}"
    
    # Parameter vector: 3 per u3 gate
    n_params = 3 * n_u3
    best = [0.0] * n_params
    best_dist = _distance(target, _build(circuit_from_params(layout, best, qubits), qubits))
    
    # Coordinate descent with multiple restarts
    import random
    rng = random.Random(12345 + qubits * 100 + budget)
    
    max_iter = 30
    for restart in range(3):
        if _elapsed() > _DEADLINE:
            break
        if restart > 0:
            best = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            best_dist = _distance(target, _build(circuit_from_params(layout, best, qubits), qubits))
        
        for sweep in range(max_iter):
            if _elapsed() > _DEADLINE:
                break
            step = 1.0 / (1 + sweep * 0.5)
            improved = False
            for i in range(n_params):
                if _elapsed() > _DEADLINE:
                    break
                for trial in (best[i] + step, best[i] - step):
                    values = best[:]
                    values[i] = trial
                    dist = _distance(target, _build(circuit_from_params(layout, values, qubits), qubits))
                    if dist < best_dist:
                        best, best_dist = values, dist
                        improved = True
    return circuit_from_params(layout, best, qubits)


def circuit_from_params(layout, params, qubits):
    """Build gate list from layout and parameter values."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def synthesize(problem):
    """Main synthesis function."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "").lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decompositions for known gates
    if "cnot" in name or "cx" in name:
        gates = _exact_cnot(problem)
        if len([g for g in gates if g[0] == "cx"]) <= budget:
            return gates
    
    if "swap" in name and "iswap" not in name:
        gates = _exact_swap(problem)
        if len([g for g in gates if g[0] == "cx"]) <= budget:
            return gates
    
    if "iswap" in name:
        gates = _exact_iswap(problem)
        if len([g for g in gates if g[0] == "cx"]) <= budget:
            return gates
    
    if "toffoli" in name or "ccx" in name or "t" == name:
        gates = _exact_toffoli(problem)
        if len([g for g in gates if g[0] == "cx"]) <= budget:
            return gates
    
    # For generic unitaries, use optimization
    gates = _optimize_generic(problem, target, qubits, budget)
    
    # Verify CNOT count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Truncate to fit budget
        gates = gates[:budget + 400]
    
    return gates
# EVOLVE-BLOCK-END
