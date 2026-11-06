"""Rewritten program with stronger synthesis: exact known circuits + gradient-based optimization."""
import math
import time
import random

import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build the full unitary matrix from a gate list."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
            # u3 matrix: [[c, -e^{i lam} s], [e^{i phi} s, e^{i(phi+lam)} c]]
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_U = np.zeros_like(U)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_U[i, :] = c * U[i, :] - e1 * s * U[j, :]
                    new_U[j, :] = e2 * s * U[i, :] + e2 * e1 * c * U[j, :]
            U = new_U
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            U = U[perm, :]
    return U


def _distance(gates, qubits, target):
    """Phase-insensitive distance."""
    n = 2 ** qubits
    U = _build_unitary(gates, qubits)
    tr = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(tr) / n)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _match_target(target, circuit, qubits, tol=1e-9):
    """Check if circuit matches target up to global phase."""
    dist = _distance(circuit, qubits, target)
    return dist < tol


def _try_known_circuits(target, qubits, budget):
    """Try hardcoded exact circuits for standard gates."""
    if qubits == 2:
        # CNOT
        if budget >= 1:
            for ctrl, tgt in [(0, 1), (1, 0)]:
                circuit = [_cx(ctrl, tgt)]
                if _match_target(target, circuit, 2):
                    return circuit

        # SWAP
        if budget >= 3:
            circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
            if _match_target(target, circuit, 2):
                return circuit

        # iSWAP: standard 2-CNOT decomposition
        # iSWAP = (H⊗H) · CX(0,1) · (Rz(π/4)⊗Rz(-π/4)) · CX(1,0) · (H⊗H)
        # But let's use the well-known one:
        # iSWAP = (S⊗I) · (H⊗H) · CX(0,1) · (H⊗H) · (I⊗S†) · CX(1,0) · ... 
        # Actually the standard decomposition is:
        # iSWAP = (H⊗H) · CX(0,1) · (S⊗I) · (Rz(π/4)⊗Rz(-π/4)) · CX(1,0) · (H⊗H)
        # Let me use a verified one:
        if budget >= 2:
            # Standard iSWAP decomposition with 2 CNOTs
            # From Nielsen & Chuang / Qiskit:
            # iSWAP = (H⊗H) · CX(0,1) · (S⊗I) · CX(1,0) · (H⊗H) · (I⊗S)
            # Let's try multiple variants
            variants = [
                # Variant 1
                [
                    _u3(0, 0, 0, math.pi/4),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(0, 1),
                    _u3(0, math.pi/2, 0, 0),
                    _u3(1, math.pi/2, 0, 0),
                    _cx(1, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _u3(1, 0, 0, math.pi/4),
                ],
                # Variant 2
                [
                    _u3(0, math.pi/2, 0, math.pi/2),
                    _u3(1, math.pi/2, 0, math.pi/2),
                    _cx(0, 1),
                    _u3(0, 0, 0, math.pi/4),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(1, 0),
                    _u3(0, math.pi/2, 0, math.pi/2),
                    _u3(1, math.pi/2, 0, math.pi/2),
                ],
            ]
            for circuit in variants:
                if _match_target(target, circuit, 2):
                    return circuit

    if qubits == 3:
        # Toffoli with 6 CNOTs - try all 3 target qubit assignments
        if budget >= 6:
            # Toffoli(c1, c2, t) standard 6-CNOT decomposition
            toffoli_variants = [
                # Target 2, controls 0,1
                [
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
                ],
                # Target 1, controls 0,2
                [
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
                ],
                # Target 0, controls 1,2
                [
                    _u3(0, 0, 0, math.pi/2),
                    _cx(2, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(1, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _cx(2, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(1, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _u3(0, 0, 0, -math.pi/2),
                    _u3(2, 0, 0, math.pi/2),
                    _cx(1, 2),
                    _u3(2, 0, 0, -math.pi/4),
                    _cx(1, 2),
                    _u3(2, 0, 0, math.pi/4),
                    _u3(2, 0, 0, -math.pi/2),
                ],
            ]
            for circuit in toffoli_variants:
                if _match_target(target, circuit, 3):
                    return circuit

    return None


def _build_circuit_from_params(params, layout, qubits):
    """Build gate list from parameter values and fixed layout."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _optimize_generic(target, qubits, budget):
    """Optimize a generic unitary using gradient-based methods."""
    n = 2 ** qubits
    rng = random.Random(12345 + qubits * 100 + budget)

    # Build a layout with budget CNOTs and u3 gates around them
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0, 0, 0])
    for layer in range(budget):
        pair = pairs[layer % len(pairs)]
        layout.append(["cx", pair[0], pair[1]])
        for q in range(qubits):
            layout.append(["u3", q, 0, 0, 0])
    for q in range(qubits):
        layout.append(["u3", q, 0, 0, 0])

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")

    def objective(params):
        gates = _build_circuit_from_params(params, layout, qubits)
        return _distance(gates, qubits, target)

    # Multiple random restarts with L-BFGS-B
    best_params = None
    best_dist = float('inf')

    n_restarts = 8
    for r in range(n_restarts):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)
        try:
            result = minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: just use zeros
        best_params = np.zeros(n_params)
        best_dist = objective(best_params)

    return _build_circuit_from_params(best_params, layout, qubits), best_dist


def _optimize_generic_v2(target, qubits, budget):
    """Alternative layout: more u3 gates, different CNOT ordering."""
    n = 2 ** qubits
    rng = random.Random(67890 + qubits * 100 + budget)

    # Different layout: alternate CNOT directions
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0, 0, 0])
    for layer in range(budget):
        pair = pairs[(layer * 2) % len(pairs)]
        layout.append(["cx", pair[0], pair[1]])
        for q in range(qubits):
            layout.append(["u3", q, 0, 0, 0])
    for q in range(qubits):
        layout.append(["u3", q, 0, 0, 0])

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")

    def objective(params):
        gates = _build_circuit_from_params(params, layout, qubits)
        return _distance(gates, qubits, target)

    best_params = None
    best_dist = float('inf')

    n_restarts = 6
    for r in range(n_restarts):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)
        try:
            result = minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros(n_params)
        best_dist = objective(best_params)

    return _build_circuit_from_params(best_params, layout, qubits), best_dist


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")

    # 1. Try hardcoded exact circuits for common gates
    known = _try_known_circuits(target, qubits, budget)
    if known is not None:
        return known

    # 2. Use numerical optimizer
    best_circuit = None
    best_dist = float('inf')

    # Try layout v1
    circuit1, dist1 = _optimize_generic(target, qubits, budget)
    if dist1 < best_dist:
        best_dist = dist1
        best_circuit = circuit1

    # Try layout v2 if we have time
    if time.time() - _START_TIME < _DEADLINE - 30:
        circuit2, dist2 = _optimize_generic_v2(target, qubits, budget)
        if dist2 < best_dist:
            best_dist = dist2
            best_circuit = circuit2

    if best_circuit is None:
        # Emergency fallback
        best_circuit = []
        for q in range(qubits):
            best_circuit.append(_u3(q, 0, 0, 0))

    return best_circuit
