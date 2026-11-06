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
        # iSWAP(0,1): (H⊗H) · CX(0,1) · (S†⊗S) · CX(1,0) · (H⊗H)
        # H = u3(pi/2, 0, pi)
        # S = u3(0, 0, pi/2)
        # S† = u3(0, 0, -pi/2)
        if budget >= 2:
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
            
            # Toffoli(1,2,0) - controls 1,2, target 0
            circuit4 = [
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
            ]
            if _match_target(target, circuit4, 3):
                return circuit4
            
            # Toffoli(2,0,1) - controls 2,0, target 1
            circuit5 = [
                _u3(1, 0, 0, math.pi/2),
                _cx(0, 1),
                _u3(1, 0, 0, -math.pi/4),
                _cx(2, 1),
                _u3(1, 0, 0, math.pi/4),
                _cx(0, 1),
                _u3(1, 0, 0, -math.pi/4),
                _cx(2, 1),
                _u3(1, 0, 0, math.pi/4),
                _u3(1, 0, 0, -math.pi/2),
                _u3(0, 0, 0, math.pi/2),
                _cx(2, 0),
                _u3(0, 0, 0, -math.pi/4),
                _cx(2, 0),
                _u3(0, 0, 0, math.pi/4),
                _u3(0, 0, 0, -math.pi/2),
            ]
            if _match_target(target, circuit5, 3):
                return circuit5
            
            # Toffoli(2,1,0) - controls 2,1, target 0
            circuit6 = [
                _u3(0, 0, 0, math.pi/2),
                _cx(1, 0),
                _u3(0, 0, 0, -math.pi/4),
                _cx(2, 0),
                _u3(0, 0, 0, math.pi/4),
                _cx(1, 0),
                _u3(0, 0, 0, -math.pi/4),
                _cx(2, 0),
                _u3(0, 0, 0, math.pi/4),
                _u3(0, 0, 0, -math.pi/2),
                _u3(1, 0, 0, math.pi/2),
                _cx(2, 1),
                _u3(1, 0, 0, -math.pi/4),
                _cx(2, 1),
                _u3(1, 0, 0, math.pi/4),
                _u3(1, 0, 0, -math.pi/2),
            ]
            if _match_target(target, circuit6, 3):
                return circuit6
    
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

    # 2. Build circuit layout
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
    n = 2 ** qubits

    # Fast objective: compute fidelity without building full matrix each time
    # Use the fact that we only need |tr(target^dag U)|/n
    def objective(values):
        gates = circuit(values)
        return _build_fast(gates, qubits, target)

    # Nelder-Mead simplex method
    def nelder_mead(f, x0, max_iter=200, tol=1e-10):
        n = len(x0)
        # Initialize simplex
        simplex = [x0[:]]
        for i in range(n):
            point = x0[:]
            step = 0.5 if x0[i] == 0 else abs(x0[i]) * 0.5
            point[i] += step
            simplex.append(point)

        # Evaluate initial simplex
        fvals = [f(x) for x in simplex]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for iteration in range(max_iter):
            if time.time() - _START_TIME > _DEADLINE:
                break

            # Sort simplex by function value
            order = sorted(range(n + 1), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break

            # Centroid of all but worst point
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

            # Reflection
            xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = f(xr)

            if fr < fvals[0]:
                # Expansion
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
                else:
                    # Inside contraction
                    xc = [centroid[j] + rho * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = f(simplex[i])

        # Return best point
        best_idx = min(range(n + 1), key=lambda i: fvals[i])
        return simplex[best_idx], fvals[best_idx]

    # Multi-start Nelder-Mead
    best_params = [0.0] * n_params
    best_distance = objective(best_params)

    n_starts = 8
    for start in range(n_starts):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break

        # Different initializations
        if start == 0:
            x0 = [0.0] * n_params
        elif start == 1:
            x0 = [math.pi / 4 * (1 if i % 2 == 0 else -1) for i in range(n_params)]
        elif start == 2:
            x0 = [math.pi / 2 * (1 if i % 3 == 0 else -1) for i in range(n_params)]
        else:
            x0 = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]

        # Limit iterations based on remaining time
        remaining = _DEADLINE - time.time()
        max_iter = max(50, int(remaining * 10))

        result, dist = nelder_mead(objective, x0, max_iter=max_iter)
        if dist < best_distance:
            best_params, best_distance = result, dist

    # Final polish with very tight tolerance
    if time.time() - _START_TIME < _DEADLINE - 10:
        remaining = _DEADLINE - time.time()
        max_iter = max(30, int(remaining * 5))
        result, dist = nelder_mead(objective, best_params, max_iter=max_iter, tol=1e-12)
        if dist < best_distance:
            best_params, best_distance = result, dist

    return circuit(best_params)
# EVOLVE-BLOCK-END
