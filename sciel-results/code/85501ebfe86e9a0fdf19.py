"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

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

    return None


def _nelder_mead(f, x0, max_iter=200, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """Nelder-Mead simplex optimization."""
    n = len(x0)
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        step = 0.1 if x0[i] == 0 else 0.05 * abs(x0[i])
        x[i] += step
        simplex.append(x)

    values = [f(x) for x in simplex]

    for iteration in range(max_iter):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break

        # Sort simplex by function value
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if abs(values[-1] - values[0]) < tol:
            break

        # Centroid of all but worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)

        if fr < values[0]:
            # Expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = f(simplex[i])

    # Return best point
    best_idx = min(range(n + 1), key=lambda i: values[i])
    return simplex[best_idx], values[best_idx]


def _optimize_layout(layout, qubits, target, budget):
    """Optimize the u3 parameters in a fixed CNOT layout using Nelder-Mead."""
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

    # Initial guess: all zeros
    x0 = [0.0] * n_params
    best_distance = _build_fast(circuit(x0), qubits, target)

    # Try multiple random initializations
    best_x = x0[:]
    best_dist = best_distance

    for trial in range(5):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        x_init = [0.0 if i % 3 == 0 else 0.0 for i in range(n_params)]
        # Add small random perturbations
        for i in range(n_params):
            if i % 3 == 0:  # theta
                x_init[i] = (trial + 1) * 0.1
            elif i % 3 == 1:  # phi
                x_init[i] = trial * 0.3
            else:  # lam
                x_init[i] = trial * 0.2

        result_x, result_f = _nelder_mead(
            lambda x: _build_fast(circuit(x), qubits, target),
            x_init,
            max_iter=100,
            tol=1e-12
        )

        if result_f < best_dist:
            best_dist = result_f
            best_x = result_x

    # Final polish with tighter tolerance
    if best_dist > 1e-8 and time.time() - _START_TIME < _DEADLINE - 5:
        result_x, result_f = _nelder_mead(
            lambda x: _build_fast(circuit(x), qubits, target),
            best_x,
            max_iter=50,
            tol=1e-14
        )
        if result_f < best_dist:
            best_dist = result_f
            best_x = result_x

    return circuit(best_x), best_dist


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")

    # 1. Try hardcoded exact circuits for common gates
    known = _try_known_circuits(target, qubits, budget)
    if known is not None:
        return known

    # 2. Build a layout with CNOTs and u3 gates, then optimize
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]

    # Build layout: for each CNOT budget, add a layer of u3s + one CNOT
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    # Optimize the u3 parameters
    circuit, dist = _optimize_layout(layout, qubits, target, budget)

    return circuit
# EVOLVE-BLOCK-END
