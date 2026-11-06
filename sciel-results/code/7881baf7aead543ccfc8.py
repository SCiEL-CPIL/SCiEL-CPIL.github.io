"""Quantum circuit synthesis: exact known circuits + coordinate descent."""
import time
import math
import numpy as np

_T0 = time.monotonic()
_DEADLINE = 255.0


def _time_left():
    return _DEADLINE - (time.monotonic() - _T0)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(q, theta, phi, lam, nq):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
    op = np.array([[1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for k in range(nq):
        op = np.kron(op, single if k == q else eye2)
    return op


def _cx_matrix(control, target, nq):
    n = 2 ** nq
    idx = np.arange(n)
    c_bit = (idx >> (nq - 1 - control)) & 1
    t_shift = (nq - 1 - target)
    new_idx = idx ^ (c_bit << t_shift)
    op = np.zeros((n, n), dtype=complex)
    op[new_idx, idx] = 1.0
    return op


def _build_fast(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            op = _u3_matrix(q, theta, phi, lam, qubits)
        else:
            _, c, t = gate
            op = _cx_matrix(c, t, qubits)
        matrix = op @ matrix
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", int(control), int(target)]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]

    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

    if name == "iSWAP" and qubits == 2:
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


def _peel_2qubit(target, budget):
    """Exact SVD layer peeling for 2-qubit unitaries."""
    n = 4
    residual = target.copy()
    gates = []
    cnot_pattern = [(0, 1), (1, 0)]

    for layer in range(budget):
        if _time_left() < 5.0:
            break

        c, t = cnot_pattern[layer % 2]
        cx_op = _cx_matrix(c, t, 2)

        C = np.eye(2, dtype=complex)
        D = np.eye(2, dtype=complex)

        for _ in range(30):
            if _time_left() < 2.0:
                break

            cd = np.kron(C, D)
            left = residual @ cd.conj().T
            ab = left @ cx_op

            B_est = ab[0:2, 0:2]
            if np.linalg.norm(B_est) > 1e-12:
                U_p, _, Vh_p = np.linalg.svd(B_est)
                B_est = U_p @ Vh_p

                if abs(B_est[0, 0]) > 1e-12:
                    A00 = ab[0, 0] / B_est[0, 0]
                    A01 = ab[0, 2] / B_est[0, 0]
                    A10 = ab[2, 0] / B_est[0, 0]
                    A11 = ab[2, 2] / B_est[0, 0]
                    A_est = np.array([[A00, A01], [A10, A11]], dtype=complex)
                else:
                    A_est = np.eye(2, dtype=complex)
            else:
                A_est = np.eye(2, dtype=complex)
                B_est = np.eye(2, dtype=complex)

            U_p, _, Vh_p = np.linalg.svd(A_est)
            A_est = U_p @ Vh_p

            C = B_est.conj().T
            D = A_est.conj().T

            cd_new = np.kron(C, D)
            left_new = residual @ cd_new.conj().T
            ab_new = left_new @ cx_op
            error = np.linalg.norm(ab_new - ab)
            if error < 1e-10:
                break

        theta_c, phi_c, lam_c = _u3_from_matrix(C)
        theta_d, phi_d, lam_d = _u3_from_matrix(D)

        gates.append(_u3(0, theta_c, phi_c, lam_c))
        gates.append(_u3(1, theta_d, phi_d, lam_d))
        gates.append(_cx(c, t))

        cd = np.kron(C, D)
        residual = residual @ cd.conj().T @ cx_op

    if _time_left() > 1.0:
        B_est = residual[0:2, 0:2]
        if np.linalg.norm(B_est) > 1e-12:
            U_p, _, Vh_p = np.linalg.svd(B_est)
            B_est = U_p @ Vh_p
            if abs(B_est[0, 0]) > 1e-12:
                A00 = residual[0, 0] / B_est[0, 0]
                A01 = residual[0, 2] / B_est[0, 0]
                A10 = residual[2, 0] / B_est[0, 0]
                A11 = residual[2, 2] / B_est[0, 0]
                A_est = np.array([[A00, A01], [A10, A11]], dtype=complex)
            else:
                A_est = np.eye(2, dtype=complex)
        else:
            A_est = np.eye(2, dtype=complex)
            B_est = np.eye(2, dtype=complex)

        U_p, _, Vh_p = np.linalg.svd(A_est)
        A_est = U_p @ Vh_p

        theta_a, phi_a, lam_a = _u3_from_matrix(A_est)
        theta_b, phi_b, lam_b = _u3_from_matrix(B_est)

        gates.append(_u3(0, theta_a, phi_a, lam_a))
        gates.append(_u3(1, theta_b, phi_b, lam_b))

    return gates


def _u3_from_matrix(mat2):
    """Extract (theta, phi, lam) from a 2x2 unitary, up to global phase."""
    det = mat2[0, 0] * mat2[1, 1] - mat2[0, 1] * mat2[1, 0]
    if abs(det) > 1e-15:
        phase = det / abs(det)
        mat2 = mat2 / phase

    a = mat2[0, 0]
    b = mat2[0, 1]
    c = mat2[1, 0]
    d = mat2[1, 1]

    theta = 2.0 * math.atan2(abs(b), abs(a))

    if abs(b) < 1e-12 and abs(c) < 1e-12:
        if abs(a) > 1e-12 and abs(d) > 1e-12:
            phi = 0.0
            lam = math.atan2(np.imag(d / a), np.real(d / a))
        else:
            phi = 0.0
            lam = 0.0
        return (theta, phi, lam)

    if abs(b) > 1e-12:
        lam = math.atan2(-np.imag(b), -np.real(b))
    else:
        lam = 0.0

    if abs(c) > 1e-12:
        phi = math.atan2(np.imag(c), np.real(c))
    else:
        phi = 0.0

    return (theta, phi, lam)


def _make_unitary_from_params(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _coordinate_descent_2qubit(target, budget, max_iter=2000):
    """Coordinate descent for 2-qubit generic unitaries."""
    # Initialize with peeling
    gates = _peel_2qubit(target, budget)
    
    # Build initial circuit
    current = _build_fast(gates, 2)
    current_dist = _distance(target, current)
    
    # Extract parameters for optimization
    params = []
    for g in gates:
        if g[0] == 'u3':
            params.append([g[2], g[3], g[4]])
    
    def build_from_params(params):
        new_gates = []
        pi = 0
        for g in gates:
            if g[0] == 'u3':
                q = g[1]
                new_gates.append(_u3(q, params[pi][0], params[pi][1], params[pi][2]))
                pi += 1
            else:
                new_gates.append(g)
        return _build_fast(new_gates, 2)
    
    best_params = [list(p) for p in params]
    best_dist = current_dist
    
    step = 0.1
    for iteration in range(max_iter):
        if _time_left() < 3.0:
            break
        if len(params) == 0:
            break
        
        improved = False
        for i, p in enumerate(params):
            for j in range(3):  # theta, phi, lam
                old_val = p[j]
                for delta in [step, -step]:
                    p[j] = old_val + delta
                    test_circ = build_from_params(params)
                    test_dist = _distance(target, test_circ)
                    if test_dist < current_dist:
                        current_dist = test_dist
                        improved = True
                        if test_dist < best_dist:
                            best_dist = test_dist
                            best_params = [list(pp) for pp in params]
                    else:
                        p[j] = old_val
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step *= 1.05
    
    # Rebuild with best params
    best_gates = []
    pi = 0
    for g in gates:
        if g[0] == 'u3':
            q = g[1]
            best_gates.append(_u3(q, best_params[pi][0], best_params[pi][1], best_params[pi][2]))
            pi += 1
        else:
            best_gates.append(g)
    
    return best_gates, best_dist


def _coordinate_descent_3qubit(target, budget, max_iter=1500):
    """Coordinate descent for 3-qubit generic unitaries."""
    # Initialize with a simple pattern
    n = 8
    gates = []
    cnot_pattern = [(0, 1), (0, 2), (1, 2)]
    
    for layer in range(budget):
        c, t = cnot_pattern[layer % 3]
        for q in range(3):
            gates.append(_u3(q, 0.0, 0.0, 0.0))
        gates.append(_cx(c, t))
        for q in range(3):
            gates.append(_u3(q, 0.0, 0.0, 0.0))
    
    for q in range(3):
        gates.append(_u3(q, 0.0, 0.0, 0.0))
    
    current = _build_fast(gates, 3)
    current_dist = _distance(target, current)
    
    params = []
    for g in gates:
        if g[0] == 'u3':
            params.append([g[2], g[3], g[4]])
    
    def build_from_params(params):
        new_gates = []
        pi = 0
        for g in gates:
            if g[0] == 'u3':
                q = g[1]
                new_gates.append(_u3(q, params[pi][0], params[pi][1], params[pi][2]))
                pi += 1
            else:
                new_gates.append(g)
        return _build_fast(new_gates, 3)
    
    best_params = [list(p) for p in params]
    best_dist = current_dist
    
    step = 0.1
    for iteration in range(max_iter):
        if _time_left() < 3.0:
            break
        if len(params) == 0:
            break
        
        improved = False
        for i, p in enumerate(params):
            for j in range(3):
                old_val = p[j]
                for delta in [step, -step]:
                    p[j] = old_val + delta
                    test_circ = build_from_params(params)
                    test_dist = _distance(target, test_circ)
                    if test_dist < current_dist:
                        current_dist = test_dist
                        improved = True
                        if test_dist < best_dist:
                            best_dist = test_dist
                            best_params = [list(pp) for pp in params]
                    else:
                        p[j] = old_val
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step *= 1.05
    
    best_gates = []
    pi = 0
    for g in gates:
        if g[0] == 'u3':
            q = g[1]
            best_gates.append(_u3(q, best_params[pi][0], best_params[pi][1], best_params[pi][2]))
            pi += 1
        else:
            best_gates.append(g)
    
    return best_gates, best_dist


def synthesize(problem):
    start_time = time.monotonic()
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build_fast(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known

    # For generic unitaries, use coordinate descent
    if qubits == 2:
        result, dist = _coordinate_descent_2qubit(target, budget)
    else:
        result, dist = _coordinate_descent_3qubit(target, budget)

    # Verify budget constraints
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        truncated = []
        for g in result:
            if g[0] == "cx":
                if count < budget:
                    truncated.append(g)
                    count += 1
            else:
                truncated.append(g)
        result = truncated

    if len(result) > 400:
        result = result[:400]

    return result
