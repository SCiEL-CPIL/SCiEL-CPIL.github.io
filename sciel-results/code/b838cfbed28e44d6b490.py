"""Quantum circuit synthesis: exact algebraic decompositions + fast coordinate descent."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0  # Safety margin before 270s wall-clock kill


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]], dtype=complex)


def _precompute_cnot_perm(qubits):
    """Precompute CNOT permutation matrices for all (control, target) pairs."""
    perms = {}
    n = 2 ** qubits
    for c in range(qubits):
        for t in range(qubits):
            if c == t:
                continue
            idx = np.arange(n)
            c_bit = (idx >> (qubits - 1 - c)) & 1
            perm = idx ^ (c_bit << (qubits - 1 - t))
            perms[(c, t)] = perm
    return perms


def _build_fast(gates, qubits, cnot_perms):
    """Build the full unitary from a gate list using precomputed permutations."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)

    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            perm = cnot_perms[(control, target)]
            matrix = matrix[perm, :]

    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / 2^qubits"""
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _solve_u3_params(M):
    """
    Solve for theta, phi, lam such that u3(theta, phi, lam) = M (up to global phase).
    u3 = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
          [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    """
    det = np.linalg.det(M)
    if abs(det) > 1e-12:
        global_phase = np.sqrt(det)
        U = M / global_phase
    else:
        U = M

    cos_half = np.real(U[0, 0])
    cos_half = np.clip(cos_half, -1.0, 1.0)
    sin_half_abs = np.sqrt(max(0, 1 - cos_half ** 2))

    if sin_half_abs < 1e-10:
        if abs(cos_half - 1) < 1e-10:
            return 0.0, 0.0, 0.0
        else:
            lam = -np.angle(U[0, 1])
            phi = np.angle(U[1, 0])
            return math.pi, phi, lam

    theta = 2 * np.arccos(cos_half)
    sin_half = np.sin(theta / 2)

    lam = np.angle(-U[0, 1] / sin_half)
    phi = np.angle(U[1, 0] / sin_half)

    return float(theta), float(phi), float(lam)


def _u3_from_matrix(M, q):
    theta, phi, lam = _solve_u3_params(M)
    return [['u3', q, theta, phi, lam]]


def _is_product_of_two(U):
    """Check if a 4x4 unitary is a product of two 2x2 unitaries."""
    if abs(U[0, 0]) < 1e-10:
        if abs(U[0, 2]) < 1e-10:
            return False
        A00 = U[0, 0]
        A01 = U[0, 2]
        A10 = U[2, 0]
        A11 = U[2, 2]
        A = np.array([[A00, A01], [A10, A11]], dtype=complex)

        if abs(A00) < 1e-10:
            return False
        B00 = 1
        B01 = U[0, 1] / A00
        B10 = U[1, 0] / A00
        B11 = U[1, 1] / A00
        B = np.array([[B00, B01], [B10, B11]], dtype=complex)

        recon = np.kron(A, B)
        return np.max(np.abs(recon - U)) < 1e-8

    A00 = 1
    A01 = U[0, 2] / U[0, 0]
    A10 = U[2, 0] / U[0, 0]
    A11 = U[2, 2] / U[0, 0]
    A = np.array([[A00, A01], [A10, A11]], dtype=complex)

    B00 = U[0, 0]
    B01 = U[0, 1]
    B10 = U[1, 0]
    B11 = U[1, 1]
    B = np.array([[B00, B01], [B10, B11]], dtype=complex)

    recon = np.kron(A, B)
    if np.max(np.abs(recon - U)) < 1e-8:
        return True

    # Try other gauge: B[0,0] = 1
    A00 = U[0, 0]
    A01 = U[0, 2]
    A10 = U[2, 0]
    A11 = U[2, 2]
    A = np.array([[A00, A01], [A10, A11]], dtype=complex)

    B00 = 1
    B01 = U[0, 1] / A00 if abs(A00) > 1e-10 else 0
    B10 = U[1, 0] / A00 if abs(A00) > 1e-10 else 0
    B11 = U[1, 1] / A00 if abs(A00) > 1e-10 else 0
    B = np.array([[B00, B01], [B10, B11]], dtype=complex)

    recon = np.kron(A, B)
    return np.max(np.abs(recon - U)) < 1e-8


def _decompose_product(U):
    """Decompose U = A ⊗ B. Returns (A, B)."""
    if abs(U[0, 0]) < 1e-10:
        if abs(U[0, 2]) < 1e-10:
            # Fallback
            A = np.eye(2, dtype=complex)
            B = np.eye(2, dtype=complex)
            return A, B
        A00 = U[0, 0]
        A01 = U[0, 2]
        A10 = U[2, 0]
        A11 = U[2, 2]
        A = np.array([[A00, A01], [A10, A11]], dtype=complex)

        if abs(A00) < 1e-10:
            A = np.eye(2, dtype=complex)
            B = np.eye(2, dtype=complex)
            return A, B
        B00 = 1
        B01 = U[0, 1] / A00
        B10 = U[1, 0] / A00
        B11 = U[1, 1] / A00
        B = np.array([[B00, B01], [B10, B11]], dtype=complex)
        return A, B

    A00 = 1
    A01 = U[0, 2] / U[0, 0]
    A10 = U[2, 0] / U[0, 0]
    A11 = U[2, 2] / U[0, 0]
    A = np.array([[A00, A01], [A10, A11]], dtype=complex)

    B00 = U[0, 0]
    B01 = U[0, 1]
    B10 = U[1, 0]
    B11 = U[1, 1]
    B = np.array([[B00, B01], [B10, B11]], dtype=complex)
    return A, B


def _cnot_matrix(c, t, qubits):
    """Build CNOT matrix for control c, target t on qubits."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    idx = np.arange(n)
    bit = (idx >> (qubits - 1 - c)) & 1
    perm = idx ^ (bit << (qubits - 1 - t))
    U = U[perm, :]
    return U


def _h(q):
    return ['u3', q, math.pi / 2, 0, math.pi]


def _s(q):
    return ['u3', q, 0, 0, math.pi / 2]


def _sdag(q):
    return ['u3', q, 0, 0, -math.pi / 2]


def _t(q):
    return ['u3', q, math.pi / 4, 0, 0]


def _tdag(q):
    return ['u3', q, -math.pi / 4, 0, 0]


def _cnot(c, t):
    return ['cx', c, t]


def _exact_cnot_decomposition(c, t):
    return [_cnot(c, t)]


def _exact_swap_decomposition():
    return [_cnot(0, 1), _cnot(1, 0), _cnot(0, 1)]


def _exact_iswap_decomposition():
    # iSWAP = [[0, 0, 0, 0],
    #           [0, 0, i, 0],
    #           [0, i, 0, 0],
    #           [0, 0, 0, 0]]
    # This can be done with 2 CNOTs
    gates = [
        _h(0),
        _h(1),
        _cnot(0, 1),
        _sdag(0),
        _s(1),
        _cnot(0, 1),
        _h(0),
        _h(1),
    ]
    return gates


def _exact_toffoli_decomposition(c1, c2, t):
    gates = [
        _h(t),
        _cnot(c1, t),
        _tdag(t),
        _t(c2),
        _cnot(c2, t),
        _t(t),
        _tdag(c2),
        _cnot(c1, t),
        _t(c2),
        _tdag(t),
        _h(t),
        _cnot(c1, c2),
        _t(c1),
        _tdag(c2),
        _cnot(c1, c2),
    ]
    return gates


def _match_known(target, qubits, cnot_perms):
    """Try to match the target against known gate decompositions."""
    if qubits == 2:
        candidates = [
            (_exact_cnot_decomposition(0, 1), 1),
            (_exact_cnot_decomposition(1, 0), 1),
            (_exact_swap_decomposition(), 3),
            (_exact_iswap_decomposition(), 2),
        ]
        for gates, cnot_count in candidates:
            U = _build_fast(gates, qubits, cnot_perms)
            if _distance(target, U) < 1e-10:
                return gates
    elif qubits == 3:
        for c1 in range(3):
            for c2 in range(3):
                if c2 == c1:
                    continue
                t = 3 - c1 - c2
                gates = _exact_toffoli_decomposition(c1, c2, t)
                U = _build_fast(gates, qubits, cnot_perms)
                if _distance(target, U) < 1e-10:
                    return gates
    return None


def _optimize_generic(target, qubits, budget, cnot_perms):
    """
    Optimize a circuit with a fixed CNOT structure using coordinate descent.
    Structure: [u3s] CNOT [u3s] CNOT ... [u3s]
    """
    n = 2 ** qubits

    # Choose CNOT positions
    if qubits == 2:
        # Alternate between (0,1) and (1,0)
        cnot_pairs = []
        for i in range(budget):
            if i % 2 == 0:
                cnot_pairs.append((0, 1))
            else:
                cnot_pairs.append((1, 0))
    else:
        # For 3 qubits, use a pattern that covers all pairs
        pattern = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
        cnot_pairs = [pattern[i % len(pattern)] for i in range(budget)]

    num_cnots = len(cnot_pairs)
    num_segments = num_cnots + 1  # segments of u3 gates

    # Parameters: for each segment, 3 params per qubit (theta, phi, lam)
    num_params = num_segments * qubits * 3

    # Initialize with zeros
    params = np.zeros(num_params)

    def build_and_eval(p):
        gates = []
        idx = 0
        for seg in range(num_segments):
            for q in range(qubits):
                theta = p[idx]
                phi = p[idx + 1]
                lam = p[idx + 2]
                gates.append(['u3', q, float(theta), float(phi), float(lam)])
                idx += 3
            if seg < num_cnots:
                c, t = cnot_pairs[seg]
                gates.append(_cnot(c, t))

        U = _build_fast(gates, qubits, cnot_perms)
        return _distance(target, U)

    best_dist = build_and_eval(params)
    best_params = params.copy()

    # Coordinate descent with adaptive step
    step = 1.0
    no_improve_count = 0
    max_sweeps = 300
    patience = 8

    for sweep in range(max_sweeps):
        if _check_deadline():
            break

        improved = False
        for i in range(num_params):
            if _check_deadline():
                break

            base_val = params[i]
            for sign in (1, -1):
                params[i] = base_val + sign * step
                dist = build_and_eval(params)
                if dist < best_dist:
                    best_dist = dist
                    best_params = params.copy()
                    improved = True
                else:
                    params[i] = base_val

        if improved:
            no_improve_count = 0
            step *= 0.7
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
            step *= 0.5

    # Build final gates
    gates = []
    idx = 0
    for seg in range(num_segments):
        for q in range(qubits):
            theta = best_params[idx]
            phi = best_params[idx + 1]
            lam = best_params[idx + 2]
            gates.append(['u3', q, float(theta), float(phi), float(lam)])
            idx += 3
        if seg < num_cnots:
            c, t = cnot_pairs[seg]
            gates.append(_cnot(c, t))

    return gates


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")

    # Precompute CNOT permutations
    cnot_perms = _precompute_cnot_perm(qubits)

    # Try known gate matching first
    known = _match_known(target, qubits, cnot_perms)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known

    # Generic synthesis with coordinate descent
    return _optimize_generic(target, qubits, budget, cnot_perms)
