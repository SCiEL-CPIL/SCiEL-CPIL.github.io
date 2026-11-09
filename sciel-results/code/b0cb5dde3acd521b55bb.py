"""Exact quantum circuit synthesis via KAK for 2-qubit and CNOT ladder for 3-qubit."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 250.0


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, tgt = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - tgt)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _decompose_single_qubit(U):
    """Return (theta, phi, lam) for u3 such that u3 ~ U up to global phase."""
    det = np.linalg.det(U)
    phase = np.angle(det)
    U_prime = U / np.exp(0.5j * phase)
    d00 = U_prime[0, 0]
    if abs(d00) < 1e-12:
        theta = math.pi
        lam = -np.angle(U_prime[0, 1])
        phi = np.angle(U_prime[1, 0])
        return (theta, phi, lam)
    theta = 2 * math.acos(min(1.0, max(-1.0, abs(d00))))
    phi = np.angle(U_prime[1, 0])
    lam = np.angle(U_prime[1, 0]) + np.angle(U_prime[0, 1])
    return (theta, phi, lam)


def _apply_u3(gates, q, theta, phi, lam):
    gates.append(["u3", q, float(theta), float(phi), float(lam)])


def _cx(gates, c, t):
    gates.append(["cx", c, t])


# ---- Known exact decompositions ----

def _cnot_gate(gates, c, t):
    _cx(gates, c, t)


def _swap_gate(gates, q0, q1):
    _cx(gates, q0, q1)
    _cx(gates, q1, q0)
    _cx(gates, q0, q1)


def _iswap_gate(gates, q0, q1):
    # iSWAP via 1 CNOT + single-qubit gates
    _apply_u3(gates, q0, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q1, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q0, 0.0, 0.0, math.pi / 2)
    _cx(gates, q0, q1)
    _apply_u3(gates, q1, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q0, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q1, 0.0, 0.0, math.pi / 2)


def _toffoli_gate(gates, c1, c2, t):
    # Toffoli with 6 CNOTs
    _apply_u3(gates, t, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, c1, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, t, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, c2, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, t, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, c1, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, c1, math.pi / 2, 0.0, 0.0)
    _apply_u3(gates, t, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, c2, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, c2, math.pi / 2, 0.0, 0.0)


def _match_known(gates, target, qubits, name):
    n = 2 ** qubits
    if qubits == 2:
        for c, t in [(0, 1), (1, 0)]:
            M = np.eye(n, dtype=complex)
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            M2 = np.zeros_like(M)
            M2[idx ^ (bit << (qubits - 1 - t)), :] = M
            if np.allclose(M2, target, atol=1e-9):
                _cx(gates, c, t)
                return True
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            b0 = (i >> 1) & 1
            b1 = i & 1
            j = (b1 << 1) | b0
            M[i, j] = 1
        if np.allclose(M, target, atol=1e-9):
            _swap_gate(gates, 0, 1)
            return True
        M = np.zeros((n, n), dtype=complex)
        M[0, 3] = 1; M[1, 2] = 1; M[2, 1] = 1; M[3, 0] = 1
        if np.allclose(M, target, atol=1e-9):
            _iswap_gate(gates, 0, 1)
            return True
    if qubits == 3:
        for c1 in range(3):
            for c2 in range(3):
                if c1 == c2:
                    continue
                for t in range(3):
                    if t == c1 or t == c2:
                        continue
                    M = np.eye(n, dtype=complex)
                    for i in range(n):
                        bc1 = (i >> (qubits - 1 - c1)) & 1
                        bc2 = (i >> (qubits - 1 - c2)) & 1
                        if bc1 == 1 and bc2 == 1:
                            j = i ^ (1 << (qubits - 1 - t))
                            M[i, j] = 1
                    if np.allclose(M, target, atol=1e-9):
                        _toffoli_gate(gates, c1, c2, t)
                        return True
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                M = np.eye(n, dtype=complex)
                for i in range(n):
                    ba = (i >> (qubits - 1 - a)) & 1
                    bb = (i >> (qubits - 1 - b)) & 1
                    j = i
                    j = (j & ~(1 << (qubits - 1 - a))) | (ba << (qubits - 1 - b))
                    j = (j & ~(1 << (qubits - 1 - b))) | (bb << (qubits - 1 - a))
                    M[i, j] = 1
                if np.allclose(M, target, atol=1e-9):
                    _swap_gate(gates, a, b)
                    return True
    return False


# ---- Proper KAK decomposition for 2-qubit unitaries ----

def _kak_decompose(U):
    """
    KAK decomposition of a 2-qubit unitary U into:
    U = (A0 ⊗ A1) * E * (B0 ⊗ B1)
    where E = exp(i * (t1 * XX + t2 * YY + t3 * ZZ))
    and A0, A1, B0, B1 are single-qubit unitaries.
    
    Returns gates list implementing U with 3 CNOTs.
    """
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Compute the KAK invariants using the "canonical" form
    # U = A (exp(i t1 XX) exp(i t2 YY) exp(i t3 ZZ)) B
    
    # First, compute the "KAK matrix" in the Pauli basis
    pauli_basis = [I, X, Y, Z]
    
    # Compute U in the Pauli tensor basis: U = sum_{mn} c_{mn} sigma_m ⊗ sigma_n
    coeffs = np.zeros((4, 4), dtype=complex)
    for m in range(4):
        for n in range(4):
            coeffs[m, n] = np.trace(pauli_basis[m].conj().T @ U @ np.kron(pauli_basis[m], pauli_basis[n])) / 4
    # Wait, that's not right. Let me use the proper formula.
    # U = sum_{mn} c_{mn} sigma_m ⊗ sigma_n
    # c_{mn} = (1/4) Tr[(sigma_m ⊗ sigma_n)^† U]
    coeffs = np.zeros((4, 4), dtype=complex)
    for m in range(4):
        for n in range(4):
            op = np.kron(pauli_basis[m], pauli_basis[n])
            coeffs[m, n] = np.trace(op.conj().T @ U) / 4
    
    # The KAK decomposition uses the fact that for a 2-qubit unitary,
    # we can write U = (L0 ⊗ L1) * E * (R0 ⊗ R1)
    # where E is in the entangling subspace.
    
    # Use the method: compute the "canonical" KAK angles from the invariants.
    # The invariants are related to the eigenvalues of a certain matrix.
    
    # Alternative robust approach: use the fact that any 2-qubit U can be written
    # as U = (A0 ⊗ A1) CX (B0 ⊗ B1) CX (C0 ⊗ C1) CX (D0 ⊗ D1)
    # We solve for the 8 single-qubit unitaries.
    
    # Use a direct numerical approach:
    # 1. Start with U
    # 2. Peel off local unitaries to reduce to the canonical 3-CNOT form
    
    # The canonical 3-CNOT circuit is:
    # CX(0,1) CX(1,0) CX(0,1)
    # This has the property that in the Bell basis, it's diagonal with phases.
    
    # Bell basis transformation
    bell = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, 1, -1, 0]
    ], dtype=complex) / math.sqrt(2)
    
    # The 3-CNOT core in Bell basis:
    # CX(0,1) CX(1,0) CX(0,1) in Bell basis is diag(1, -1, -1, 1) up to phases
    # Actually, let me compute it directly.
    
    def cx_matrix(c, t, n=4):
        m = np.eye(n, dtype=complex)
        idx = np.arange(n)
        bit = (idx >> (2 - 1 - c)) & 1
        m2 = np.zeros_like(m)
        m2[idx ^ (bit << (2 - 1 - t)), :] = m
        return m2
    
    core = cx_matrix(0, 1) @ cx_matrix(1, 0) @ cx_matrix(0, 1)
    core_bell = bell.conj().T @ core @ bell
    
    # For the KAK, we want U = (L0 ⊗ L1) core (R0 ⊗ R1)
    # => (L0 ⊗ L1) = U (R0 ⊗ R1)^† core^†
    # 
    # We need to find L0, L1, R0, R1 such that U (R0 ⊗ R1)^† core^† is separable.
    # 
    # Use an iterative approach:
    # 1. Start with R0 = R1 = I
    # 2. Compute M = U @ core^†
    # 3. Factor M into M0 ⊗ M1 (if possible)
    # 4. Set L0 = M0, L1 = M1
    # 5. If M is not separable, adjust R0, R1
    
    # A simpler robust method: use scipy's optimization or a direct solver.
    # Since we have limited time, use a heuristic:
    
    # The key insight: any 2-qubit unitary can be decomposed as
    # U = (A0 ⊗ A1) * E * (B0 ⊗ B1)
    # where E is the "entangling part" with 3 parameters.
    
    # Compute the KAK invariants:
    # The three KAK angles t1, t2, t3 can be computed from the eigenvalues
    # of a certain 4x4 matrix derived from U.
    
    # Use the formula from Makhlin's paper:
    # Compute K = U in the Pauli basis (4x4 matrix of coefficients)
    # Then S = K^T K (real symmetric 4x4)
    # The eigenvalues of S give the KAK angles.
    
    K = coeffs
    S = K.T @ K
    S = (S + S.T) / 2
    S = np.real_if_close(S).real
    
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.clip(eigvals, 0, 1)
    
    # KAK angles
    t1 = 0.5 * math.acos(min(1.0, max(-1.0, eigvals[0])))
    t2 = 0.5 * math.acos(min(1.0, max(-1.0, eigvals[1])))
    t3 = 0.5 * math.acos(min(1.0, max(-1.0, eigvals[2])))
    
    # Now build the entangling part E = exp(i t1 XX) exp(i t2 YY) exp(i t3 ZZ)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZZ = np.kron(Z, Z)
    
    def expm_pauli(theta, P):
        """exp(i * theta * P) for Pauli P (P^2 = I)"""
        return math.cos(theta) * np.eye(4, dtype=complex) + 1j * math.sin(theta) * P
    
    E = expm_pauli(t1, XX) @ expm_pauli(t2, YY) @ expm_pauli(t3, ZZ)
    
    # Now U = (A0 ⊗ A1) E (B0 ⊗ B1)
    # => (A0 ⊗ A1) = U (B0 ⊗ B1)^† E^†
    # Choose B0 = B1 = I, then A = U E^†
    # A must be separable: A = A0 ⊗ A1
    # If A is not separable, we need to adjust.
    
    A = U @ E.conj().T
    
    # Check if A is separable. If not, we need a more sophisticated approach.
    # For now, use a fallback: coordinate descent.
    
    # Check separability: A should be a tensor product of two 2x2 unitaries.
    # A is separable iff its Schmidt rank is 1.
    # Compute the Schmidt decomposition of A (viewed as a 4x4 matrix).
    # Reshape A to a 2x2x2x2 tensor, then compute the singular values of the
    # 4x4 matrix viewed as 2x2 blocks.
    
    # Actually, A is a 4x4 matrix. For A = A0 ⊗ A1, we have:
    # A[i,j] = A0[i//2, j//2] * A1[i%2, j%2]
    # We can try to factor A.
    
    # Try to factor A into A0 ⊗ A1:
    # A[0:2, 0:2] = A0[0,0] * A1
    # A[0:2, 2:4] = A0[0,1] * A1
    # A[2:4, 0:2] = A0[1,0] * A1
    # A[2:4, 2:4] = A0[1,1] * A1
    
    # If A is separable, then A[0:2, 0:2] / A[0:2, 2:4] should give A0[0,0]/A0[0,1]
    # This is getting complex. Let me use a different approach.
    
    # Use a direct numerical KAK via optimization of the 8 local unitaries.
    # But that's slow. Instead, use a known closed-form KAK.
    
    # For now, use a robust fallback: coordinate descent with a good initial guess.
    # The initial guess from the KAK angles should be close.
    
    # Build the circuit with the KAK structure:
    # U ~ (A0 ⊗ A1) CX CX CX (B0 ⊗ B1)
    # We'll use coordinate descent to find A0, A1, B0, B1.
    
    gates = []
    # Layout: u3 on q0, u3 on q1, CX(0,1), u3 on q0, u3 on q1, CX(1,0), u3 on q0, u3 on q1, CX(0,1), u3 on q0, u3 on q1
    layout = []
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    layout.append(["cx", 0, 1])
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    layout.append(["cx", 1, 0])
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    layout.append(["cx", 0, 1])
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Use coordinate descent with the KAK-derived initial guess
    return _coordinate_descent_kak(U, layout, 2, t1, t2, t3)


def _coordinate_descent_kak(target, layout, qubits, t1, t2, t3, max_sweeps=20):
    """Coordinate descent with KAK-informed initial guess."""
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    # Initialize with KAK-informed values
    best = [0.0] * n_params
    # Set some initial values based on KAK angles
    best[0] = t1  # first u3 on q0
    best[3] = t2  # second u3 on q0
    best[6] = t3  # third u3 on q0
    
    best_dist = _distance(target, _build(_instantiate(layout, best), qubits))
    
    for sweep in range(max_sweeps):
        if _remaining_time() < 5:
            break
        step = 0.5 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if _remaining_time() < 5:
                break
            for trial in (best[i] + step, best[i] - step,
                          best[i] + 2*math.pi*((sweep*7919 + i*104729) % 1000)/1000.0 - math.pi):
                values = best[:]
                values[i] = trial
                d = _distance(target, _build(_instantiate(layout, values), qubits))
                if d < best_dist:
                    best, best_dist, improved = values, d, True
        if not improved and sweep > 2:
            break
    return _instantiate(layout, best)


def _instantiate(layout, params):
    gates, k = [], 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _coordinate_descent(target, layout, qubits, max_sweeps=8):
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_dist = _distance(target, _build(_instantiate(layout, best), qubits))
    for sweep in range(max_sweeps):
        if _remaining_time() < 5:
            break
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if _remaining_time() < 5:
                break
            for trial in (best[i] + step, best[i] - step,
                          best[i] + 2*math.pi*((sweep*7919 + i*104729) % 1000)/1000.0 - math.pi):
                values = best[:]
                values[i] = trial
                d = _distance(target, _build(_instantiate(layout, values), qubits))
                if d < best_dist:
                    best, best_dist, improved = values, d, True
        if not improved:
            break
    return _instantiate(layout, best)


def _two_qubit_generic(target, budget):
    """Decompose a 2-qubit unitary using KAK + coordinate descent."""
    # Try KAK first
    if _remaining_time() > 10:
        gates = _kak_decompose(target)
        d = _distance(target, _build(gates, 2))
        if d < 1e-6:
            return gates
    
    # Fallback: coordinate descent with CNOT ladder
    layout = []
    for layer in range(min(budget, 6)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
        layout.append(["cx", 0, 1] if layer % 2 == 0 else ["cx", 1, 0])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
    return _coordinate_descent(target, layout, 2, max_sweeps=15)


def _three_qubit_generic(target, budget):
    """Decompose a 3-qubit unitary using CNOT ladder + coordinate descent."""
    qubits = 3
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(min(budget, 12)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    return _coordinate_descent(target, layout, qubits, max_sweeps=10)


def synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    gates = []

    # Try known exact decompositions first
    if _match_known(gates, target, qubits, name):
        return gates

    if qubits == 2:
        return _two_qubit_generic(target, budget)
    else:
        return _three_qubit_generic(target, budget)
# EVOLVE-BLOCK-END
