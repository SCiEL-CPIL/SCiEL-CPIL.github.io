"""Quantum circuit synthesis with exact known-gate circuits and fast optimization fallback."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            # Apply single-qubit gate on qubit q
            # qubit 0 is MSB, so for 2 qubits: q=0 is left factor, q=1 is right factor
            # We need to build the full operator
            eye = np.eye(2, dtype=complex)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            # qubit 0 is MSB
            idx = np.arange(n)
            ctrl_bit = (idx >> (qubits - 1 - control)) & 1
            tgt_shift = (qubits - 1 - target)
            # Only flip where control bit is 1
            mask = (ctrl_bit == 1)
            new_idx = idx.copy()
            new_idx[mask] = idx[mask] ^ (1 << tgt_shift)
            U = U[new_idx, :]
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim."""
    dim = target.shape[0]
    val = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(val) / dim)


def _check_close(A, B, tol=1e-6):
    """Check if two unitaries are equal up to global phase."""
    dim = A.shape[0]
    # Find a non-zero element to fix phase
    max_idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[max_idx]) < 1e-12:
        return False
    phase = A[max_idx] / B[max_idx]
    phase = phase / abs(phase)
    return np.allclose(A, phase * B, atol=tol)


def _known_circuit_cnot_2q():
    """CNOT(0,1) for 2 qubits: just cx(0,1)."""
    return [["cx", 0, 1]]


def _known_circuit_swap_2q():
    """SWAP for 2 qubits: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _known_circuit_iswap_2q():
    """iSWAP for 2 qubits: 2 CNOTs + single-qubit rotations.
    iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    """
    pi = math.pi
    # Standard 2-CNOT iSWAP decomposition
    # U0 = diag(1, i) on q0, diag(1, i) on q1  ->  S gates
    # Then CNOT(0,1), then single qubit rotations, then CNOT(1,0)
    return [
        ["u3", 0, 0, 0, pi / 2],       # S on q0
        ["u3", 1, 0, 0, pi / 2],       # S on q1
        ["cx", 0, 1],
        ["u3", 0, pi / 2, 0, 0],       # H on q0
        ["u3", 1, pi / 2, 0, 0],       # H on q1
        ["cx", 1, 0],
        ["u3", 0, pi / 2, 0, 0],       # H on q0
        ["u3", 1, pi / 2, 0, 0],       # H on q1
    ]


def _known_circuit_toffoli_3q():
    """Toffoli (CCX) for 3 qubits: 6 CNOTs standard circuit.
    q0=control1, q1=control2, q2=target
    """
    pi = math.pi
    # Standard 6-CNOT Toffoli decomposition
    return [
        ["u3", 2, pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["u3", 2, pi / 2, 0, 0],
    ]


def _known_circuit_fredkin_3q():
    """Fredkin (controlled-SWAP) for 3 qubits: 6 CNOTs.
    q0=control, q1=swap_a, q2=swap_b
    """
    pi = math.pi
    return [
        ["cx", 1, 2],
        ["cx", 0, 2],
        ["cx", 1, 2],
        ["cx", 0, 1],
        ["cx", 1, 2],
        ["cx", 0, 2],
        ["cx", 1, 2],
    ]


def _known_circuit_ccz_3q():
    """CCZ for 3 qubits: 6 CNOTs."""
    pi = math.pi
    # CCZ = diag(1,1,1,1,1,1,1,-1)
    # Use H on target to convert CCX to CCZ
    return [
        ["u3", 2, 0, 0, pi],  # S gate
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -pi / 2],  # S-dagger
        ["cx", 0, 2],
        ["u3", 2, 0, 0, pi / 2],  # S
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -pi / 2],  # S-dagger
        ["cx", 0, 2],
    ]


def _detect_known_gate(target, qubits, budget):
    """Try to identify the target as a known gate and return exact circuit if so."""
    
    if qubits == 2:
        if budget >= 1:
            # CNOT(0,1)
            ref = _build_fast([["cx", 0, 1]], 2)
            if _check_close(target, ref):
                return _known_circuit_cnot_2q()
            # CNOT(1,0)
            ref = _build_fast([["cx", 1, 0]], 2)
            if _check_close(target, ref):
                return [["cx", 1, 0]]
        if budget >= 3:
            # SWAP
            circ = _known_circuit_swap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
        if budget >= 2:
            # iSWAP
            circ = _known_circuit_iswap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
    elif qubits == 3:
        if budget >= 6:
            # Toffoli (CCX) with q0,q1 as controls, q2 as target
            circ = _known_circuit_toffoli_3q()
            ref = _build_fast(circ, 3)
            if _check_close(target, ref):
                return circ
            # Other control/target combinations for Toffoli
            # We'll check all 6 permutations of (c1,c2,t)
            for c1, c2, t in [(0,1,2), (1,0,2), (0,2,1), (2,0,1), (1,2,0), (2,1,0)]:
                if c1 == 0 and c2 == 1 and t == 2:
                    continue  # already checked
                circ_alt = _permute_toffoli((c1, c2, t))
                if circ_alt:
                    ref = _build_fast(circ_alt, 3)
                    if _check_close(target, ref):
                        return circ_alt
            
            # Fredkin variants (3 CNOTs)
            if budget >= 3:
                for ctrl, a, b in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                    circ = _fredkin_circuit(ctrl, a, b)
                    ref = _build_fast(circ, 3)
                    if _check_close(target, ref):
                        return circ
            
            # CCZ variants (6 CNOTs)
            if budget >= 6:
                for c1, c2, t in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                    circ = _ccz_circuit(c1, c2, t)
                    ref = _build_fast(circ, 3)
                    if _check_close(target, ref):
                        return circ
    return None


def _permute_toffoli(controls_target):
    """Generate Toffoli circuit for arbitrary control/target qubits.
    controls_target = (c1, c2, tgt) where c1,c2 are controls and tgt is target.
    """
    c1, c2, tgt = controls_target
    pi = math.pi
    # Standard Toffoli: controls 0,1 target 2
    # We can use SWAP gates to permute qubits, apply standard Toffoli, then unpermute.
    # But that adds CNOTs. Instead, let's directly construct for each case.
    # For now, handle the most common cases directly.
    
    # If target is q2 and controls are {0,1}: standard
    if tgt == 2 and c1 == 0 and c2 == 1:
        return _known_circuit_toffoli_3q()
    if tgt == 2 and c1 == 1 and c2 == 0:
        return _known_circuit_toffoli_3q()  # symmetric in controls
    
    # If target is q1:
    if tgt == 1:
        # SWAP q1<->q2, apply Toffoli(0,2,1->now at position 2), SWAP back
        # This uses 3+6+3=12 CNOTs, too many.
        # Alternative: use the fact that CCX(c1,c2,t) can be built with 6 CNOTs for any assignment
        # Let me construct it directly
        # For CCX(c1,c2,t) where t != 2:
        # We need a 6-CNOT circuit. The standard one assumes t=2.
        # Let's just return None for non-standard and let the optimizer handle it.
        return None
    
    # If target is q0:
    if tgt == 0:
        return None
    
    return None


def _fredkin_circuit(ctrl, a, b):
    """Fredkin (controlled-SWAP) circuit for given control and swap qubits.
    Uses 3 CNOTs: cx(a,b), cx(ctrl,b), cx(a,b)
    """
    return [
        ["cx", a, b],
        ["cx", ctrl, b],
        ["cx", a, b],
    ]


def _ccz_circuit(c1, c2, t):
    """CCZ circuit for given controls and target."""
    pi = math.pi
    return [
        ["u3", t, 0, 0, pi],
        ["cx", c2, t],
        ["u3", t, 0, 0, -pi / 2],
        ["cx", c1, t],
        ["u3", t, 0, 0, pi / 2],
        ["cx", c2, t],
        ["u3", t, 0, 0, -pi / 2],
        ["cx", c1, t],
    ]


def _householder_decompose(target, qubits, budget):
    """
    Recursive Householder-like exact decomposition.
    Zeroes entries of target using single-qubit rotations + CNOTs.
    Returns list of gates or None if budget exceeded.
    """
    n = 2 ** qubits
    U = target.copy()
    gates = []
    cnot_count = 0
    
    def count_cnots(g):
        return 1 if g[0] == "cx" else 0
    
    # Base case: 1 qubit - extract u3 parameters directly
    if qubits == 1:
        # U is 2x2 unitary. Extract u3(theta, phi, lam)
        # u3 matrix: [[c, -e^{i*lam}*s], [e^{i*phi}*s, e^{i*(phi+lam)}*c]]
        # where c = cos(theta/2), s = sin(theta/2)
        c00, c01 = U[0, 0], U[0, 1]
        c10, c11 = U[1, 0], U[1, 1]
        
        # c = |c00|, s = |c10| (for standard u3)
        c = abs(c00)
        s = abs(c10)
        
        if s < 1e-10:
            # theta = 0, U is diagonal: diag(e^{i*phi/2}, e^{-i*phi/2}) approximately
            # Actually if s=0: [[1, 0], [0, e^{i*lam}]]
            if abs(c00) < 1e-10:
                # degenerate
                return None
            lam = np.angle(c11 / c00)
            phi = 0.0
            theta = 0.0
            return [["u3", 0, theta, phi, lam]]
        elif c < 1e-10:
            # theta = pi
            theta = math.pi
            phi = np.angle(c10 / c01) if abs(c01) > 1e-10 else 0.0
            lam = np.angle(c11 / c10) if abs(c10) > 1e-10 else 0.0
            return [["u3", 0, theta, phi, lam]]
        else:
            theta = 2 * math.atan2(s, c)
            # c00 = c, c01 = -e^{i*lam}*s => e^{i*lam} = -c01/s
            lam = np.angle(-c01 / s)
            # c10 = e^{i*phi}*s => e^{i*phi} = c10/s
            phi = np.angle(c10 / s)
            return [["u3", 0, theta, phi, lam]]
    
    # For 2+ qubits, we need to zero out entries.
    # Strategy: Use CNOTs to bring off-diagonal elements into position where
    # single-qubit rotations can zero them out.
    
    # For 2 qubits: target is 4x4.
    # We want to transform U into block-diagonal form using CNOTs and single-qubit gates.
    
    # Approach for 2 qubits with budget CNOTs:
    # Use the fact that any 2-qubit unitary can be decomposed with at most 3 CNOTs.
    # But we have a budget constraint.
    
    # Let's use a different approach: iterative entry zeroing.
    # For 2 qubits, the basis is |00>, |01>, |10>, |11>
    # qubit 0 is MSB: |q0 q1>
    
    # We'll use a greedy approach: find the largest off-diagonal element in the
    # "upper" part and zero it using a rotation on the appropriate qubit.
    
    # Actually, let's use a more systematic approach based on the structure.
    
    # For 2 qubits, we can decompose as:
    # U = (A ⊗ I) * CX * (B ⊗ I) * CX * (C ⊗ I)  [with 2 CNOTs]
    # or similar patterns.
    
    # Let me implement a specific decomposition for 2 qubits.
    
    if qubits == 2:
        return _decompose_2q(U, budget)
    elif qubits == 3:
        return _decompose_3q(U, budget)
    
    return None


def _decompose_2q(U, budget):
    """Decompose 2-qubit unitary into u3 and cx gates."""
    if budget < 1:
        # Only single-qubit gates possible
        # Check if U is already product of single-qubit gates
        return _try_single_qubit_only(U, 2)
    
    # Standard 2-qubit decomposition using CNOTs
    # Any 2-qubit U can be written as:
    # U = (A0 ⊗ A1) * CX * (B0 ⊗ B1) * CX * (C0 ⊗ C1)  [2 CNOTs]
    # or with 3 CNOTs for general case.
    
    # Let's use the KAK-like approach or direct matrix matching.
    
    # For simplicity, let's try a pattern-based approach:
    # Pattern 1: 1 CNOT
    # U = (A ⊗ I) * CX * (B ⊗ I)
    # This gives: U = A * CX * B where A, B act on qubit 0
    # CX = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    
    # Let me implement a direct solver for small budgets.
    
    # For budget >= 3, use standard 3-CNOT decomposition
    if budget >= 3:
        result = _decompose_2q_three_cnot(U)
        if result is not None:
            return result
    
    # For budget >= 2, try 2-CNOT decomposition
    if budget >= 2:
        result = _decompose_2q_two_cnot(U)
        if result is not None:
            return result
    
    # For budget >= 1, try 1-CNOT decomposition
    if budget >= 1:
        result = _decompose_2q_one_cnot(U)
        if result is not None:
            return result
    
    return None


def _extract_u3_from_2x2(M):
    """Extract u3 parameters from a 2x2 unitary matrix."""
    c00, c01 = M[0, 0], M[0, 1]
    c10, c11 = M[1, 0], M[1, 1]
    
    c = abs(c00)
    s = abs(c10)
    
    if s < 1e-10:
        if abs(c00) < 1e-10:
            return None
        lam = np.angle(c11 / c00)
        return (0.0, 0.0, lam)
    elif c < 1e-10:
        theta = math.pi
        phi = np.angle(c10 / c01) if abs(c01) > 1e-10 else 0.0
        lam = np.angle(c11 / c10) if abs(c10) > 1e-10 else 0.0
        return (theta, phi, lam)
    else:
        theta = 2 * math.atan2(s, c)
        lam = np.angle(-c01 / s)
        phi = np.angle(c10 / s)
        return (theta, phi, lam)


def _decompose_2q_one_cnot(U):
    """Try to decompose as (A ⊗ I) * CX * (B ⊗ I)"""
    # CX in computational basis (q0 MSB):
    # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    
    # U = (A ⊗ I) * CX * (B ⊗ I)
    # Let V = U * CX^dag = U * CX (since CX is Hermitian)
    # Then V = (A ⊗ I) * (B ⊗ I) = (A*B) ⊗ I
    # So V must be of the form M ⊗ I
    
    V = U @ CX.conj().T
    
    # Check if V = M ⊗ I for some 2x2 M
    # V[0,0] = M[0,0], V[0,1] = 0, V[0,2] = 0, V[0,3] = M[0,1]
    # V[1,0] = 0, V[1,1] = M[0,0], V[1,2] = M[0,1], V[1,3] = 0
    # V[2,0] = M[1,0], V[2,1] = 0, V[2,2] = M[1,1], V[2,3] = 0
    # V[3,0] = 0, V[3,1] = M[1,0], V[3,2] = 0, V[3,3] = M[1,1]
    
    M = np.array([[V[0,0], V[0,3]],
                   [V[2,0], V[2,3]]], dtype=complex)
    
    # Verify
    expected = np.kron(M, np.eye(2))
    if not np.allclose(V, expected, atol=1e-8):
        return None
    
    # Now M = A * B, need to factor into A and B
    # Use polar decomposition or just set B = I, A = M
    # But we need A and B to be unitary (they should be from u3)
    
    # Simplest: set B = I, A = M
    # But M might not be directly a u3 gate... it should be unitary
    # Let's extract u3 for A and B=I
    
    params_A = _extract_u3_from_2x2(M)
    if params_A is None:
        return None
    
    theta_a, phi_a, lam_a = params_A
    gates = [
        ["u3", 0, theta_a, phi_a, lam_a],
        ["cx", 0, 1],
    ]
    return gates


def _decompose_2q_two_cnot(U):
    """Try to decompose using 2 CNOTs.
    Pattern: (A0⊗A1) * CX(0,1) * (B0⊗B1) * CX(1,0) * (C0⊗C1)
    """
    CX_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    CX_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    
    # This is more complex. Let's use a different approach.
    # Actually, let's just fall through to the 3-CNOT version if 2-CNOT doesn't work easily.
    return None


def _decompose_2q_three_cnot(U):
    """Decompose 2-qubit unitary using 3 CNOTs (standard)."""
    # Standard decomposition: U = (A⊗B) * CX * (C⊗D) * CX * (E⊗F) * CX * (G⊗H)
    # This requires solving for 8 single-qubit unitaries.
    
    # Simpler approach: use the fact that any 2-qubit unitary can be written as:
    # U = e^{iα} (A0 ⊗ A1) * exp(iβ CX) * (B0 ⊗ B1)
    # but this is KAK form.
    
    # Let me use a direct matrix approach.
    # We'll use the pattern: U = (A⊗I) * CX * (B⊗I) * CX * (C⊗I)
    # This has 3 CNOTs but only acts on qubit 0 with single-qubit gates.
    # Not general enough.
    
    # Better: use full single-qubit gates on both qubits.
    # U = (A0⊗A1) * CX(0,1) * (B0⊗B1) * CX(0,1) * (C0⊗C1)
    
    # Let me implement a numerical solver.
    # For now, return None and let the caller handle fallback.
    return None


def _decompose_3q(U, budget):
    """Decompose 3-qubit unitary into u3 and cx gates."""
    if budget < 6:
        # Can't do full 3-qubit decomposition with fewer than 6 CNOTs
        # Try partial decomposition or return None
        return None
    
    # Standard 3-qubit decomposition uses 6 CNOTs minimum for general unitaries.
    # Let's use a recursive approach.
    
    # For 3 qubits, we can decompose as:
    # U = (single-qubit gates) * CX * (single-qubit gates) * CX * ... 
    # with 6 CNOTs in a specific pattern.
    
    # Pattern for 3-qubit: use CNOTs (0,1), (0,2), (1,2) in a specific order.
    # This is complex to solve directly.
    
    # Alternative: use the known Toffoli-like structure.
    # For now, return None to use fallback.
    return None


def _try_single_qubit_only(U, qubits):
    """Check if U can be decomposed into single-qubit gates only."""
    # For 2 qubits: U = A ⊗ B
    # For 3 qubits: U = A ⊗ B ⊗ C
    
    if qubits == 2:
        # Check if U = A ⊗ B
        # U[0,0] = A[0,0]*B[0,0], U[0,1] = A[0,0]*B[0,1], etc.
        A00 = U[0,0]
        if abs(A00) < 1e-10:
            return None
        A01 = U[0,3] / A00 if abs(A00) > 1e-10 else 0
        # This is getting complex. Let me just check if it's a product state.
        
        # For U = A ⊗ B:
        # U[i,j] = A[i//2, j//2] * B[i%2, j%2]
        A = np.array([[U[0,0], U[0,2]],
                       [U[1,0], U[1,2]]], dtype=complex)
        B = np.array([[U[0,0], U[0,1]],
                       [U[2,0], U[2,1]]], dtype=complex)
        
        expected = np.kron(A, B)
        if not np.allclose(U, expected, atol=1e-8):
            return None
        
        pa = _extract_u3_from_2x2(A)
        pb = _extract_u3_from_2x2(B)
        if pa is None or pb is None:
            return None
        
        return [
            ["u3", 0, pa[0], pa[1], pa[2]],
            ["u3", 1, pb[0], pb[1], pb[2]],
        ]
    
    elif qubits == 3:
        # Similar check for 3 qubits
        # This is more complex, skip for now
        return None
    
    return None


def _extract_u3_from_2x2(M):
    """Extract u3(theta, phi, lam) parameters from a 2x2 unitary matrix."""
    c00, c01 = M[0, 0], M[0, 1]
    c10, c11 = M[1, 0], M[1, 1]
    
    c = abs(c00)
    s = abs(c10)
    
    if s < 1e-10:
        if abs(c00) < 1e-10:
            return None
        lam = float(np.angle(c11 / c00))
        return (0.0, 0.0, lam)
    elif c < 1e-10:
        theta = math.pi
        phi = float(np.angle(c10 / c01)) if abs(c01) > 1e-10 else 0.0
        lam = float(np.angle(c11 / c10)) if abs(c10) > 1e-10 else 0.0
        return (theta, phi, lam)
    else:
        theta = float(2 * math.atan2(s, c))
        lam = float(np.angle(-c01 / s))
        phi = float(np.angle(c10 / s))
        return (theta, phi, lam)


def _try_single_qubit_only(U, qubits):
    """Check if U can be decomposed into single-qubit gates only (no CNOTs)."""
    if qubits == 2:
        # Check if U = A ⊗ B
        A = np.array([[U[0,0], U[0,2]],
                       [U[1,0], U[1,2]]], dtype=complex)
        B = np.array([[U[0,0], U[0,1]],
                       [U[2,0], U[2,1]]], dtype=complex)
        
        expected = np.kron(A, B)
        if not np.allclose(U, expected, atol=1e-8):
            return None
        
        pa = _extract_u3_from_2x2(A)
        pb = _extract_u3_from_2x2(B)
        if pa is None or pb is None:
            return None
        
        return [
            ["u3", 0, pa[0], pa[1], pa[2]],
            ["u3", 1, pb[0], pb[1], pb[2]],
        ]
    elif qubits == 3:
        # Check if U = A ⊗ B ⊗ C
        # U is 8x8. For U = A⊗B⊗C:
        # U[i,j] = A[i//4, j//4] * B[(i//2)%2, (j//2)%2] * C[i%2, j%2]
        A = np.array([[U[0,0], U[0,4]],
                       [U[1,0], U[1,4]]], dtype=complex)
        B = np.array([[U[0,0], U[0,2]],
                       [U[1,0], U[1,2]]], dtype=complex)
        C = np.array([[U[0,0], U[0,1]],
                       [U[2,0], U[2,1]]], dtype=complex)
        
        expected = np.kron(np.kron(A, B), C)
        if not np.allclose(U, expected, atol=1e-8):
            return None
        
        pa = _extract_u3_from_2x2(A)
        pb = _extract_u3_from_2x2(B)
        pc = _extract_u3_from_2x2(C)
        if pa is None or pb is None or pc is None:
            return None
        
        return [
            ["u3", 0, pa[0], pa[1], pa[2]],
            ["u3", 1, pb[0], pb[1], pb[2]],
            ["u3", 2, pc[0], pc[1], pc[2]],
        ]
    return None


def _decompose_2q_one_cnot(U):
    """Try to decompose 2-qubit U as (A ⊗ I) * CX(0,1) * (B ⊗ I)."""
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    
    # U = (A ⊗ I) * CX * (B ⊗ I)
    # V = U * CX† = (A ⊗ I) * (B ⊗ I) = (A*B) ⊗ I
    V = U @ CX.conj().T
    
    # Extract M from V = M ⊗ I
    M = np.array([[V[0,0], V[0,3]],
                   [V[2,0], V[2,3]]], dtype=complex)
    
    expected = np.kron(M, np.eye(2))
    if not np.allclose(V, expected, atol=1e-8):
        return None
    
    # M = A * B. Set B = I, A = M
    params_A = _extract_u3_from_2x2(M)
    if params_A is None:
        return None
    
    theta_a, phi_a, lam_a = params_A
    gates = [
        ["u3", 0, theta_a, phi_a, lam_a],
        ["cx", 0, 1],
    ]
    
    # Verify
    U_recon = _build_fast(gates, 2)
    if not _check_close(U, U_recon, tol=1e-6):
        return None
    
    return gates


def _decompose_2q_two_cnot(U):
    """Try 2-CNOT decomposition: (A0⊗A1)*CX(0,1)*(B0⊗B1)*CX(1,0)*(C0⊗C1)"""
    # This is complex to solve analytically. Skip for now.
    return None


def _decompose_2q_three_cnot(U):
    """
    Decompose general 2-qubit unitary using 3 CNOTs.
    Uses the standard decomposition:
    U = (A0⊗A1) * CX(0,1) * (B0⊗B1) * CX(0,1) * (C0⊗C1)
    """
    # We need to solve for A0, A1, B0, B1, C0, C1 (each 2x2 unitary)
    # such that U = (A0⊗A1) * CX * (B0⊗B1) * CX * (C0⊗C1)
    
    # Let's use a numerical approach: fix some parameters and solve for others.
    # Actually, let's use a known decomposition method.
    
    # Standard approach: 
    # 1. U = (L0 ⊗ L1) * CX * (M0 ⊗ M1) * CX * (R0 ⊗ R1)
    #    where L, M, R are single-qubit unitaries
    
    # This requires 6 single-qubit unitaries to match a 4x4 unitary (16 complex params = 32 real, 
    # minus 1 global phase = 31 real params). 6 SU(2) params = 18 real, plus 3 CNOT structure.
    # Actually 2-qubit unitary has 15 real params (SU(4) dim), and 6 SU(2) have 18 params, 
    # so there's redundancy.
    
    # Let me use a simpler approach: iterate and optimize.
    # For now, return None and use the optimizer as fallback.
    return None


def _householder_decompose(target, qubits, budget):
    """
    Recursive Householder-like exact decomposition.
    Returns list of gates or None if not solvable within budget.
    """
    # First try single-qubit-only
    result = _try_single_qubit_only(target, qubits)
    if result is not None:
        return result
    
    if qubits == 2:
        if budget >= 1:
            result = _decompose_2q_one_cnot(target)
            if result is not None:
                return result
        # Fall through to optimizer
        return None
    elif qubits == 3:
        # For 3 qubits, try known patterns first (handled in _detect_known_gate)
        # Then fall through to optimizer
        return None
    
    return None


def _optimize(gates_template, target, qubits, budget, max_sweeps=12):
    """Optimize parameters using coordinate descent with adaptive step size."""
    n_params = 3 * sum(1 for g in gates_template if g[0] == "u3")
    
    def circuit(params):
        gates, k = [], 0
        for gate in gates_template:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_fast(circuit(best), qubits))
    
    for sweep in range(max_sweeps):
        if _time_left() < 15:
            break
        step = 0.5 / (1 + sweep)
        for i in range(n_params):
            if _time_left() < 5:
                break
            improved = False
            for trial in (best[i] + step, best[i] - step):
                params = best[:]
                params[i] = trial
                dist = _distance(target, _build_fast(circuit(params), qubits))
                if dist < best_dist:
                    best, best_dist = params, dist
                    improved = True
            # If no improvement, try smaller step
            if not improved and sweep > 2:
                for trial in (best[i] + step * 0.1, best[i] - step * 0.1):
                    params = best[:]
                    params[i] = trial
                    dist = _distance(target, _build_fast(circuit(params), qubits))
                    if dist < best_dist:
                        best, best_dist = params, dist
    
    # Final polish pass with very small steps
    if _time_left() > 20:
        for _ in range(4):
            if _time_left() < 10:
                break
            for i in range(n_params):
                if _time_left() < 5:
                    break
                for trial in (best[i] + 0.01, best[i] - 0.01, best[i] + 0.001, best[i] - 0.001):
                    params = best[:]
                    params[i] = trial
                    dist = _distance(target, _build_fast(circuit(params), qubits))
                    if dist < best_dist:
                        best, best_dist = params, dist
    
    return circuit(best)


def synthesize(problem):
    global problem_name
    problem_name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate detection first
    known = _detect_known_gate(target, qubits, budget)
    if known is not None:
        # Verify it's within budget
        n_cnot = sum(1 for g in known if g[0] == "cx")
        if n_cnot <= budget:
            return known
    
    # Try Householder-like exact decomposition
    householder_result = _householder_decompose(target, qubits, budget)
    if householder_result is not None:
        n_cnot = sum(1 for g in householder_result if g[0] == "cx")
        if n_cnot <= budget and len(householder_result) <= 400:
            return householder_result
    
    # Fallback: generic optimization
    # Build a template with up to budget CNOTs and single-qubit rotations
    # Use a structured pattern: alternate between different CNOT pairs
    if qubits == 2:
        cnot_pairs = [(0, 1), (1, 0)]
    else:
        cnot_pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    
    template = []
    for layer in range(budget):
        template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        template.append(["cx", *cnot_pairs[layer % len(cnot_pairs)]])
    template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    result = _optimize(template, target, qubits, budget)
    return result
# EVOLVE-BLOCK-END
