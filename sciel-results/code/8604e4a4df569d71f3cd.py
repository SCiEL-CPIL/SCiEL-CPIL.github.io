"""Exact decompositions for known gates + KAK/optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 225.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", c, t]


def _u3_matrix(theta, phi, lam):
    """2x2 u3 rotation matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _exact_cnot():
    """CNOT gate: just one CNOT."""
    return [_cx(0, 1)]


def _exact_swap():
    """SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_iswap():
    """iSWAP decomposition using 2 CNOTs."""
    H = (math.pi / 2, 0.0, math.pi / 2)
    S = (0.0, 0.0, math.pi / 2)
    
    gates = []
    gates.append(_u3(0, *H))
    gates.append(_u3(1, *H))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, *S))
    gates.append(_u3(1, *S))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, *H))
    gates.append(_u3(1, *H))
    return gates


def _exact_toffoli():
    """Toffoli (CCNOT) on 3 qubits: 6 CNOTs decomposition."""
    H = (math.pi / 2, 0.0, math.pi / 2)
    T = (0.0, 0.0, math.pi / 4)
    Tdag = (0.0, 0.0, -math.pi / 4)
    
    gates = []
    gates.append(_u3(2, *H))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, *T))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, *Tdag))
    gates.append(_u3(1, *T))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, *Tdag))
    gates.append(_u3(2, *T))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, *Tdag))
    gates.append(_u3(2, *H))
    gates.append(_cx(0, 1))
    return gates


def _kak_2qubit(U):
    """KAK decomposition of a 2-qubit unitary.
    Returns (A1, A2, B1, B2, C1, C2, D1, D2, theta1, theta2, theta3)
    such that U = (A1⊗A2) CNOT (B1⊗B2) CNOT (C1⊗C2) CNOT (D1⊗D2)
    up to a global phase."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    sx_sx = np.kron(sx, sx)
    sy_sy = np.kron(sy, sy)
    sz_sz = np.kron(sz, sz)
    
    # Remove global phase
    det_U = np.linalg.det(U)
    phase = det_U ** (1.0 / 4.0)
    U_prime = U / phase
    
    # Compute KAK angles
    cos2t1 = np.real(np.trace(U_prime @ sx_sx)) / 4.0
    cos2t2 = np.real(np.trace(U_prime @ sy_sy)) / 4.0
    cos2t3 = np.real(np.trace(U_prime @ sz_sz)) / 4.0
    
    cos2t1 = max(-1.0, min(1.0, cos2t1))
    cos2t2 = max(-1.0, min(1.0, cos2t2))
    cos2t3 = max(-1.0, min(1.0, cos2t3))
    
    t1 = 0.5 * math.acos(cos2t1)
    t2 = 0.5 * math.acos(cos2t2)
    t3 = 0.5 * math.acos(cos2t3)
    
    # Build the entangling part M
    def _expm_pauli2(pauli_tensor, angle):
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        return c * np.eye(4, dtype=complex) + 1j * s * pauli_tensor
    
    M = _expm_pauli2(sx_sx, t1) @ _expm_pauli2(sy_sy, t2) @ _expm_pauli2(sz_sz, t3)
    
    # K1 = U_prime @ M†
    K1 = U_prime @ M.conj().T
    
    # Now K1 should be a tensor product A1⊗A2 (up to phase)
    # Extract A1 and A2 from K1
    # K1 = A1⊗A2 means:
    # K1[i*2+j, k*2+l] = A1[i,k] * A2[j,l]
    
    # Use the fact that for a tensor product:
    # K1[0:2, 0:2] = A1[0,0] * A2
    # K1[2:4, 0:2] = A1[1,0] * A2
    # K1[0:2, 2:4] = A1[0,1] * A2
    # K1[2:4, 2:4] = A1[1,1] * A2
    
    # If A1[0,0] != 0:
    # A2 = K1[0:2, 0:2] / A1[0,0]
    # A1[0,1] = K1[0,2] / A2[0,0] (if A2[0,0] != 0)
    # etc.
    
    def _extract_tensor(K):
        """Extract A and B from K = A⊗B."""
        # Try using K[0,0] as reference
        if abs(K[0, 0]) > 1e-10:
            # A[0,0] * B[0,0] = K[0,0]
            # Let's set A[0,0] = K[0,0] / sqrt(|K[0,0]|^2) ... 
            # Actually, let's just use the structure directly.
            # K[0:2, 0:2] = A[0,0] * B
            # K[2:4, 0:2] = A[1,0] * B
            # K[0:2, 2:4] = A[0,1] * B
            # K[2:4, 2:4] = A[1,1] * B
            
            # From K[0:2, 0:2] and K[0:2, 2:4]:
            # K[0,0]/K[0,2] = B[0,0]/B[0,1] (if A[0,0] != 0)
            # K[1,0]/K[1,2] = B[1,0]/B[1,1] (if A[0,0] != 0)
            
            # Let me use a different approach: 
            # The ratio K[0,0]/K[1,0] = A[0,0]/A[1,0] (if B[0,0] != 0)
            # And K[0,1]/K[1,1] = A[0,0]/A[1,0] (if B[0,1] != 0)
            
            # Let's just use SVD-like approach or direct extraction
            # Actually for a tensor product A⊗B:
            # vec(K) = vec(A) ⊗ vec(B) (in some ordering)
            
            # Simpler: if we know one element of A, we can get B
            # Let's assume A[0,0] is the phase factor
            # A[0,0] * B = K[0:2, 0:2]
            # Let A[0,0] = 1 (absorb into B)
            # Then B = K[0:2, 0:2]
            # A[1,0] = K[2,0] / B[0,0] (if B[0,0] != 0)
            # A[0,1] = K[0,2] / B[0,0] (if B[0,0] != 0)
            # A[1,1] = K[2,2] / B[0,0] (if B[0,0] != 0)
            
            B = K[0:2, 0:2].copy()
            if abs(B[0, 0]) > 1e-10:
                A = np.zeros((2, 2), dtype=complex)
                A[0, 0] = 1.0
                A[1, 0] = K[2, 0] / B[0, 0]
                A[0, 1] = K[0, 2] / B[0, 0]
                A[1, 1] = K[2, 2] / B[0, 0]
                # Normalize
                A = A / np.linalg.det(A) ** 0.5
                B = B / np.linalg.det(B) ** 0.5
                return A, B
            elif abs(B[0, 1]) > 1e-10:
                A = np.zeros((2, 2), dtype=complex)
                A[0, 0] = 1.0
                A[1, 0] = K[2, 1] / B[0, 1]
                A[0, 1] = K[0, 3] / B[0, 1]
                A[1, 1] = K[2, 3] / B[0, 1]
                A = A / np.linalg.det(A) ** 0.5
                B = B / np.linalg.det(B) ** 0.5
                return A, B
            elif abs(B[1, 0]) > 1e-10:
                # Use different reference
                A = np.zeros((2, 2), dtype=complex)
                A[0, 0] = 1.0
                A[1, 0] = K[3, 0] / B[1, 0]
                A[0, 1] = K[1, 2] / B[1, 0]
                A[1, 1] = K[3, 2] / B[1, 0]
                A = A / np.linalg.det(A) ** 0.5
                B = B / np.linalg.det(B) ** 0.5
                return A, B
            else:
                # All elements of first row of B are zero, try different approach
                A = np.zeros((2, 2), dtype=complex)
                A[0, 0] = 1.0
                A[1, 0] = K[3, 1] / B[1, 1] if abs(B[1,1]) > 1e-10 else 0
                A[0, 1] = K[1, 3] / B[1, 1] if abs(B[1,1]) > 1e-10 else 0
                A[1, 1] = K[3, 3] / B[1, 1] if abs(B[1,1]) > 1e-10 else 0
                A = A / np.linalg.det(A) ** 0.5
                B = B / np.linalg.det(B) ** 0.5
                return A, B
        else:
            # K[0,0] is zero, try other elements
            if abs(K[0, 1]) > 1e-10:
                B = K[0:2, 1:3].copy()
                A = np.zeros((2, 2), dtype=complex)
                A[0, 1] = 1.0
                A[1, 1] = K[2, 1] / B[0, 0] if abs(B[0,0]) > 1e-10 else 0
                A[0, 0] = K[0, 0] / B[0, 0] if abs(B[0,0]) > 1e-10 else 0
                A[1, 0] = K[2, 0] / B[0, 0] if abs(B[0,0]) > 1e-10 else 0
                A = A / np.linalg.det(A) ** 0.5
                B = B / np.linalg.det(B) ** 0.5
                return A, B
            elif abs(K[1, 0]) > 1e-10:
                B = K[1:3, 0:2].copy()
                A = np.zeros((2, 2), dtype=complex)
                A[1, 0] = 1.0
                A[0, 0] = K[0, 0] / B[0, 0] if abs(B[0,0]) > 1e-10 else 0
                A[1, 1] = K[3, 0] / B[0, 0] if abs(B[0,0]) > 1e-10 else 0
                A[0, 1] = K[0, 1] / B[0, 0] if abs(B[0,0]) > 1e-10 else 0
                A = A / np.linalg.det(A) ** 0.5
                B = B / np.linalg.det(B) ** 0.5
                return A, B
            else:
                # Fallback: return identity
                return np.eye(2, dtype=complex), np.eye(2, dtype=complex)
    
    A1, A2 = _extract_tensor(K1)
    
    # Now for the CNOT decomposition:
    # U = (A1⊗A2) M (B1⊗B2)
    # M = CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    # We need to find B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, G1, G2, H1, H2
    
    # Standard CNOT decomposition of M:
    # M = (X⊗I) CNOT (Y⊗I) CNOT (Z⊗I) CNOT (W⊗I)
    # where X, Y, Z, W are 2x2 unitaries
    
    # For the specific M from KAK:
    # exp(iθ1 sx⊗sx/2) = cos(θ1/2) I + i sin(θ1/2) sx⊗sx
    # This can be decomposed as:
    # (R_x(θ1/2) ⊗ R_x(θ1/2)) CNOT (R_x(-θ1/2) ⊗ R_x(-θ1/2)) ... 
    # Actually, let me use a known decomposition.
    
    # A simpler approach: use the fact that we can write M as a product of 3 CNOTs
    # with specific single-qubit rotations.
    
    # For now, let me use a numerical approach to find the decomposition.
    # Actually, let me just use the known formula for CNOT decomposition of Pauli products.
    
    # exp(iθ σx⊗σx/2) = (H⊗H) CNOT (R_z(θ)⊗I) CNOT (H⊗H) ... no that's not right.
    
    # Let me use a different approach: just optimize the single-qubit gates.
    # For 2 qubits with 3 CNOTs, we have 12 single-qubit gates (3 parameters each) = 36 params
    # But the CNOT structure is fixed, so we just need to find the right rotations.
    
    # Actually, let me just return the KAK angles and let the caller use optimization
    # if needed, or use a known decomposition.
    
    # For the purpose of this task, let me use a simple approach:
    # Just use the KAK angles and a known decomposition pattern.
    
    # The key insight: for 2-qubit unitaries, we can always decompose into
    # (A1⊗A2) CNOT (B1⊗B2) CNOT (C1⊗C2) CNOT (D1⊗D2)
    # Let me just use optimization for the single-qubit parts.
    
    return A1, A2, t1, t2, t3


def _optimize_unitary(target, qubits, budget, max_time=30.0):
    """Optimize single-qubit rotations to match target unitary within CNOT budget."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    dim = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Create a layout: single-qubit rotations before each CNOT and after the last CNOT
    layout = []
    for layer in range(budget):
        layout.append(("u3", list(range(qubits))))
        layout.append(("cx", pairs[layer % len(pairs)]))
    layout.append(("u3", list(range(qubits))))
    
    # Count parameters
    n_params = 0
    for item in layout:
        if item[0] == "u3":
            n_params += 3 * len(item[1])
    
    # Initialize parameters
    np.random.seed(42)
    params = np.random.uniform(-math.pi, math.pi, size=n_params)
    
    def _build_from_params(p):
        """Build unitary from parameter vector."""
        matrix = np.eye(dim, dtype=complex)
        eye = np.eye(2, dtype=complex)
        idx = 0
        for item in layout:
            if item[0] == "u3":
                for q in item[1]:
                    theta = p[idx]
                    phi = p[idx + 1]
                    lam = p[idx + 2]
                    idx += 3
                    single = _u3_matrix(theta, phi, lam)
                    operator = np.array([[1]], dtype=complex)
                    for k in range(qubits):
                        operator = np.kron(operator, single if k == q else eye)
                    matrix = operator @ matrix
            else:
                control, target_q = item[1]
                index = np.arange(dim)
                bit = (index >> (qubits - 1 - control)) & 1
                matrix = matrix[index ^ (bit << (qubits - 1 - target_q)), :]
        return matrix
    
    def _fidelity(p):
        U = _build_from_params(p)
        return abs(np.trace(target.conj().T @ U)) / dim
    
    # Gradient-based optimization using finite differences
    best_params = params.copy()
    best_fid = _fidelity(best_params)
    
    start = time.time()
    step_size = 0.5
    lr = 0.1
    
    for iteration in range(200):
        if time.time() - start > max_time or time.time() - _start_time > _DEADLINE:
            break
        
        # Compute gradient using central differences
        grad = np.zeros_like(params)
        eps = 0.01
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (_fidelity(p_plus) - _fidelity(p_minus)) / (2 * eps)
        
        # Gradient ascent
        params = params + lr * grad
        # Wrap angles to [-pi, pi]
        params = (params + math.pi) % (2 * math.pi) - math.pi
        
        fid = _fidelity(params)
        if fid > best_fid:
            best_fid = fid
            best_params = params.copy()
        
        # Adaptive learning rate
        if fid < best_fid - 0.001:
            lr *= 0.8
        else:
            lr = min(lr * 1.05, 0.5)
        
        if best_fid > 0.9999:
            break
    
    # Convert best_params to gate list
    gates = []
    idx = 0
    for item in layout:
        if item[0] == "u3":
            for q in item[1]:
                gates.append(_u3(q, best_params[idx], best_params[idx + 1], best_params[idx + 2]))
                idx += 3
        else:
            gates.append(_cx(*item[1]))
    
    return gates, 1.0 - best_fid


def synthesize(problem):
    """Synthesize quantum circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if qubits == 2:
        # Check if it's CNOT
        cnot_target = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)
        if _distance(target, cnot_target) < 1e-6:
            return _exact_cnot()
        
        # Check if it's SWAP
        swap_target = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=complex)
        if _distance(target, swap_target) < 1e-6:
            return _exact_swap()
        
        # Check if it's iSWAP (up to global phase)
        iswap_target = np.array([[0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0]], dtype=complex)
        if _distance(target, iswap_target) < 1e-6:
            return _exact_iswap()
        
        # Also check iSWAP with different phase conventions
        iswap_target2 = np.array([[0, 1j, 0, 0],
                                   [1j, 0, 0, 0],
                                   [0, 0, 0, 1j],
                                   [0, 0, 1j, 0]], dtype=complex)
        if _distance(target, iswap_target2) < 1e-6:
            # This is a different iSWAP convention, use optimization
            pass
    
    if qubits == 3:
        # Check if it's Toffoli
        toffoli_target = np.array([[1,0,0,0,0,0,0,0],
                                     [0,1,0,0,0,0,0,0],
                                     [0,0,1,0,0,0,0,0],
                                     [0,0,0,1,0,0,0,0],
                                     [0,0,0,0,1,0,0,0],
                                     [0,0,0,0,0,1,0,0],
                                     [0,0,0,0,0,0,0,1],
                                     [0,0,0,0,0,0,1,0]], dtype=complex)
        if _distance(target, toffoli_target) < 1e-6:
            return _exact_toffoli()
    
    # For generic unitaries, use optimization
    remaining_time = _DEADLINE - (time.time() - _start_time)
    max_time = min(30.0, remaining_time)
    if max_time < 5:
        max_time = 5
    
    gates, dist = _optimize_unitary(target, qubits, budget, max_time=max_time)
    return gates
# EVOLVE-BLOCK-END
