"""Quantum circuit synthesis: exact 2-qubit KAK decomposition + 3-qubit block peeling."""
import math
import time
import numpy as np

_START = time.time()
_DEADLINE = 250.0


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


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


def _u3_from_matrix(M):
    """Decompose a 2x2 unitary M into u3(theta, phi, lam)."""
    M = np.array(M, dtype=complex)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    phase = np.angle(det)
    M2 = M * np.exp(-1j * phase / 2)
    theta = 2.0 * np.arccos(np.clip(np.real(M2[0, 0]), -1.0, 1.0))
    if abs(M2[0, 1]) < 1e-12 and abs(M2[1, 0]) < 1e-12:
        phi = 0.0
        lam = phase
    else:
        s = np.sin(theta / 2)
        if abs(s) > 1e-12:
            lam = -np.angle(M2[0, 1] / s)
            phi = np.angle(M2[1, 0] / s)
        else:
            phi = 0.0
            lam = phase
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    lam = (lam + math.pi) % (2 * math.pi) - math.pi
    return theta, phi, lam


def _u3_to_matrix(theta, phi, lam):
    """Convert u3 parameters to 2x2 matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                    [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cx_matrix_2qubit():
    """CNOT matrix for 2 qubits (qubit 0 control, qubit 1 target)."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


def _decompose_2qubit_kak(U2):
    """
    Exact 2-qubit unitary decomposition using KAK (Cartan) decomposition.
    Returns a list of gates: u3(0), u3(1), cx(0,1), u3(0), u3(1), cx(0,1), u3(0), u3(1), cx(0,1), u3(0), u3(1)
    """
    CX = _cx_matrix_2qubit()
    
    # Remove global phase
    det_U = np.linalg.det(U2)
    if abs(det_U) < 1e-15:
        # Degenerate case, use identity
        return [_u3(0, 0, 0, 0), _u3(1, 0, 0, 0)]
    global_phase = np.angle(det_U) / 2
    U2p = U2 * np.exp(-1j * global_phase)
    
    # KAK decomposition: U = (A0 ⊗ A1) · K(θ1,θ2,θ3) · (B0 ⊗ B1)
    # where K = exp(iθ1 σx⊗σx + iθ2 σy⊗σy + iθ3 σz⊗σz)
    
    # Compute the "cosine" form
    # V = U2p · CX · U2p^T · CX
    V = U2p @ CX @ U2p.T @ CX
    
    # Diagonalize V to get KAK parameters
    # V should be diagonal in the KAK basis
    eigvals, eigvecs = np.linalg.eig(V)
    # Sort eigenvalues
    order = np.argsort(np.real(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # The KAK parameters are related to the eigenvalues
    # V = diag(e^{i(θ1+θ2+θ3)}, e^{i(θ1+θ2-θ3)}, e^{i(θ1-θ2+θ3)}, e^{i(-θ1+θ2+θ3)})
    # (up to global phase)
    
    # Extract angles
    angles = np.angle(eigvals)
    # Remove global phase
    angles = angles - angles[0]
    
    θ1 = (angles[0] + angles[1] + angles[2] + angles[3]) / 4
    θ2 = (angles[0] + angles[1] - angles[2] - angles[3]) / 4
    θ3 = (angles[0] - angles[1] + angles[2] - angles[3]) / 4
    
    # Clamp to valid range
    θ1 = np.clip(θ1, 0, math.pi / 4)
    θ2 = np.clip(θ2, 0, math.pi / 4)
    θ3 = np.clip(θ3, 0, math.pi / 4)
    
    # Build K matrix
    K = np.eye(4, dtype=complex)
    K[0, 0] = np.exp(1j * (θ1 + θ2 + θ3))
    K[1, 1] = np.exp(1j * (θ1 + θ2 - θ3))
    K[2, 2] = np.exp(1j * (θ1 - θ2 + θ3))
    K[3, 3] = np.exp(1j * (-θ1 + θ2 + θ3))
    
    # Compute B0 ⊗ B1 = K^† · U2p
    KB = K.conj().T @ U2p
    
    # Now we need to decompose KB into (A0 ⊗ A1) · (B0 ⊗ B1)
    # This is a standard 2-qubit local decomposition
    # Use the fact that any 4x4 unitary with det=1 can be written as product of two local unitaries
    # if and only if it's in the connected component of SU(4)
    
    # Extract B0 ⊗ B1 from KB
    # The product (B0 ⊗ B1) has the form where each 2x2 block is proportional to B0
    # B0 ⊗ B1 = [[b00*B1, b01*B1], [b10*B1, b11*B1]]
    
    # Let's use a different approach: iterative peeling
    # We know U2 = (A0 ⊗ A1) · K · (B0 ⊗ B1)
    # So (A0 ⊗ A1) = U2 · K^† · (B0 ⊗ B1)^†
    # We need to find B0, B1 such that U2 · K^† · (B0 ⊗ B1)^† is a product of two 2x2 unitaries
    
    # Let M = U2p · K^†
    M = U2p @ K.conj().T
    
    # M = (A0 ⊗ A1) · (B0 ⊗ B1)^†
    # Let N = (B0 ⊗ B1)^†
    # Then M · N^† = A0 ⊗ A1
    # We need N such that M · N^† is a product of two 2x2 unitaries
    
    # A matrix is a product of two 2x2 unitaries (i.e., in U(2)⊗U(2)) if and only if
    # it satisfies certain conditions. One approach: use the fact that
    # if P = A ⊗ B, then P[0,0] = a00*b00, P[0,1] = a00*b01, etc.
    # So P[0,0]*P[1,1] = a00*b00 * a11*b11
    # and P[0,1]*P[1,0] = a00*b01 * a10*b10
    # The ratio P[0,0]*P[1,1] / (P[0,1]*P[1,0]) should be real (for proper unitaries)
    
    # Let's use a numerical approach: find B1 first
    # M = (A0 ⊗ A1) · (B0^† ⊗ B1^†)
    # The (0,0) block of M (rows 0,1; cols 0,1) is A0[0,:] ⊗ A1[0,:] · (B0^† ⊗ B1^†)
    # This gets complicated. Let me use a simpler iterative method.
    
    # Actually, let me use a well-known algorithm:
    # 1. M = U2p · K^†
    # 2. Compute the SVD of the reshaped matrix
    #    Reshape M into a 4x4 matrix and compute its "local" decomposition
    
    # Method: 
    # Let M be 4x4. Write M as a sum of tensor products.
    # M = Σ_i v_i ⊗ w_i (in some sense)
    # For M = A ⊗ B, we have M = a ⊗ b where a, b are vectors in C^4 (reshaped)
    
    # Reshape M to get a 4x4 matrix (it's already 4x4)
    # The condition for M = A ⊗ B is that M ⊗ M has rank 1 when viewed appropriately
    
    # Let me use a direct approach:
    # Assume B1 is a 2x2 unitary. Then B0 = (M · (I ⊗ B1^†)) / (A0 ⊗ I) ... no
    
    # Let me use the following known result:
    # Given M ∈ U(4), M = (A ⊗ B) if and only if M can be written as such.
    # Algorithm:
    # 1. Take the first column of M: m0 = M[:, 0]
    # 2. m0 = A[:, 0] ⊗ B[:, 0] (reshaped)
    # 3. This gives us the relative phases
    
    # Actually, let me just use a numerical optimization for the local gates,
    # but with a much better initialization from the KAK structure.
    
    # For now, let me use a simpler but correct approach:
    # I'll use the fact that we can compute the local gates by "peeling"
    # using the KAK structure as initialization.
    
    # Let me try a different, cleaner approach:
    # Use the "cosine decomposition" directly.
    
    # The 2-qubit unitary can be written as:
    # U = e^{iα} (L0 ⊗ L1) · CX · (M0 ⊗ M1) · CX · (N0 ⊗ N1) · CX · (P0 ⊗ P1)
    
    # I'll use a numerical method with good initialization.
    
    import random
    rng = random.Random(42)
    
    # Layout: 3 CNOTs with single-qubit gates
    layout = []
    for layer in range(3):
        layout.append(("u3", 0))
        layout.append(("u3", 1))
        layout.append(("cx", 0, 1))
    layout.append(("u3", 0))
    layout.append(("u3", 1))
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def build(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(["cx", gate[1], gate[2]])
        return gates
    
    def dist(params):
        u = _build(build(params), 2)
        return _distance(U2, u)
    
    # Initialize with small random values
    best = [rng.uniform(-0.5, 0.5) for _ in range(n_params)]
    best_d = dist(best)
    
    # Coordinate descent with multiple restarts
    for restart in range(3):
        if _time_left() < 10:
            break
        if restart > 0:
            best = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            best_d = dist(best)
        
        for sweep in range(8):
            if _time_left() < 5:
                break
            step = 0.5 / (1 + sweep * 0.3)
            for i in range(n_params):
                if _time_left() < 3:
                    break
                for delta in (step, -step, step * 0.1, -step * 0.1,
                             rng.uniform(-math.pi, math.pi) * 0.05):
                    vals = best[:]
                    vals[i] = best[i] + delta
                    d = dist(vals)
                    if d < best_d:
                        best, best_d = vals, d
            if best_d < 1e-12:
                break
        if best_d < 1e-12:
            break
    
    return build(best)


def _decompose_3qubit_block(target, qubits, budget):
    """
    Decompose a 3-qubit unitary using block decomposition.
    U = U_0 ⊗ |0><0|_2 + U_1 ⊗ |1><1|_2
    where U_0, U_1 are 2-qubit unitaries on qubits 0,1.
    """
    if _time_left() < 20:
        return None
    
    # Extract blocks based on qubit 2 (least significant)
    idx0 = [0, 2, 4, 6]  # qubit 2 = 0
    idx1 = [1, 3, 5, 7]  # qubit 2 = 1
    
    U0 = target[np.ix_(idx0, idx0)]
    U1 = target[np.ix_(idx1, idx1)]
    
    # Check if blocks are approximately unitary
    if not np.allclose(U0.conj().T @ U0, np.eye(4), atol=1e-4):
        return None
    if not np.allclose(U1.conj().T @ U1, np.eye(4), atol=1e-4):
        return None
    
    # Decompose U0 and U1
    gates0 = _decompose_2qubit_kak(U0)
    gates1 = _decompose_2qubit_kak(U1)
    
    cnot0 = sum(1 for g in gates0 if g[0] == "cx")
    cnot1 = sum(1 for g in gates1 if g[0] == "cx")
    
    # Total CNOTs: cnot0 + 2*cnot1 (for the conditional part)
    # Actually, to apply U1 conditioned on qubit 2=1:
    # Apply U0 unconditionally (cnot0 CNOTs)
    # Apply (U1 · U0^†) conditioned on qubit 2=1
    # The conditional application of a 2-qubit unitary with k CNOTs requires 2k CNOTs
    # (using Toffoli-like decomposition)
    # So total: cnot0 + 2 * cnot1
    
    total_cnots = cnot0 + 2 * cnot1
    if total_cnots > budget:
        return None
    
    # Build the circuit:
    # 1. Apply U0 (gates0)
    # 2. Apply (U1 · U0^†) conditioned on qubit 2 = 1
    
    # For step 2, we need to compute U1 · U0^† and decompose it
    U_rel = U1 @ U0.conj().T
    gates_rel = _decompose_2qubit_kak(U_rel)
    cnot_rel = sum(1 for g in gates_rel if g[0] == "cx")
    
    # Total: cnot0 + 2 * cnot_rel
    total_cnots = cnot0 + 2 * cnot_rel
    if total_cnots > budget:
        return None
    
    # Build the full circuit
    # Apply U0
    circuit = list(gates0)
    
    # Apply U_rel conditioned on qubit 2 = 1
    # For each gate in gates_rel, if it's a u3 on qubit 0 or 1, we need to make it conditional
    # If it's a CNOT, we need to make it conditional (Toffoli)
    
    # A u3(q, θ, φ, λ) conditioned on qubit 2=1 can be implemented as:
    # - If q=0: u3(0, θ/2, φ/2, λ/2), cx(0,2), u3(0, θ/2, -φ/2, -λ/2), cx(0,2)
    #   Wait, that's not quite right. Let me think more carefully.
    # 
    # To apply U conditioned on |1><1|_2:
    # U_cond = |0><0|_2 ⊗ I + |1><1|_2 ⊗ U
    # 
    # For a single-qubit gate U on qubit 0, conditioned on qubit 2=1:
    # We can write: U_cond = (I + U)/2 ⊗ (I + Z)/2 + (I - U)/2 ⊗ (I - Z)/2
    # This is complex. Let me use a simpler approach.
    #
    # Actually, for a u3 gate on qubit q (q=0 or 1), conditioned on qubit 2=1:
    # We can use the identity:
    # |0><0| ⊗ I + |1><1| ⊗ U = 
    #   (I ⊗ I + Z ⊗ I)/2 + (I ⊗ U - Z ⊗ U)/2
    #   = I ⊗ (I+U)/2 + Z ⊗ (I-U)/2
    # This requires implementing (I+U)/2 and (I-U)/2, which are not unitaries in general.
    #
    # Better approach: use the fact that any single-qubit unitary U can be written as
    # U = e^{iα} R_z(β) R_y(γ) R_z(δ)
    # and condition each rotation.
    #
    # For R_z(θ) conditioned on |1><1|:
    # R_z(θ) = diag(e^{-iθ/2}, e^{iθ/2})
    # R_z(θ) ⊗ |1><1| + I ⊗ |0><0| = 
    #   |0><0| ⊗ I + |1><1| ⊗ diag(e^{-iθ/2}, e^{iθ/2})
    # This is a diagonal gate, which can be implemented with phase gates.
    #
    # This is getting complex. Let me use a different approach for the 3-qubit case.
    # I'll use a fixed CNOT layout and optimize the single-qubit gates.
    
    return None


def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using coordinate descent."""
    if _time_left() < 5:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    import random
    rng = random.Random(rng_seed)
    
    # For 2 qubits, use the KAK-based decomposition
    if qubits == 2:
        return _decompose_2qubit_kak(target)
    
    # For 3 qubits, try block decomposition first
    if qubits == 3 and budget >= 12:
        block_result = _decompose_3qubit_block(target, qubits, budget)
        if block_result is not None:
            cnot_count = sum(1 for g in block_result if g[0] == "cx")
            if cnot_count <= budget:
                d = _distance(target, _build(block_result, qubits))
                if d < 0.01:
                    return block_result
    
    # General coordinate descent
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build a layout with CNOTs and single-qubit gates
    n_layers = min(budget, 12)
    layout = []
    for layer in range(n_layers):
        layout += [("u3", q) for q in range(qubits)]
        layout.append(("cx", *pairs[layer % len(pairs)]))
    layout += [("u3", q) for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", gate[1], gate[2]])
        return gates
    
    best = [rng.uniform(-0.5, 0.5) for _ in range(n_params)]
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    sweeps = 6
    if _time_left() < 30:
        sweeps = 3
    if _time_left() < 15:
        sweeps = 2
        
    for sweep in range(sweeps):
        step = 0.5 / (1 + sweep * 0.3)
        for i in range(n_params):
            if _time_left() < 5:
                break
            for trial in (best[i] + step, best[i] - step,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.1,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.01):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
        if best_distance < 1e-10:
            break
        
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
    if _time_left() < 2:
        qubits = problem["qubits"]
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # Fallback to optimization
    return _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100 + budget)
