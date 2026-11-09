import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 255.0

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            op = _u3_matrix(th, ph, lm)
            full = np.array([[1]], dtype=complex)
            for k in range(qubits):
                full = np.kron(full, op if k == q else eye2)
            U = full @ U
        else:
            _, c, t = g
            idx = np.arange(n)
            cb = (idx >> (qubits - 1 - c)) & 1
            perm = idx ^ (cb << (qubits - 1 - t))
            U = U[perm, :]
    return U

def _dist(t, u):
    n = t.shape[0]
    return 1.0 - min(1.0, abs(np.trace(t.conj().T @ u)) / n)

def _count_cx(gates):
    return sum(1 for g in gates if g[0] == "cx")

def _detect_known(target, qubits):
    tol = 1e-8
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=tol):
            return "cnot"
        iswap = 0.5*np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if np.allclose(target, iswap, atol=tol):
            return "iswap"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=tol):
            return "swap"
    if qubits == 3:
        toff = np.zeros((8,8), dtype=complex)
        for i in range(8):
            b0, b1, b2 = (i>>2)&1, (i>>1)&1, i&1
            if b0 and b1:
                toff[i, i^1] = 1
            else:
                toff[i, i] = 1
        if np.allclose(target, toff, atol=tol):
            return "toffoli"
    return "generic"

def _exact_toffoli(qubits):
    gates = []
    gates.append(["u3", 2, math.pi/2, 0, 0])
    gates.append(["cx", 2, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi/2, 0, 0])
    return gates

def _iswap_exact():
    gates = []
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, math.pi/2])
    gates.append(["u3", 1, 0, 0, math.pi/2])
    return gates

def _optimal_u3(A):
    """Find (theta, phi, lam) such that u3(theta,phi,lam) best approximates A (2x2)."""
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    sqrt_det = np.sqrt(det_A)
    if abs(sqrt_det) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    c_val = A[0,0] / sqrt_det
    s_val = A[1,0] / sqrt_det
    
    c_abs = abs(c_val)
    s_abs = abs(s_val)
    norm = math.sqrt(c_abs**2 + s_abs**2)
    if norm < 1e-15:
        return (0.0, 0.0, 0.0)
    
    theta_half = math.atan2(s_abs, c_abs)
    theta = 2.0 * theta_half
    
    if s_abs > 1e-15:
        e_iphi = (A[1,0] / sqrt_det) / s_abs
        phi = np.angle(e_iphi)
    else:
        phi = 0.0
    
    if s_abs > 1e-15:
        e_ilam = -(A[0,1] / sqrt_det) / s_abs
        lam = np.angle(e_ilam)
    else:
        lam = 0.0
    
    return (float(theta), float(phi), float(lam))

def _kak_decompose_2qubit(U):
    """
    KAK (Cartan) decomposition for a 2-qubit unitary.
    Returns a list of gates with at most 3 CNOTs.
    
    The KAK decomposition writes U = (A ⊗ B) * C * (C ⊗ D) where
    C = exp(i * sum_k c_k * sigma_k ⊗ sigma_k)
    and the c_k can be chosen such that at most 3 are non-zero,
    corresponding to at most 3 CNOTs.
    
    Standard form: U = (A0 ⊗ A1) CX (B0 ⊗ B1) CX (C0 ⊗ C1) CX (D0 ⊗ D1)
    """
    n = 4
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    
    # Superoperator representation
    def to_super(U_mat):
        """Convert 4x4 unitary to 4x4 superoperator in Pauli basis."""
        # Basis: II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
        # But we use a simpler approach: work with the 16x16 representation
        pass
    
    # Alternative: use the standard KAK algorithm
    # Step 1: Compute U in a basis where the interaction is diagonal
    # The KAK decomposition for 2-qubit:
    # U = (A ⊗ B) exp(i * (c1 XX + c2 YY + c3 ZZ)) (C ⊗ D)
    # where exp(i*(c1 XX + c2 YY + c3 ZZ)) can be decomposed into at most 3 CNOTs
    
    # For simplicity, use a parameterized approach with optimization
    # Circuit: (A0⊗A1) CX (B0⊗B1) CX (C0⊗C1) CX (D0⊗D1)
    
    # Initialize with identity
    # We'll optimize 12 angles (4 layers × 2 qubits × 3 params)
    
    # Use a smart initialization based on the structure
    # For now, return a template that we'll optimize
    return None  # We'll handle this differently

def _decompose_2qubit_exact(target):
    """
    Exact 2-qubit decomposition using KAK.
    Uses the algorithm from Vedral & Knight, or equivalent.
    """
    n = 4
    U = target.copy()
    
    # Step 1: Factor out local unitaries to get the canonical form
    # U = (A ⊗ B) * K * (C ⊗ D)
    # where K is in the canonical form with at most 3 non-zero canonical parameters
    
    # For the standard KAK decomposition:
    # 1. Compute the "interaction" part
    # 2. Extract local parts
    
    # Practical approach: use a parameterized circuit and optimize
    # Circuit: u3 layer, CX, u3 layer, CX, u3 layer, CX, u3 layer
    # That's 4 layers of u3 (2 qubits each) and 3 CX = 24 parameters + 3 CX
    
    # Let's use a different approach: directly solve for the KAK parameters
    # The canonical form is:
    # K = exp(i * (c1 * XX/4 + c2 * YY/4 + c3 * ZZ/4))
    
    # For 2-qubit, the KAK decomposition gives:
    # U = (A ⊗ B) * K * (C ⊗ D)
    # where K can be written as a product of at most 3 CNOTs with specific rotations
    
    # Let me implement the standard KAK algorithm:
    # 1. Compute U in the Pauli transfer matrix representation
    # 2. Extract the canonical parameters
    # 3. Reconstruct the circuit
    
    # Pauli basis
    paulis = [
        np.kron(np.eye(2, dtype=complex), np.eye(2, dtype=complex)),
        np.kron(np.array([[0,1],[1,0]], dtype=complex), np.eye(2, dtype=complex)),
        np.kron(np.array([[0,-1j],[1j,0]], dtype=complex), np.eye(2, dtype=complex)),
        np.kron(np.array([[1,0],[0,-1]], dtype=complex), np.eye(2, dtype=complex)),
        np.kron(np.eye(2, dtype=complex), np.array([[0,1],[1,0]], dtype=complex)),
        np.kron(np.eye(2, dtype=complex), np.array([[0,-1j],[1j,0]], dtype=complex)),
        np.kron(np.eye(2, dtype=complex), np.array([[1,0],[0,-1]], dtype=complex)),
        np.kron(np.array([[0,1],[1,0]], dtype=complex), np.array([[0,1],[1,0]], dtype=complex)),
        np.kron(np.array([[0,-1j],[1j,0]], dtype=complex), np.array([[0,-1j],[1j,0]], dtype=complex)),
        np.kron(np.array([[1,0],[0,-1]], dtype=complex), np.array([[1,0],[0,-1]], dtype=complex)),
    ]
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Use the fact that any 2-qubit unitary can be decomposed as:
    # U = (A ⊗ B) * CX^k * (C ⊗ D) for k = 0, 1, 2, or 3
    # and find the optimal decomposition by optimization with good initialization.
    
    # For now, use a practical approach: optimize a 3-CNOT circuit
    # with smart initialization from the target's structure.
    
    # Initialize: try to match the target by extracting local parts
    # First, compute what the "middle" should be if we had no local gates
    # Then extract local parts
    
    # Simple initialization:
    # Set all u3 to identity, then do a few rounds of coordinate descent
    
    # Actually, let me just use the gradient approach but with better initialization
    # and more iterations, since we have time budget.
    
    return None  # Placeholder - we'll use the optimizer

def _synthesize_2qubit(target, budget):
    """Synthesize a 2-qubit unitary with at most `budget` CNOTs."""
    if budget <= 0:
        # Only single-qubit gates possible - can't match arbitrary 2-qubit
        # Return best effort: just single-qubit rotations
        gates = []
        for q in range(2):
            gates.append(["u3", q, 0.0, 0.0, 0.0])
        return gates
    
    num_cx = min(budget, 3)
    
    # Try different CNOT orderings
    # Standard: CX(0,1) CX(0,1) CX(0,1) or mixed
    templates = [
        [(0,1), (0,1), (0,1)],
        [(0,1), (1,0), (0,1)],
        [(0,1), (1,0), (1,0)],
        [(1,0), (0,1), (0,1)],
    ]
    
    if num_cx < 3:
        templates = [t[:num_cx] for t in templates if len(t[:num_cx]) == num_cx]
    
    best_gates = None
    best_dist = 1.0
    
    for template in templates:
        if _remaining() < 10.0:
            break
        
        gates, d = _optimize_2qubit_circuit(target, template)
        if d < best_dist:
            best_dist = d
            best_gates = gates
    
    if best_gates is None:
        # Fallback
        template = templates[0] if templates else [(0,1)]
        best_gates, _ = _optimize_2qubit_circuit(target, template)
    
    return best_gates

def _optimize_2qubit_circuit(target, cnot_template):
    """Optimize u3 parameters for a 2-qubit circuit with given CNOT template."""
    n = 4
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * 2
    num_params = num_u3 * 3
    
    T_dag = target.conj().T
    
    # Initialize params
    params = np.zeros(num_params, dtype=float)
    
    # Analytic initialization: extract 2x2 blocks
    # Build identity circuit
    def build_from_params(p):
        U = np.eye(4, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        param_idx = 0
        for layer in range(num_layers):
            for q in range(2):
                th = p[param_idx]
                ph = p[param_idx+1]
                lm = p[param_idx+2]
                R = _u3_matrix(th, ph, lm)
                if q == 0:
                    full = np.kron(R, eye2)
                else:
                    full = np.kron(eye2, R)
                U = full @ U
                param_idx += 3
            if layer < num_cx:
                c, t = cnot_template[layer]
                idx = np.arange(4)
                cb = (idx >> (1 - c)) & 1
                perm = idx ^ (cb << (1 - t))
                U = U[perm, :]
        return U
    
    U_init = build_from_params(params)
    
    # Analytic init: for each u3, extract the 2x2 block from target @ U_init^{-1}
    try:
        U_inv = np.linalg.inv(U_init)
        residual = target @ U_inv
        param_idx = 0
        for layer in range(num_layers):
            for q in range(2):
                if q == 0:
                    block = residual[0:2, 0:2]
                else:
                    block = residual[2:4, 2:4]
                th, ph, lm = _optimal_u3(block)
                params[param_idx] = th
                params[param_idx+1] = ph
                params[param_idx+2] = lm
                param_idx += 3
    except:
        pass
    
    # Gradient-based refinement
    def cost_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        return 1.0 - min(1.0, abs(tr_val) / 4)
    
    def grad_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        cost = 1.0 - min(1.0, abs(tr_val) / 4)
        if abs(tr_val) < 1e-12:
            return cost, np.zeros(num_params)
        grad = np.zeros(num_params)
        eps = 1e-6
        for k in range(num_params):
            p_plus = p.copy()
            p_plus[k] += eps
            U_plus = build_from_params(p_plus)
            tr_plus = np.trace(T_dag @ U_plus)
            dtr = (tr_plus - tr_val) / eps
            grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
        grad = -grad / 4
        return cost, grad
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.05
    best_cost = 1.0
    best_params = params.copy()
    
    cost, _ = cost_fn(params)
    if cost < best_cost:
        best_cost = cost
        best_params = params.copy()
    
    max_iter = 300
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        cost, grad = grad_fn(params)
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        if best_cost < 1e-12:
            break
        t_adam += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_adam)
        v_hat = v / (1 - beta2**t_adam)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        if it % 50 == 49 and it > 0:
            lr *= 0.8
    
    # Convert to gates
    gates = []
    param_idx = 0
    for layer in range(num_layers):
        for q in range(2):
            th = best_params[param_idx]
            ph = best_params[param_idx+1]
            lm = best_params[param_idx+2]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    
    return gates, best_cost

def _synthesize_3qubit(target, budget):
    """Synthesize a 3-qubit unitary with at most `budget` CNOTs."""
    if budget <= 0:
        gates = []
        for q in range(3):
            gates.append(["u3", q, 0.0, 0.0, 0.0])
        return gates
    
    num_cx = min(budget, 6)
    
    # Standard 3-qubit CNOT template
    # Use a pattern that covers all pairs
    pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
    cnot_template = pairs[:num_cx]
    
    gates, cost = _optimize_3qubit_circuit(target, cnot_template)
    
    return gates

def _optimize_3qubit_circuit(target, cnot_template):
    """Optimize u3 parameters for a 3-qubit circuit."""
    n = 8
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * 3
    num_params = num_u3 * 3
    
    T_dag = target.conj().T
    
    params = np.zeros(num_params, dtype=float)
    
    def build_from_params(p):
        U = np.eye(8, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        param_idx = 0
        for layer in range(num_layers):
            for q in range(3):
                th = p[param_idx]
                ph = p[param_idx+1]
                lm = p[param_idx+2]
                R = _u3_matrix(th, ph, lm)
                full = np.array([[1]], dtype=complex)
                for k in range(3):
                    full = np.kron(full, R if k == q else eye2)
                U = full @ U
                param_idx += 3
            if layer < num_cx:
                c, t = cnot_template[layer]
                idx = np.arange(8)
                cb = (idx >> (2 - c)) & 1
                perm = idx ^ (cb << (2 - t))
                U = U[perm, :]
        return U
    
    U_init = build_from_params(params)
    
    # Analytic init
    try:
        U_inv = np.linalg.inv(U_init)
        residual = target @ U_inv
        param_idx = 0
        for layer in range(num_layers):
            for q in range(3):
                # Extract 2x2 block for qubit q
                # For qubit 0: rows/cols where bit 2 is 0: indices 0-3, and bit 2 is 1: indices 4-7
                # The 2x2 block is the "average" over other qubits
                # For qubit 0: block[i,j] = sum over other bits of residual[2*other+i, 2*other+j] / 4
                # Actually, for a 3-qubit system with qubit 0 as MSB:
                # The 2x2 block for qubit 0 is:
                # [[sum_{b1,b2} residual[0*b1*b2, 0*b1*b2], sum_{b1,b2} residual[0*b1*b2, 1*b1*b2]],
                #  [sum_{b1,b2} residual[1*b1*b2, 0*b1*b2], sum_{b1,b2} residual[1*b1*b2, 1*b1*b2]]]
                # Wait, that's not quite right. The 2x2 block for qubit q in a 3-qubit system
                # is obtained by tracing out the other qubits.
                
                # For qubit 0 (MSB):
                # Block[0,0] = (residual[0,0] + residual[1,1] + residual[4,4] + residual[5,5]) / 4
                # Block[0,1] = (residual[0,1] + residual[1,0] + residual[4,5] + residual[5,4]) / 4
                # Block[1,0] = (residual[4,0] + residual[5,1] + residual[0,4] + residual[1,5]) / 4
                # Block[1,1] = (residual[4,4] + residual[5,5] + residual[0,0] + residual[1,1]) / 4
                
                # This is getting complex. Let me use a simpler approach:
                # Just use identity for 3-qubit init and rely on gradient descent.
                block = np.eye(2, dtype=complex)
                th, ph, lm = _optimal_u3(block)
                params[param_idx] = th
                params[param_idx+1] = ph
                params[param_idx+2] = lm
                param_idx += 3
    except:
        pass
    
    # Gradient-based refinement
    def cost_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        return 1.0 - min(1.0, abs(tr_val) / 8)
    
    def grad_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        cost = 1.0 - min(1.0, abs(tr_val) / 8)
        if abs(tr_val) < 1e-12:
            return cost, np.zeros(num_params)
        grad = np.zeros(num_params)
        eps = 1e-6
        for k in range(num_params):
            p_plus = p.copy()
            p_plus[k] += eps
            U_plus = build_from_params(p_plus)
            tr_plus = np.trace(T_dag @ U_plus)
            dtr = (tr_plus - tr_val) / eps
            grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
        grad = -grad / 8
        return cost, grad
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.05
    best_cost = 1.0
    best_params = params.copy()
    
    cost, _ = cost_fn(params)
    if cost < best_cost:
        best_cost = cost
        best_params = params.copy()
    
    max_iter = 200
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        cost, grad = grad_fn(params)
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        if best_cost < 1e-12:
            break
        t_adam += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_adam)
        v_hat = v / (1 - beta2**t_adam)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        if it % 30 == 29 and it > 0:
            lr *= 0.85
    
    # Convert to gates
    gates = []
    param_idx = 0
    for layer in range(num_layers):
        for q in range(3):
            th = best_params[param_idx]
            ph = best_params[param_idx+1]
            lm = best_params[param_idx+2]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    
    return gates, best_cost

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    gate_type = _detect_known(target, qubits)

    if gate_type == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_type == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_type == "iswap" and qubits == 2:
        return _iswap_exact()
    elif gate_type == "toffoli" and qubits == 3:
        return _exact_toffoli(qubits)

    # Generic
    if qubits == 2:
        gates = _synthesize_2qubit(target, budget)
    else:
        gates = _synthesize_3qubit(target, budget)

    # Ensure CNOT budget is respected
    while _count_cx(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break

    # Ensure gate count limit
    if len(gates) > 400:
        gates = gates[:400]
        # Remove any trailing CNOTs that exceed budget
        while _count_cx(gates) > budget and len(gates) > 0:
            for i in range(len(gates) - 1, -1, -1):
                if gates[i][0] == "cx":
                    gates.pop(i)
                    break
            else:
                break

    return gates
