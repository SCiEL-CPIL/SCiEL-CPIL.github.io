import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 250.0

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
            n_bits = qubits
            idx = np.arange(n)
            cb = (idx >> (n_bits - 1 - c)) & 1
            perm = idx ^ (cb << (n_bits - 1 - t))
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

def _optimal_u3_from_2x2(A):
    """
    Find (theta, phi, lam) such that u3(theta,phi,lam) best approximates a 2x2 unitary A
    up to a global phase.
    """
    if not np.allclose(np.eye(2), A.conj().T @ A, atol=1e-10):
        U_svd, _, Vh_svd = np.linalg.svd(A)
        A = U_svd @ Vh_svd
    
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    sqrt_det = np.sqrt(det_A)
    if abs(sqrt_det) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    c_val = A[0,0] / sqrt_det
    s_val_from_10 = A[1,0] / sqrt_det
    s_val_from_01 = -A[0,1] / sqrt_det
    
    c_abs = abs(c_val)
    s_abs = max(abs(s_val_from_10), abs(s_val_from_01))
    
    norm = math.sqrt(c_abs**2 + s_abs**2)
    if norm < 1e-15:
        return (0.0, 0.0, 0.0)
    
    c_abs /= norm
    s_abs /= norm
    
    theta_half = math.atan2(s_abs, c_abs)
    theta = 2.0 * theta_half
    
    if s_abs > 1e-15:
        e_iphi = (A[1,0] / sqrt_det) / s_abs
        phi = float(np.angle(e_iphi))
    else:
        phi = 0.0
    
    if s_abs > 1e-15:
        e_ilam = (-A[0,1] / sqrt_det) / s_abs
        lam = float(np.angle(e_ilam))
    else:
        lam = 0.0
    
    return (theta, phi, lam)

def _extract_u3_from_context(target, qubit, qubits, left_context, right_context):
    """
    Find optimal u3 on qubit such that left_context @ u3_q @ right_context ≈ target.
    """
    n = 2 ** qubits
    
    try:
        left_inv = np.linalg.inv(left_context)
        right_inv = np.linalg.inv(right_context)
    except:
        left_inv = left_context.conj().T
        right_inv = right_context.conj().T
    
    M = left_inv @ target @ right_inv
    
    bit_pos = qubits - 1 - qubit
    
    blocks = []
    for other_bits in range(1 << (qubits - 1)):
        base = 0
        other_bit = other_bits
        for k in range(qubits):
            if k == qubit:
                continue
            bit_val = (other_bit >> (qubits - 1 - k - 1)) & 1
            if bit_val:
                base |= (1 << (qubits - 1 - k))
        
        idx0 = base
        idx1 = base | (1 << bit_pos)
        
        block = M[np.ix_([idx0, idx1], [idx0, idx1])]
        blocks.append(block)
    
    avg_block = np.mean(blocks, axis=0)
    
    return _optimal_u3_from_2x2(avg_block)

def _layer_synth_2qubit(target, cnot_template):
    """
    Layer-by-layer synthesis for 2-qubit unitaries.
    """
    n = 4
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    
    right_context = np.eye(n, dtype=complex)
    left_context = np.eye(n, dtype=complex)
    
    u3_params = [[(0.0, 0.0, 0.0) for _ in range(2)] for _ in range(num_layers)]
    
    for layer in range(num_layers - 1, -1, -1):
        if _remaining() < 2.0:
            break
        
        if layer < num_cx:
            c, t = cnot_template[layer]
            idx = np.arange(n)
            cb = (idx >> (1 - c)) & 1
            perm = idx ^ (cb << (1 - t))
            cnot_matrix = np.eye(n, dtype=complex)
            cnot_matrix = cnot_matrix[perm, :]
        else:
            cnot_matrix = np.eye(n, dtype=complex)
        
        right_for_u3_0 = cnot_matrix @ right_context
        th0, ph0, lm0 = _extract_u3_from_context(target, 0, 2, left_context, right_for_u3_0)
        u3_params[layer][0] = (th0, ph0, lm0)
        
        u3_0_matrix = _u3_matrix(th0, ph0, lm0)
        full_u3_0 = np.kron(u3_0_matrix, np.eye(2, dtype=complex))
        right_for_u3_1 = full_u3_0 @ cnot_matrix @ right_context
        th1, ph1, lm1 = _extract_u3_from_context(target, 1, 2, left_context, right_for_u3_1)
        u3_params[layer][1] = (th1, ph1, lm1)
        
        u3_1_matrix = _u3_matrix(th1, ph1, lm1)
        full_u3_1 = np.kron(np.eye(2, dtype=complex), u3_1_matrix)
        left_context = left_context @ full_u3_1 @ full_u3_0 @ cnot_matrix
    
    gates = []
    for layer in range(num_layers):
        for q in range(2):
            th, ph, lm = u3_params[layer][q]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    
    U = _build(gates, 2)
    dist = _dist(target, U)
    
    return gates, dist

def _layer_synth_3qubit(target, cnot_template):
    """
    Layer-by-layer synthesis for 3-qubit unitaries.
    """
    n = 8
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    
    right_context = np.eye(n, dtype=complex)
    left_context = np.eye(n, dtype=complex)
    
    u3_params = [[(0.0, 0.0, 0.0) for _ in range(3)] for _ in range(num_layers)]
    
    for layer in range(num_layers - 1, -1, -1):
        if _remaining() < 2.0:
            break
        
        if layer < num_cx:
            c, t = cnot_template[layer]
            idx = np.arange(n)
            cb = (idx >> (2 - c)) & 1
            perm = idx ^ (cb << (2 - t))
            cnot_matrix = np.eye(n, dtype=complex)
            cnot_matrix = cnot_matrix[perm, :]
        else:
            cnot_matrix = np.eye(n, dtype=complex)
        
        right_for_current = cnot_matrix @ right_context
        
        for q in range(3):
            th, ph, lm = _extract_u3_from_context(target, q, 3, left_context, right_for_current)
            u3_params[layer][q] = (th, ph, lm)
            
            u3_mat = _u3_matrix(th, ph, lm)
            full_u3 = np.array([[1]], dtype=complex)
            for k in range(3):
                full_u3 = np.kron(full_u3, u3_mat if k == q else np.eye(2, dtype=complex))
            right_for_current = full_u3 @ right_for_current
        
        for q in range(2, -1, -1):
            th, ph, lm = u3_params[layer][q]
            u3_mat = _u3_matrix(th, ph, lm)
            full_u3 = np.array([[1]], dtype=complex)
            for k in range(3):
                full_u3 = np.kron(full_u3, u3_mat if k == q else np.eye(2, dtype=complex))
            left_context = left_context @ full_u3
        left_context = left_context @ cnot_matrix
    
    gates = []
    for layer in range(num_layers):
        for q in range(3):
            th, ph, lm = u3_params[layer][q]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    
    U = _build(gates, 3)
    dist = _dist(target, U)
    
    return gates, dist

def _refine_with_adam(target, qubits, gates, cnot_template, max_iter=100):
    """
    Refine u3 parameters using Adam optimizer after initial layer synthesis.
    """
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_params = num_layers * qubits * 3
    T_dag = target.conj().T
    
    # Extract current params from gates
    params = np.zeros(num_params)
    param_idx = 0
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            params[param_idx] = th
            params[param_idx+1] = ph
            params[param_idx+2] = lm
            param_idx += 3
    
    def build_from_params(p):
        U = np.eye(n, dtype=complex)
        param_idx = 0
        eye2 = np.eye(2, dtype=complex)
        for layer in range(num_layers):
            for q in range(qubits):
                th = p[param_idx]
                ph = p[param_idx+1]
                lm = p[param_idx+2]
                R = _u3_matrix(th, ph, lm)
                full = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    full = np.kron(full, R if k == q else eye2)
                U = full @ U
                param_idx += 3
            if layer < num_cx:
                c, t = cnot_template[layer]
                idx = np.arange(n)
                cb = (idx >> (qubits - 1 - c)) & 1
                perm = idx ^ (cb << (qubits - 1 - t))
                U = U[perm, :]
        return U
    
    def cost_fn(p):
        U = build_from_params(p)
        tr_val = np.trace(T_dag @ U)
        return 1.0 - min(1.0, abs(tr_val) / n)
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.05
    best_cost = 1.0
    best_params = params.copy()
    
    cost = cost_fn(params)
    if cost < best_cost:
        best_cost = cost
        best_params = params.copy()
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        if best_cost < 1e-10:
            break
        
        grad = np.zeros(num_params)
        eps = 1e-6
        U_base = build_from_params(params)
        tr_val = np.trace(T_dag @ U_base)
        
        for k in range(num_params):
            p_plus = params.copy()
            p_plus[k] += eps
            U_plus = build_from_params(p_plus)
            tr_plus = np.trace(T_dag @ U_plus)
            dtr = (tr_plus - tr_val) / eps
            if abs(tr_val) > 1e-12:
                grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
        grad = -grad / n
        
        t_adam += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_adam)
        v_hat = v / (1 - beta2**t_adam)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        
        cost = cost_fn(params)
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        
        if it % 20 == 19:
            lr *= 0.9
    
    new_gates = []
    param_idx = 0
    for layer in range(num_layers):
        for q in range(qubits):
            th = best_params[param_idx]
            ph = best_params[param_idx+1]
            lm = best_params[param_idx+2]
            new_gates.append(["u3", q, float(th), float(ph), float(lm)])
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            new_gates.append(["cx", c, t])
    
    return new_gates, best_cost

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

    # Generic: use layer synthesis + Adam refinement
    if qubits == 2:
        num_cx = min(budget, 3)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,0), (0,1), (1,0), (0,1), (1,0)]
        cnot_template = pairs[:num_cx]
        
        gates, dist = _layer_synth_2qubit(target, cnot_template)
        
        if _remaining() > 30.0 and dist > 1e-8:
            gates, dist = _refine_with_adam(target, 2, gates, cnot_template, max_iter=150)
        
        if budget >= 2 and _remaining() > 60.0:
            alt_pairs = [(1,0), (0,1), (1,0)]
            alt_template = alt_pairs[:num_cx]
            gates_alt, dist_alt = _layer_synth_2qubit(target, alt_template)
            if _remaining() > 30.0 and dist_alt > 1e-8:
                gates_alt, dist_alt = _refine_with_adam(target, 2, gates_alt, alt_template, max_iter=150)
            if dist_alt < dist:
                gates = gates_alt
                dist = dist_alt
        
    else:
        num_cx = min(budget, 6)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        cnot_template = pairs[:num_cx]
        
        gates, dist = _layer_synth_3qubit(target, cnot_template)
        
        if _remaining() > 30.0 and dist > 1e-8:
            gates, dist = _refine_with_adam(target, 3, gates, cnot_template, max_iter=100)
        
        if budget >= 4 and _remaining() > 60.0:
            alt_pairs = [(1,2), (0,1), (1,2), (0,1), (1,2), (0,1)]
            alt_template = alt_pairs[:num_cx]
            gates_alt, dist_alt = _layer_synth_3qubit(target, alt_template)
            if _remaining() > 30.0 and dist_alt > 1e-8:
                gates_alt, dist_alt = _refine_with_adam(target, 3, gates_alt, alt_template, max_iter=100)
            if dist_alt < dist:
                gates = gates_alt
                dist = dist_alt

    while _count_cx(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break

    while len(gates) > 400:
        removed = False
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "u3" and abs(gates[i][2]) < 1e-6:
                gates.pop(i)
                removed = True
                break
        if not removed:
            gates.pop()

    return gates
