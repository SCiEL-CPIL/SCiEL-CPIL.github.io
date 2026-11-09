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

def _optimal_u3(A):
    """Find (theta, phi, lam) such that u3(theta,phi,lam) best approximates A (2x2)."""
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    sqrt_det = np.sqrt(det_A)
    if abs(sqrt_det) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    c_val = A[0,0] / sqrt_det
    s_val_neg = A[0,1] / sqrt_det
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

def _apply_u3_to_layer(U, qubit, theta, phi, lam, qubits):
    n = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    op = _u3_matrix(theta, phi, lam)
    full = np.array([[1]], dtype=complex)
    for k in range(qubits):
        full = np.kron(full, op if k == qubit else eye2)
    return full @ U

def _apply_cx(U, control, target, qubits):
    n = 2 ** qubits
    idx = np.arange(n)
    cb = (idx >> (qubits - 1 - control)) & 1
    perm = idx ^ (cb << (qubits - 1 - target))
    return U[perm, :]

def _build_from_params(params, cnot_template, qubits):
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    U = np.eye(n, dtype=complex)
    param_idx = 0
    eye2 = np.eye(2, dtype=complex)
    for layer in range(num_layers):
        for q in range(qubits):
            th = params[param_idx]
            ph = params[param_idx+1]
            lm = params[param_idx+2]
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

def _cost_fn(params, target, cnot_template, qubits):
    n = 2 ** qubits
    U = _build_from_params(params, cnot_template, qubits)
    tr_val = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(tr_val) / n)

def _grad_fn(params, target, cnot_template, qubits):
    n = 2 ** qubits
    num_params = len(params)
    U = _build_from_params(params, cnot_template, qubits)
    tr_val = np.trace(target.conj().T @ U)
    cost = 1.0 - min(1.0, abs(tr_val) / n)
    
    if abs(tr_val) < 1e-12:
        return cost, np.zeros(num_params)
    
    grad = np.zeros(num_params)
    eps = 1e-6
    for k in range(num_params):
        p_plus = params.copy()
        p_plus[k] += eps
        U_plus = _build_from_params(p_plus, cnot_template, qubits)
        tr_plus = np.trace(target.conj().T @ U_plus)
        dtr = (tr_plus - tr_val) / eps
        grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
    grad = -grad / n
    return cost, grad

def _optimize_circuit(target, qubits, cnot_template, max_iter=200):
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * qubits
    num_params = num_u3 * 3
    
    params = np.zeros(num_params, dtype=float)
    
    # Analytic initialization
    U_init = _build_from_params(params, cnot_template, qubits)
    try:
        U_inv = np.linalg.inv(U_init)
        residual = target @ U_inv
        param_idx = 0
        for layer in range(num_layers):
            for q in range(qubits):
                if qubits == 2:
                    if q == 0:
                        block = residual[0:2, 0:2]
                    else:
                        block = residual[2:4, 2:4]
                else:
                    block = np.eye(2, dtype=complex)
                
                th, ph, lm = _optimal_u3(block)
                params[param_idx] = th
                params[param_idx+1] = ph
                params[param_idx+2] = lm
                param_idx += 3
    except:
        pass
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.1
    best_cost = _cost_fn(params, target, cnot_template, qubits)
    best_params = params.copy()
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        cost, grad = _grad_fn(params, target, cnot_template, qubits)
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        if best_cost < 1e-10:
            break
        t_adam += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t_adam)
        v_hat = v / (1 - beta2**t_adam)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        if it % 30 == 29 and it > 0:
            lr *= 0.85
    
    return best_params, best_cost

def _params_to_gates(params, cnot_template, qubits):
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    gates = []
    param_idx = 0
    for layer in range(num_layers):
        for q in range(qubits):
            th = params[param_idx]
            ph = params[param_idx+1]
            lm = params[param_idx+2]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
            param_idx += 3
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
    return gates

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

    if qubits == 2:
        num_cx = min(budget, 3)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,0), (0,1), (1,0), (0,1), (1,0)]
        cnot_template = pairs[:num_cx]
    else:
        num_cx = min(budget, 6)
        if num_cx < 1:
            num_cx = 1
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        cnot_template = pairs[:num_cx]

    max_iter = 200 if qubits == 2 else 150
    params, cost = _optimize_circuit(target, qubits, cnot_template, max_iter=max_iter)
    gates = _params_to_gates(params, cnot_template, qubits)

    while _count_cx(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break

    return gates
