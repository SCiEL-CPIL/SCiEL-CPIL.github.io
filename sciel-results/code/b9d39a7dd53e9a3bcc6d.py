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
    # iSWAP exact decomposition
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

def _compute_circuit(params, cnot_template, qubits):
    """Compute the circuit unitary from params and fixed CNOT template."""
    n = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    U = np.eye(n, dtype=complex)
    param_idx = 0
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
            CX_full = np.zeros((n, n), dtype=complex)
            for i in range(n):
                CX_full[i, perm[i]] = 1
            U = CX_full @ U
    return U

def _optimize_circuit(target, qubits, cnot_template, max_iter=300):
    """
    Optimize u3 parameters for a fixed CNOT template using Adam gradient descent.
    Uses numerical gradients (forward differences).
    """
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3 = num_layers * qubits
    num_params = num_u3 * 3
    T_dag = target.conj().T

    np.random.seed(42)
    params = np.zeros(num_params, dtype=float)
    for i in range(num_u3):
        params[3*i] = np.random.uniform(0, math.pi)
        params[3*i+1] = np.random.uniform(-math.pi, math.pi)
        params[3*i+2] = np.random.uniform(-math.pi, math.pi)

    def cost_fn(p):
        U = _compute_circuit(p, cnot_template, qubits)
        tr_val = np.trace(T_dag @ U)
        return 1.0 - min(1.0, abs(tr_val) / n), tr_val

    def grad_fn(p):
        U = _compute_circuit(p, cnot_template, qubits)
        tr_val = np.trace(T_dag @ U)
        cost = 1.0 - min(1.0, abs(tr_val) / n)
        if abs(tr_val) < 1e-12:
            return cost, np.zeros(num_params)
        grad = np.zeros(num_params)
        eps = 1e-5
        for k in range(num_params):
            p_plus = p.copy()
            p_plus[k] += eps
            U_plus = _compute_circuit(p_plus, cnot_template, qubits)
            tr_plus = np.trace(T_dag @ U_plus)
            dtr = (tr_plus - tr_val) / eps
            # d|tr|/dp = Re(conj(tr) * dtr) / |tr|
            grad[k] = np.real(np.conj(tr_val) * dtr) / abs(tr_val)
        # We minimize cost = 1 - |tr|/n, so dcost/dp = -grad/n
        grad = -grad / n
        return cost, grad

    # Adam
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    t_adam = 0
    lr = 0.05
    best_cost = 1.0
    best_params = params.copy()

    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        cost, grad = grad_fn(params)
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
        if it % 50 == 49 and it > 0:
            lr *= 0.8

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

    # Generic: gradient-based optimization with fixed CNOT template
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

    max_iter = 300 if qubits == 2 else 200
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
