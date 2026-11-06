"""Exact decompositions for known gates + gradient-based synthesis for generic unitaries."""
import math
import time
import numpy as np

_DEADLINE = 250.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _cx_matrix(qubits, control, target):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    for i in range(dim):
        bit = (i >> (qubits - 1 - control)) & 1
        j = i ^ (bit << (qubits - 1 - target))
        mat[j, i] = 1.0
    return mat


def _build(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            eye2 = np.eye(2, dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            mat = _cx_matrix(qubits, control, target)
            matrix = mat @ matrix
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(problem, name=None):
    target = _target(problem)
    qubits = problem["qubits"]
    
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"
    
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6,6] = 0
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        toffoli[7,7] = 0
        if np.allclose(target, toffoli, atol=1e-10):
            return "toffoli"
    
    return None


def _decompose_cnot():
    return [_cx(0, 1)]


def _decompose_swap():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _decompose_iswap():
    S = math.pi / 2
    return [
        _u3(0, 0, 0, S),
        _u3(1, 0, 0, S),
        _cx(0, 1),
        _u3(0, 0, 0, -S),
        _u3(1, 0, 0, S),
        _cx(1, 0),
        _u3(0, 0, 0, S),
        _u3(1, 0, 0, -S),
    ]


def _decompose_toffoli():
    T = math.pi / 4
    H_theta = math.pi / 2
    H_phi = 0.0
    H_lam = math.pi
    return [
        _u3(2, H_theta, H_phi, H_lam),
        _cx(1, 2),
        _u3(2, 0, 0, -T),
        _cx(0, 2),
        _u3(1, 0, 0, T),
        _u3(2, 0, 0, -T),
        _cx(0, 2),
        _u3(2, 0, 0, T),
        _cx(1, 2),
        _u3(1, 0, 0, -T),
        _u3(2, 0, 0, T),
        _u3(2, H_theta, H_phi, H_lam),
    ]


def _gradient_optimize(problem, layout, max_iter=500):
    """Optimize single-qubit angles using gradient descent with analytic gradients."""
    qubits = problem["qubits"]
    target = _target(problem)
    dim = 2 ** qubits
    
    # Identify parameter slots
    param_slots = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_slots.append(i)
    
    n_params = len(param_slots)
    if n_params == 0:
        return layout, 0.0
    
    # Initialize parameters to zero
    params = np.zeros(n_params)
    
    # Pre-compute CNOT matrices for the layout
    cnot_mats = []
    for gate in layout:
        if gate[0] == "cx":
            cnot_mats.append(_cx_matrix(qubits, gate[1], gate[2]))
        else:
            cnot_mats.append(None)
    
    def build_with_params(params):
        matrix = np.eye(dim, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        for i, gate in enumerate(layout):
            if gate[0] == "u3":
                pi = param_slots.index(i)
                theta, phi, lam = params[3*pi], params[3*pi+1], params[3*pi+2]
                single = _u3_matrix(theta, phi, lam)
                operator = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    operator = np.kron(operator, single if k == gate[1] else eye2)
                matrix = operator @ matrix
            else:
                matrix = cnot_mats[i] @ matrix
        return matrix
    
    def fidelity(params):
        matrix = build_with_params(params)
        return abs(np.trace(target.conj().T @ matrix)) / dim
    
    def gradient(params):
        """Compute gradient of fidelity w.r.t. parameters using finite differences (fast enough for small systems)."""
        eps = 1e-4
        grad = np.zeros_like(params)
        base = fidelity(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            grad[i] = (fidelity(params_plus) - fidelity(params_minus)) / (2 * eps)
        return grad
    
    # Adam optimizer
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    lr = 0.01
    
    best_params = params.copy()
    best_fid = fidelity(params)
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        
        grad = -gradient(params)  # negative gradient to maximize fidelity
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**(it+1))
        v_hat = v / (1 - beta2**(it+1))
        params = params + lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        
        # Wrap angles to [-pi, pi]
        for i in range(len(params)):
            params[i] = ((params[i] + np.pi) % (2 * np.pi)) - np.pi
        
        fid = fidelity(params)
        if fid > best_fid:
            best_fid = fid
            best_params = params.copy()
        
        if best_fid > 1.0 - 1e-6:
            break
    
    # Build final gate list
    gates = []
    pi = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], float(best_params[pi]), float(best_params[pi+1]), float(best_params[pi+2])])
            pi += 3
        else:
            gates.append(list(gate))
    
    return gates, 1.0 - best_fid


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    
    # Try to identify known gates
    known = _is_known_gate(problem, name)
    
    if known == "cnot":
        gates = _decompose_cnot()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates
    
    if known == "swap":
        gates = _decompose_swap()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates
    
    if known == "iswap":
        gates = _decompose_iswap()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates
    
    if known == "toffoli":
        gates = _decompose_toffoli()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates
    
    # For generic unitaries, use gradient-based optimization
    target = _target(problem)
    
    # Build a CNOT layout that stays within budget
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Use a balanced layout: cycle through all CNOT pairs
    layout = []
    cnot_count = 0
    for layer in range(budget):
        # Add u3 gates before each CNOT
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
        cnot_count += 1
    # Add final u3 gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Check gate count
    if len(layout) > 400:
        # Reduce CNOTs
        while len(layout) > 400 and cnot_count > 0:
            cnot_count -= 1
            budget = cnot_count
            layout = []
            for layer in range(budget):
                for q in range(qubits):
                    layout.append(["u3", q, 0.0, 0.0, 0.0])
                layout.append(["cx", *pairs[layer % len(pairs)]])
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Optimize using gradient descent
    gates, dist = _gradient_optimize(problem, layout, max_iter=300)
    
    return gates
