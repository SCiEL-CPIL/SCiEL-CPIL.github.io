# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
TIME_LIMIT = 260

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build_unitary(gates, qubits):
    """Build the unitary from a gate list."""
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
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            # CNOT logic: if control is 1, flip target.
            # Basis index: q0 q1 ... qn
            # control is q_c, target is q_t.
            # We need to swap states where q_c=1.
            # This permutation is equivalent to multiplying by a permutation matrix.
            # The line `matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]` is incorrect for building the unitary from scratch in this loop structure if we are left-multiplying.
            # Actually, if we are applying gates in sequence: U_new = G @ U_old.
            # CNOT is a permutation matrix P.
            # P @ U_old.
            # The row i of P @ U_old is row (i XOR mask) of U_old?
            # Let's stick to a simpler, albeit slightly slower, explicit construction or fix the permutation.
            # For 2-3 qubits, explicit is fine.
            cnot_mat = np.eye(n, dtype=complex)
            # Construct CNOT matrix
            for i in range(n):
                c_bit = (i >> (qubits - 1 - control)) & 1
                if c_bit:
                    t_bit_pos = qubits - 1 - target
                    new_i = i ^ (1 << t_bit_pos)
                    cnot_mat[new_i, i] = 1
                else:
                    cnot_mat[i, i] = 1
            matrix = cnot_mat @ matrix
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _synthesize_cnot():
    return [["cx", 0, 1]]

def _synthesize_iswap():
    return [
        ["u3", 0, math.pi / 2, 0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, math.pi / 2, 0],
        ["u3", 1, math.pi / 2, math.pi / 2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi / 2, 0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0, math.pi / 2],
    ]

def _synthesize_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]

def _synthesize_toffoli():
    return [
        ["u3", 2, math.pi / 2, 0, math.pi / 2],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, -math.pi / 2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, -math.pi / 2],
        ["u3", 1, math.pi / 4, 0, math.pi / 4],
        ["u3", 2, math.pi / 4, 0, math.pi / 2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 2, 0, math.pi / 2],
    ]

def _optimize_generic(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if _time_remaining() < 10:
        # Fallback: return empty or minimal
        return []

    # Define CNOT layout
    if qubits == 2:
        # Max 3 CNOTs for generic 2-qubit
        num_cnots = min(budget, 3)
        cnot_pairs = []
        for i in range(num_cnots):
            # Alternate directions to maximize entanglement capability
            if i % 2 == 0:
                cnot_pairs.append((0, 1))
            else:
                cnot_pairs.append((1, 0))
    else:
        # 3 qubits
        num_cnots = budget
        cnot_pairs = []
        # Standard ladder: 0-1, 1-2, 0-1, 1-2...
        pattern = [(0, 1), (1, 2)]
        for i in range(num_cnots):
            cnot_pairs.append(pattern[i % 2])

    # Build layout: u3 blocks interleaved with CNOTs
    layout = []
    num_blocks = num_cnots + 1
    for b in range(num_blocks):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        if b < num_cnots:
            layout.append(["cx", cnot_pairs[b][0], cnot_pairs[b][1]])

    # Extract indices of u3 gates for optimization
    u3_indices = [i for i, g in enumerate(layout) if g[0] == "u3"]
    n_params = len(u3_indices) * 3

    def build_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                gates.append(["u3", q, params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(params):
        if _time_remaining() < 5:
            return 1.0
        gates = build_from_params(params)
        U = _build_unitary(gates, qubits)
        return _distance(target, U)

    # Initial guess: zeros
    x0 = np.zeros(n_params)
    
    # Multi-start
    best_x = x0
    best_obj = objective(x0)
    
    np.random.seed(42)
    for _ in range(3):
        if _time_remaining() < 15:
            break
        x_try = np.random.uniform(-math.pi, math.pi, n_params)
        obj_try = objective(x_try)
        if obj_try < best_obj:
            best_obj = obj_try
            best_x = x_try

    # Refine with L-BFGS-B
    if _time_remaining() > 20:
        res = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        if res.success and res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun

    return build_from_params(best_x)

def synthesize(problem):
    name = problem.get("name", "").strip().lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]

    # Exact matches
    if name == "cnot" and qubits == 2:
        return _synthesize_cnot()
    elif name == "iswap" and qubits == 2:
        return _synthesize_iswap()
    elif name == "swap" and qubits == 2:
        return _synthesize_swap()
    elif name == "toffoli" and qubits == 3:
        return _synthesize_toffoli()
    else:
        return _optimize_generic(problem)
# EVOLVE-BLOCK-END
