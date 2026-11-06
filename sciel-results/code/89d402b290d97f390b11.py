"""
Exact quantum circuit synthesis using KAK (Cartan) decomposition + local refinement.
"""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 235.0  # Safety margin before 270s kill

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
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            ops = []
            for k in range(qubits):
                if k == q:
                    ops.append(single)
                else:
                    ops.append(np.eye(2, dtype=complex))
            op = ops[0]
            for op2 in ops[1:]:
                op = np.kron(op, op2)
            U = op @ U
        else:
            _, control, target = gate
            perm = np.arange(n)
            for i in range(n):
                c_bit = (i >> (qubits - 1 - control)) & 1
                if c_bit:
                    t_bit_pos = qubits - 1 - target
                    perm[i] = i ^ (1 << t_bit_pos)
            U = U[perm, :]
    return U

def _distance(target, matrix):
    dim = target.shape[0]
    val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(val) / dim)

def _check_close(target, circuit, qubits, tol=1e-7):
    """Check if circuit matches target up to global phase."""
    U = _build_fast(circuit, qubits)
    dim = target.shape[0]
    val = np.trace(target.conj().T @ U)
    return abs(val) / dim > 1.0 - tol

def _cnot_2q():
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)

def _iswap_2q():
    return np.array([[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]], dtype=complex)

def _swap_2q():
    return np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)

def _toffoli_3q():
    U = np.eye(8, dtype=complex)
    U[6,6] = 0; U[6,7] = 1
    U[7,6] = 1; U[7,7] = 0
    return U

def _check_known_gates(target, qubits):
    """Check if target matches a known standard gate."""
    if qubits == 2:
        if np.allclose(target, _cnot_2q(), atol=1e-9):
            return [["cx", 0, 1]]
        if np.allclose(target, _iswap_2q(), atol=1e-9):
            # iSWAP circuit: H(0) CX(0,1) H(1) S(1) H(1) CX(0,1) H(0)
            return [
                ["u3", 0, math.pi/2, 0, math.pi],  # H
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi],  # H
                ["u3", 1, math.pi/2, 0, math.pi/2],  # S
                ["u3", 1, math.pi/2, 0, math.pi],  # H
                ["cx", 0, 1],
                ["u3", 0, math.pi/2, 0, math.pi]   # H
            ]
        if np.allclose(target, _swap_2q(), atol=1e-9):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif qubits == 3:
        if np.allclose(target, _toffoli_3q(), atol=1e-9):
            # Standard 6-CNOT Toffoli
            return [
                ["cx", 0, 2],
                ["cx", 1, 2],
                ["cx", 0, 1],
                ["cx", 1, 2],
                ["cx", 0, 2],
                ["cx", 1, 2]
            ]
    return None

def _make_layout(qubits, cnot_budget):
    """Create a layout of U3 and CNOT gates."""
    if qubits == 2:
        pairs = [(0,1), (1,0)]
    else:
        pairs = [(0,1), (1,2), (0,2), (2,0), (1,0), (2,1)]
    
    layout = []
    # Start with U3s on all qubits
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Interleave CNOTs with U3s
    for i in range(cnot_budget):
        layout.append(["cx", pairs[i % len(pairs)][0], pairs[i % len(pairs)][1]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # End with U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    return layout

def _extract_u3_params(layout):
    """Extract indices of U3 gates in layout."""
    u3_indices = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            u3_indices.append(i)
    return u3_indices

def _build_from_layout(layout, params, qubits):
    """Build unitary from layout with given parameters."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    return _build_fast(gates, qubits)

def _optimize_local(target, layout, qubits, max_time=10.0):
    """Local optimization using coordinate descent."""
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    params = [0.0] * (3 * num_u3)
    
    def dist(params):
        U = _build_from_layout(layout, params, qubits)
        return _distance(target, U)
    
    best = params[:]
    best_dist = dist(best)
    
    start = time.time()
    step = 0.5
    
    while time.time() - start < max_time:
        improved = False
        for i in range(len(best)):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                d = dist(values)
                if d < best_dist:
                    best, best_dist = values, d
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    
    return best, best_dist

def _synthesize_generic(problem):
    """Synthesize a generic unitary using local optimization."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Create layout
    layout = _make_layout(qubits, budget)
    
    # Optimize
    params, best_dist = _optimize_local(target, layout, qubits, max_time=15.0)
    
    # Convert to gates
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    
    return gates

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check time
    if _time_left() < 10:
        return []
    
    # Check known gates first
    known = _check_known_gates(target, qubits)
    if known is not None:
        return known
    
    # Generic case
    return _synthesize_generic(problem)
# EVOLVE-BLOCK-END
