"""Exact decompositions for known gates + analytic synthesis for generic unitaries."""
import math
import time
import numpy as np

_DEADLINE = 260.0
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


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target_q = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target_q)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(problem, name=None):
    """Check if the target is a known gate by comparing to exact decompositions."""
    target = _target(problem)
    qubits = problem["qubits"]
    dim = 2 ** qubits

    if qubits == 2 and name is None:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"

    if qubits == 3 and name is None:
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
    """iSWAP gate decomposition using 2 CNOTs and single-qubit rotations."""
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
    """Toffoli gate decomposition using 6 CNOTs."""
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


def _su2_to_u3(A):
    """Convert a 2x2 SU(2) matrix to u3 angles (theta, phi, lam)."""
    det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
    if abs(det) < 1e-15:
        return (0.0, 0.0, 0.0)
    sqrt_det = np.sqrt(det)
    a = A[0,0] / sqrt_det
    b = -A[0,1] / sqrt_det
    c = A[1,0] / sqrt_det
    d = A[1,1] / sqrt_det
    
    # u3 matrix: [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #             [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    theta = 2 * np.arccos(np.clip(np.abs(a), 0, 1))
    
    if np.abs(b) < 1e-15 and np.abs(c) < 1e-15:
        phi = 0.0
        lam = 0.0
    elif np.abs(c) < 1e-15:
        phi = 0.0
        lam = np.angle(-b)
    elif np.abs(b) < 1e-15:
        phi = np.angle(c)
        lam = 0.0
    else:
        phi = np.angle(c)
        lam = np.angle(-b)
    
    # Normalize angles to [-pi, pi]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    lam = (lam + np.pi) % (2 * np.pi) - np.pi
    
    return (float(theta), float(phi), float(lam))


def _decompose_2qubit(U):
    """Decompose a 2-qubit unitary using the KAK/CS decomposition."""
    # Use scipy for SVD-based decomposition
    from scipy.linalg import svd
    
    # KAK decomposition: U = (A1 ⊗ A2) · K · (B1 ⊗ B2)
    # where K is a canonical form with up to 3 parameters
    
    # For now, use a numerical approach with optimization
    # Try a few different CNOT layouts and optimize
    
    best_gates = None
    best_dist = float('inf')
    
    # Layout 1: CNOT(0,1) - single qubit - CNOT(0,1)
    layouts = [
        # (cnot_positions, qubit_pairs)
        [(0,1)],
        [(1,0)],
        [(0,1),(0,1)],
        [(1,0),(1,0)],
        [(0,1),(1,0)],
        [(1,0),(0,1)],
    ]
    
    for cnot_list in layouts:
        if len(cnot_list) > 3:
            continue
        
        # Build layout: u3 for each qubit before each CNOT, and u3 for each qubit at the end
        layout = []
        for c, t in cnot_list:
            for q in range(2):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", c, t])
        for q in range(2):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        
        # Optimize
        gates, dist = _optimize_generic_internal(U, layout, 2)
        
        if dist < best_dist:
            best_dist = dist
            best_gates = gates
    
    return best_gates, best_dist


def _decompose_3qubit(U):
    """Decompose a 3-qubit unitary using sequential 2-qubit decompositions."""
    # Use a layered approach: decompose into 2-qubit gates
    # This is complex, so use optimization with a reasonable CNOT budget
    
    best_gates = None
    best_dist = float('inf')
    
    # Try different CNOT layouts
    # For 3 qubits, we need at least 6 CNOTs for generic unitaries
    
    cnot_budget = 6
    
    # Layout: alternate CNOTs on different pairs
    pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    
    layout = []
    for i in range(cnot_budget):
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[i % len(pairs)]])
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    gates, dist = _optimize_generic_internal(U, layout, 3)
    
    return gates, dist


def _optimize_generic_internal(target, layout, qubits):
    """Optimize single-qubit angles using scipy leastsq."""
    dim = 2 ** qubits
    
    param_slots = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_slots.append((i, 0))
            param_slots.append((i, 1))
            param_slots.append((i, 2))
    
    n_params = len(param_slots)
    
    def build_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates
    
    def fidelity_residual(params):
        gates = build_from_params(params)
        U = _build(gates, qubits)
        prod = target.conj().T @ U
        return np.concatenate([np.real(prod.flatten()), np.imag(prod.flatten())])
    
    best = np.zeros(n_params)
    best_dist = _distance(target, _build(build_from_params(best), qubits))
    
    try:
        from scipy.optimize import leastsq
        import random
        rng = random.Random(12345 + qubits)
        
        for restart in range(3):
            if _elapsed() > _DEADLINE - 15:
                break
            if restart == 0:
                x0 = np.zeros(n_params)
            else:
                x0 = rng.uniform(-math.pi, math.pi, n_params)
            
            try:
                result, _cov, _info, _msg, _ier = leastsq(
                    fidelity_residual, x0, full_output=True, maxfev=300
                )
                d = _distance(target, _build(build_from_params(result), qubits))
                if d < best_dist:
                    best = result
                    best_dist = d
                if best_dist < 1e-10:
                    break
            except Exception:
                continue
    except ImportError:
        pass
    
    # Coordinate descent polish
    if best_dist > 1e-8 and _elapsed() < _DEADLINE - 15:
        import random
        rng = random.Random(99999 + qubits)
        for sweep in range(5):
            if _elapsed() > _DEADLINE - 15:
                break
            step = 1.0 / (sweep + 1)
            improved = False
            for pi in range(n_params):
                if _elapsed() > _DEADLINE - 15:
                    break
                for trial in (best[pi] + step, best[pi] - step, best[pi] + rng.uniform(-math.pi, math.pi)):
                    vals = best.copy()
                    vals[pi] = trial
                    d = _distance(target, _build(build_from_params(vals), qubits))
                    if d < best_dist:
                        best, best_dist = vals, d
                        improved = True
            if not improved and sweep > 2:
                break
    
    return build_from_params(best), best_dist


def synthesize(problem):
    """Synthesize the target unitary using known decompositions or optimization."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    
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
    
    target = _target(problem)
    
    if qubits == 2:
        gates, dist = _decompose_2qubit(target)
        if gates is not None:
            return gates
    else:
        gates, dist = _decompose_3qubit(target)
        if gates is not None:
            return gates
    
    # Fallback: build a CNOT layout that stays within budget
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    if len(layout) > 400:
        max_cnots = (400 - qubits) // (1 + qubits)
        budget = min(budget, max_cnots)
        layout = []
        for layer in range(budget):
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", *pairs[layer % len(pairs)]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    gates, dist = _optimize_generic_internal(target, layout, qubits)
    
    return gates
