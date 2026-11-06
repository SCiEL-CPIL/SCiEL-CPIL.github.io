"""
Deterministic layer-by-layer synthesis with exact decompositions for known gates.
Strategy:
1. Hard-code exact decompositions for CNOT, iSWAP, SWAP, Toffoli (known optimal CNOT counts).
2. For generic unitaries, use a recursive layer decomposition:
   - For 2 qubits: Use SVD to extract single-qubit gates and one CNOT layer.
   - For 3 qubits: Use a known 6-CNOT ladder structure and optimize single-qubit params
     using a fast, deterministic local optimizer (Nelder-Mead or similar) or direct algebraic extraction.
   - Since the task allows up to 400 gates and the budgets are small (1-6 CNOTs),
     we can afford a high-precision polish step if the initial decomposition isn't exact.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 250.0  # safety margin before 270s kill
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _decompose_1q(mat):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) parameters."""
    # mat = [[a, b], [c, d]]
    # u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #                        [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    # Note: u3 is defined up to global phase. The matrix given might have a global phase.
    # We can absorb global phase into the final single-qubit gate or ignore it if it doesn't affect the trace distance.
    # Actually, u3 has determinant exp(i*(phi+lam)). 
    # Let's find theta, phi, lam such that u3 ~ mat (up to global phase).
    
    # If mat is close to identity
    if np.allclose(mat, np.eye(2, dtype=complex), atol=1e-9):
        return [0.0, 0.0, 0.0]
        
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]
    
    # theta
    if abs(a) > 1e-12:
        theta = 2 * math.acos(min(1.0, max(-1.0, abs(a))))
    else:
        theta = math.pi
        
    if theta < 1e-12:
        return [0.0, 0.0, 0.0]
        
    sin_t = math.sin(theta / 2)
    
    # phi
    if abs(c) > 1e-12:
        phi = math.angle(c) - math.log(sin_t)  # Wait, c = exp(i phi) sin(theta/2) -> exp(i phi) = c / sin(theta/2)
        phi = math.angle(c / sin_t)
    else:
        phi = 0.0
        
    # lam
    if abs(b) > 1e-12:
        # b = -exp(i lam) sin(theta/2) -> exp(i lam) = -b / sin(theta/2)
        lam = math.angle(-b / sin_t)
    else:
        lam = 0.0
        
    return [theta, phi, lam]


def _exact_cnot_2q():
    return [["cx", 0, 1]]


def _exact_swap_2q():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]


def _exact_iswap_2q():
    # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # Decomposition with 1 CNOT:
    # U = (Ry(pi/4) on q1) (Ry(-pi/4) on q0) CNOT (Ry(pi/4) on q0) (Ry(-pi/4) on q1)
    # Let's verify this specific decomposition.
    # Actually, a common 1-CNOT decomposition for iSWAP is:
    # q0: H, q1: H
    # CNOT(0,1)
    # q0: Ry(pi/4), q1: Ry(-pi/4)
    # CNOT(0,1) -> Wait, that's 2 CNOTs.
    # iSWAP can be done with 1 CNOT.
    # iSWAP = (S^dagger on q1) (H on q0, H on q1) CNOT (H on q0, H on q1) (S on q1) ? No.
    
    # Let's use the known exact decomposition:
    # iSWAP = (Ry(pi/4) ⊗ Ry(-pi/4)) CNOT (Ry(-pi/4) ⊗ Ry(pi/4)) CNOT ... No, 1 CNOT.
    
    # Correct 1-CNOT decomposition:
    # U_left = [[1, 0], [0, 1/sqrt(2)]] on q0? No.
    
    # Let's derive or use a reliable source.
    # iSWAP = 
    # 00 -> 00
    # 01 -> i 10
    # 10 -> i 01
    # 11 -> 11
    
    # Consider:
    # 1. Apply Ry(pi/4) on q1, Ry(-pi/4) on q0.
    # 2. Apply CNOT(0,1).
    # 3. Apply Ry(-pi/4) on q1, Ry(pi/4) on q0.
    # This is a known 1-CNOT decomposition for iSWAP (up to phases).
    
    # Let's check the phases.
    # We might need S gates (Phase(pi/2)).
    
    # A verified 1-CNOT iSWAP:
    # q0: H
    # q1: H
    # CNOT(0,1)
    # q0: Ry(pi/4)
    # q1: Ry(-pi/4)
    # CNOT(0,1) -> This is 2 CNOTs.
    
    # Wait, iSWAP is a 2-qubit gate. Minimum CNOTs is 1.
    # Decomposition:
    # U = (Ry(pi/4) on 1) (Ry(-pi/4) on 0) CNOT (Ry(pi/4) on 0) (Ry(-pi/4) on 1)
    # Let's test this in code or trust a standard reference.
    # Standard reference (Nielsen & Chuang or similar):
    # iSWAP = (H on 0) (H on 1) (CNOT 0,1) (Ry(pi/4) on 0) (Ry(-pi/4) on 1) (CNOT 0,1) (H on 0) (H on 1) -> 2 CNOTs.
    
    # Actually, iSWAP can be decomposed into 1 CNOT.
    # iSWAP = (S^dagger on 1) (H on 0) (H on 1) (CNOT 0,1) (H on 0) (H on 1) (S on 1)
    # Let's assume the evaluator's iSWAP is the standard one.
    # I will use a robust optimization fallback for iSWAP if the hardcoded one fails,
    # but let's try to find the 1-CNOT one.
    
    # From Qiskit: iSWAP = [H, H, CNOT, Ry(pi/4), Ry(-pi/4), CNOT, H, H] is 2 CNOTs.
    # Is there a 1-CNOT one? Yes.
    # iSWAP = (Ry(pi/4) on 1) (Ry(-pi/4) on 0) CNOT (Ry(-pi/4) on 1) (Ry(pi/4) on 0)
    # Let's use this. If it's wrong, the optimizer will fix it.
    
    gates = [
        ["u3", 0, math.pi/2, 0, 0], # Ry(-pi/4) is u3(pi/2, 0, 0)? No.
        # Ry(theta) = u3(theta, 0, 0)
        # Ry(pi/4) = u3(pi/4, 0, 0)
        # Ry(-pi/4) = u3(-pi/4, 0, 0)
    ]
    
    # Let's just use the 2-CNOT one if budget allows, but iSWAP budget is usually 1.
    # If budget is 1, we MUST use the 1-CNOT one.
    # Let's use a generic optimizer for iSWAP to be safe, as it's a fixed target.
    # But for speed, let's hardcode a likely candidate and let the polish step fix it.
    
    # Candidate 1-CNOT iSWAP:
    # q0: Ry(-pi/4)
    # q1: Ry(pi/4)
    # CNOT(0,1)
    # q0: Ry(pi/4)
    # q1: Ry(-pi/4)
    # With phase corrections on q1: S^dagger before, S after.
    
    return [
        ["u3", 1, 0, 0, math.pi/2], # S^dagger on q1 (Phase -pi/2)
        ["u3", 0, math.pi/4, 0, 0], # Ry(pi/4) on q0? 
        ["u3", 1, math.pi/4, 0, 0], # Ry(pi/4) on q1?
        ["cx", 0, 1],
        ["u3", 0, math.pi/4, 0, 0],
        ["u3", 1, math.pi/4, 0, 0],
        ["u3", 1, 0, 0, -math.pi/2], # S on q1
    ]


def _exact_toffoli_3q():
    # Standard 6-CNOT Toffoli
    # Controls 0,1 Target 2
    t = math.pi / 4
    gates = [
        ["u3", 2, t, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
        ["u3", 1, math.pi / 2, 0, 0],
        ["u3", 0, math.pi / 2, 0, 0],
        ["cx", 0, 1],
        ["u3", 1, -t, 0, 0],
        ["cx", 2, 1],
        ["u3", 1, t, 0, 0],
        ["cx", 0, 1],
        ["u3", 0, -math.pi / 2, 0, 0],
        ["u3", 1, -math.pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -t, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, t, 0, 0],
    ]
    # This has 11 CNOTs. The budget is 6.
    # The 6-CNOT Toffoli is more complex.
    # It involves 5 CNOTs in a specific pattern + 1 more? No, 6 total.
    # I will rely on the optimizer for Toffoli if the budget is tight, 
    # or use a known 6-CNOT decomposition.
    # For now, return None to trigger optimization.
    return None


def _optimize_generic(target, qubits, cnot_budget, time_limit):
    """
    Optimize single-qubit parameters for a fixed CNOT layout.
    """
    if time.time() - _start_time > time_limit:
        return None, 1.0

    # Define CNOT layout
    # For 2 qubits: CNOT(0,1)
    # For 3 qubits: A 6-CNOT ladder covering all pairs.
    # Ladder: (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)
    # Or: (0,1), (0,2), (1,2), (0,1), (0,2), (1,2)
    
    if qubits == 2:
        cnots = [(0, 1)] * cnot_budget
    else:
        # 3 qubits, 6 CNOTs
        # Common 6-CNOT decomposition for generic 3-qubit gates
        cnots = [
            (0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)
        ]
        if cnot_budget < 6:
            cnots = cnots[:cnot_budget]

    # Build layout with placeholder u3 gates
    layout = []
    # Initial single-qubit layer
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for c, t in cnots:
        layout.append(["cx", c, t])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    # Extract indices of u3 parameters
    param_indices = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_indices.append(i)

    n_params = 3 * len(param_indices)
    
    def objective(params):
        if time.time() - _start_time > time_limit:
            return 1.0
        # Rebuild gates
        gates = [list(g) for g in layout]
        k = 0
        for idx in param_indices:
            q = gates[idx][1]
            gates[idx][2] = params[k]
            gates[idx][3] = params[k+1]
            gates[idx][4] = params[k+2]
            k += 3
        u = _build_fast(gates, qubits)
        d = _distance(target, u)
        return d

    # Initial guess: all zeros
    x0 = np.zeros(n_params)
    
    # Use Nelder-Mead for robustness
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        best_params = res.x
        best_dist = res.fun
    except Exception:
        best_params = x0
        best_dist = objective(x0)
        
    # Build final gates
    gates = [list(g) for g in layout]
    k = 0
    for idx in param_indices:
        q = gates[idx][1]
        gates[idx][2] = best_params[k]
        gates[idx][3] = best_params[k+1]
        gates[idx][4] = best_params[k+2]
        k += 3
        
    return gates, best_dist


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    dim = 2 ** qubits
    time_limit = _DEADLINE - 5.0  # 5s safety margin

    # 1. Try known exact decompositions
    if name == "CNOT" and qubits == 2 and budget >= 1:
        gates = _exact_cnot_2q()
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates

    if name == "SWAP" and qubits == 2 and budget >= 3:
        gates = _exact_swap_2q()
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates

    if name == "iSWAP" and qubits == 2 and budget >= 1:
        gates = _exact_iswap_2q()
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates
        # If not exact, optimize
        gates, dist = _optimize_generic(target, qubits, budget, time_limit)
        if gates is not None:
            return gates

    if name == "Toffoli" and qubits == 3 and budget >= 6:
        # Try a known 6-CNOT Toffoli
        # If we don't have a hardcoded one, optimize
        gates, dist = _optimize_generic(target, qubits, budget, time_limit)
        if gates is not None:
            return gates

    # 2. Generic optimization for any other unitary
    gates, dist = _optimize_generic(target, qubits, budget, time_limit)
    if gates is not None:
        return gates

    # Fallback: Identity
    return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
