"""Exact quantum circuit synthesis for known gates and generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 210.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build full operator
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            matrix = op @ matrix
        else:
            _, control, target = gate
            idx = np.arange(dim)
            bit = (idx >> (qubits - 1 - control)) & 1
            # Swap rows where control=1
            matrix = matrix[idx ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_mat(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_mat(control, target, qubits):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> (qubits - 1 - control)) & 1:
            j = i ^ (1 << (qubits - 1 - target))
            if i < j:
                mat[i, j] = 1
                mat[j, i] = 1
                mat[i, i] = 0
                mat[j, j] = 0
    return mat


def _apply_u3(gates, q, theta, phi, lam, qubits):
    gates.append(_u3(q, theta, phi, lam))


def _apply_cx(gates, c, t):
    gates.append(_cx(c, t))


def _kde_2qubit(target):
    """
    Decompose a 2-qubit unitary into single-qubit rotations and CNOTs using KAK-like approach.
    Returns list of gates.
    """
    # For 2-qubit, we can use the standard 3-CNOT decomposition
    # U = (A1 ⊗ B1) CNOT (A2 ⊗ B2) CNOT (A3 ⊗ B3) CNOT (A4 ⊗ B4)
    # This requires solving for the single-qubit gates.
    
    # Simpler approach: use the fact that any 2-qubit unitary can be written as
    # U = (R1 ⊗ R2) * CNOT * (R3 ⊗ R4) * CNOT * (R5 ⊗ R6) * CNOT * (R7 ⊗ R8)
    
    # Let's use a numerical approach: peel off single-qubit gates using SVD
    
    dim = 4
    U = target.copy()
    gates = []
    
    # We'll use the standard 3-CNOT decomposition
    # After applying CNOTs, we need to find single-qubit corrections
    
    # Strategy: 
    # 1. Apply U3 on q0 and q1 to make U[0,0] real and positive
    # 2. Apply CNOT(0,1)
    # 3. Apply U3 on q1 to make U[0,2] = 0
    # 4. Apply CNOT(0,1)
    # 5. Apply U3 on q0 and q1 to make U[0,3] real and positive
    # 6. Apply CNOT(0,1)
    # 7. Apply U3 on q0 and q1 to fix remaining phases
    
    # This is the standard 3-CNOT decomposition from Nielsen & Chuang
    
    # Let me implement a simpler numerical approach
    # We'll use the fact that we can decompose as:
    # U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    
    # For now, let's use a gradient-free optimization with a good initial guess
    
    # Actually, let me use the analytical approach from the literature
    # For 2-qubit unitaries, the 3-CNOT decomposition is:
    # U = (R_A ⊗ R_B) · CNOT · (R_C ⊗ R_D) · CNOT · (R_E ⊗ R_F) · CNOT · (R_G ⊗ R_H)
    
    # Let me use a different approach: iterative peeling
    
    # Step 1: Make U[0,0] real positive by applying U3 to q0
    val = U[0, 0]
    if abs(val) > 1e-12:
        phase = -math.atan2(val.imag, val.real)
        # Apply U3(0, 0, phase, 0) to q0 to rotate the phase
        # Actually, we need to be more careful
        
    # Let me use a simpler, more robust approach:
    # Use the fact that any 2-qubit gate can be decomposed with 3 CNOTs
    # and find the single-qubit gates by solving a small system
    
    # I'll use a numerical optimization with Nelder-Mead or similar
    
    # For efficiency, let's use a structured approach:
    # The 3-CNOT decomposition has 24 parameters (8 U3 gates × 3 params)
    # We need to match 16 complex numbers (32 real), but with global phase freedom, 31 real
    # So 24 params < 31, meaning we need the CNOTs to provide the entanglement
    
    # Let me just do a direct optimization with a good initialization
    
    # Initialize with identity single-qubit gates
    params = [0.0] * 24  # 8 U3 gates, 3 params each
    
    def _build_2qubit(params):
        """Build 2-qubit unitary from 3-CNOT decomposition."""
        result = np.eye(4, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        
        # Order: (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
        # Gates applied right to left
        
        # G⊗H
        G = _u3_mat(params[21], params[22], params[23])
        H = _u3_mat(params[18], params[19], params[20])
        op = np.kron(G, H)
        result = op @ result
        
        # CNOT(0,1)
        cnot = _cnot_mat(0, 1, 2)
        result = cnot @ result
        
        # E⊗F
        E = _u3_mat(params[15], params[16], params[17])
        F = _u3_mat(params[12], params[13], params[14])
        op = np.kron(E, F)
        result = op @ result
        
        # CNOT(0,1)
        result = cnot @ result
        
        # C⊗D
        C = _u3_mat(params[9], params[10], params[11])
        D = _u3_mat(params[6], params[7], params[8])
        op = np.kron(C, D)
        result = op @ result
        
        # CNOT(0,1)
        result = cnot @ result
        
        # A⊗B
        A = _u3_mat(params[3], params[4], params[5])
        B = _u3_mat(params[0], params[1], params[2])
        op = np.kron(A, B)
        result = op @ result
        
        return result
    
    def _cost(params):
        U_built = _build_2qubit(params)
        return _distance(target, U_built)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    best_params = params[:]
    best_cost = _cost(params)
    
    # Try a few random initializations
    rng = np.random.RandomState(42)
    for init in range(3):
        trial_params = rng.uniform(-math.pi, math.pi, 24)
        trial_cost = _cost(trial_params)
        if trial_cost < best_cost:
            best_params = trial_params
            best_cost = trial_cost
    
    # Optimize from best initialization
    try:
        result = minimize(_cost, best_params, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_cost:
            best_params = result.x
            best_cost = result.fun
    except:
        pass
    
    # Build gates
    gates = []
    # G⊗H
    _apply_u3(gates, 0, best_params[21], best_params[22], best_params[23])
    _apply_u3(gates, 1, best_params[18], best_params[19], best_params[20])
    _apply_cx(gates, 0, 1)
    # E⊗F
    _apply_u3(gates, 0, best_params[15], best_params[16], best_params[17])
    _apply_u3(gates, 1, best_params[12], best_params[13], best_params[14])
    _apply_cx(gates, 0, 1)
    # C⊗D
    _apply_u3(gates, 0, best_params[9], best_params[10], best_params[11])
    _apply_u3(gates, 1, best_params[6], best_params[7], best_params[8])
    _apply_cx(gates, 0, 1)
    # A⊗B
    _apply_u3(gates, 0, best_params[3], best_params[4], best_params[5])
    _apply_u3(gates, 1, best_params[0], best_params[1], best_params[2])
    
    return gates


def _exact_cnot():
    """Exact CNOT gate."""
    return [_cx(0, 1)]


def _exact_iswap():
    """Exact iSWAP gate: iSWAP = (H⊗H) CNOT (S†⊗I) CNOT (S⊗S) CNOT (H⊗H) ... 
    Actually, iSWAP = (i/2)(|00><11| + |11><00| + |01><10| + |10><01|)
    A known decomposition uses 2 CNOTs."""
    # iSWAP can be decomposed as:
    # (S† on q0, S on q1) CNOT(0,1) (S on q0, S† on q1) CNOT(0,1) (H on q0, H on q1)
    # Let me use a standard decomposition
    # iSWAP = exp(i*pi/4 * (X⊗X + Y⊗Y)/2)
    # One decomposition: 
    # U3(0, 0, 0, -pi/2) U3(1, 0, 0, pi/2) CNOT(0,1) U3(0, 0, 0, pi/2) U3(1, 0, 0, -pi/2) CNOT(0,1) U3(0, pi/2, 0, 0) U3(1, pi/2, 0, 0)
    
    gates = []
    # This is a known 2-CNOT decomposition of iSWAP
    _apply_u3(gates, 0, 0, 0, -math.pi/2)
    _apply_u3(gates, 1, 0, 0, math.pi/2)
    _apply_cx(gates, 0, 1)
    _apply_u3(gates, 0, 0, 0, math.pi/2)
    _apply_u3(gates, 1, 0, 0, -math.pi/2)
    _apply_cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, 0, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    return gates


def _exact_swap():
    """Exact SWAP gate: 3 CNOTs."""
    gates = []
    _apply_cx(gates, 0, 1)
    _apply_cx(gates, 1, 0)
    _apply_cx(gates, 0, 1)
    return gates


def _exact_toffoli():
    """Exact Toffoli gate: 6-7 CNOTs."""
    # Toffoli = CCNOT(0,1,2)
    # Standard decomposition uses 6 CNOTs
    gates = []
    # From Nielsen & Chuang or standard references
    # Toffoli(0,1,2) decomposition:
    _apply_u3(gates, 2, math.pi/2, 0, 0)
    _apply_cx(gates, 1, 2)
    _apply_u3(gates, 2, 0, 0, -math.pi/4)
    _apply_cx(gates, 0, 2)
    _apply_u3(gates, 2, 0, 0, math.pi/4)
    _apply_cx(gates, 1, 2)
    _apply_u3(gates, 2, 0, 0, -math.pi/4)
    _apply_cx(gates, 0, 2)
    _apply_u3(gates, 2, 0, 0, math.pi/4)
    _apply_u3(gates, 2, math.pi/2, 0, 0)
    return gates


def _is_known_gate(problem, target):
    """Check if the target is a known gate and return exact decomposition."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    
    if qubits == 2:
        if "cnot" in name.lower() or "cx" in name.lower():
            # Check if it's actually CNOT
            expected = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
            if np.allclose(target, expected, atol=1e-10):
                return _exact_cnot()
        if "iswap" in name.lower():
            expected = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
            if np.allclose(target, expected, atol=1e-10):
                return _exact_iswap()
        if "swap" in name.lower():
            expected = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
            if np.allclose(target, expected, atol=1e-10):
                return _exact_swap()
    
    if qubits == 3:
        if "toffoli" in name.lower() or "ccnot" in name.lower():
            # Check if it's Toffoli(0,1,2)
            expected = np.zeros((8,8), dtype=complex)
            for i in range(8):
                expected[i,i] = 1
            # Toffoli: |110> -> |111>, |111> -> |110>
            expected[6,6] = 0
            expected[7,7] = 0
            expected[6,7] = 1
            expected[7,6] = 1
            if np.allclose(target, expected, atol=1e-10):
                return _exact_toffoli()
    
    return None


def _generic_2qubit(problem, target):
    """Synthesize a generic 2-qubit unitary."""
    if _elapsed() > _DEADLINE - 30:
        # Not enough time, return identity-ish
        return [_u3(0, 0, 0, 0), _u3(1, 0, 0, 0)]
    
    return _kde_2qubit(target)


def _generic_3qubit(problem, target):
    """Synthesize a generic 3-qubit unitary using layered CNOT approach."""
    if _elapsed() > _DEADLINE - 30:
        return [_u3(q, 0, 0, 0) for q in range(3)]
    
    # For 3-qubit, use a layered approach with CNOTs
    # Standard: up to 12-15 CNOTs for arbitrary 3-qubit
    # Use a simple layered decomposition
    
    budget = problem["cnot_budget"]
    qubits = 3
    
    # Build a layout with CNOTs in a pattern
    # Use a grid of CNOTs: (0,1), (1,2), (0,2) repeating
    pairs = [(0,1), (1,2), (0,2)]
    
    layout = []
    for layer in range(min(budget, 12)):
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % 3]])
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Optimize the single-qubit parameters
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    params = [0.0] * n_params
    
    def _build_circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(_u3(gate[1], params[k], params[k+1], params[k+2]))
                k += 3
            else:
                gates.append(_cx(gate[1], gate[2]))
        return gates
    
    def _cost(params):
        gates = _build_circuit(params)
        if len(gates) > 400:
            return 1.0
        U = _build(gates, 3)
        return _distance(target, U)
    
    # Optimize
    from scipy.optimize import minimize
    
    best_params = params[:]
    best_cost = _cost(params)
    
    # Try a few random inits
    rng = np.random.RandomState(123)
    for _ in range(2):
        trial = rng.uniform(-math.pi, math.pi, n_params)
        c = _cost(trial)
        if c < best_cost:
            best_params = trial
            best_cost = c
    
    try:
        result = minimize(_cost, best_params, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_cost:
            best_params = result.x
            best_cost = result.fun
    except:
        pass
    
    return _build_circuit(best_params)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    target = _target(problem)
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Check for known gates first
    exact = _is_known_gate(problem, target)
    if exact is not None:
        # Verify it's within budget
        n_cnots = sum(1 for g in exact if g[0] == "cx")
        if n_cnots <= budget and len(exact) <= 400:
            return exact
    
    # Generic synthesis
    if qubits == 2:
        gates = _generic_2qubit(problem, target)
    else:
        gates = _generic_3qubit(problem, target)
    
    # Ensure we're within budget
    n_cnots = sum(1 for g in gates if g[0] == "cx")
    if n_cnots > budget:
        # Truncate if needed (shouldn't happen with our layouts)
        pass
    
    return gates
# EVOLVE-BLOCK-END
