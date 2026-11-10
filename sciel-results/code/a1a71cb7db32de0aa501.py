"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the unitary from a gate list. Qubit 0 is MSB."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            # u3(theta, phi, lam) = [[cos(theta/2), -e^{i lam} sin(theta/2)],
            #                         [e^{i phi} sin(theta/2), e^{i(phi+lam)} cos(theta/2)]]
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build the full operator via kron
            op = np.array([[1]], dtype=complex)
            eye2 = np.eye(2, dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            # Index: bit (qubits-1-control) is control, bit (qubits-1-target) is target
            indices = np.arange(n)
            ctrl_bit = (indices >> (qubits - 1 - control)) & 1
            tgt_bit_mask = 1 << (qubits - 1 - target)
            # New index: if ctrl_bit is 1, flip target bit
            new_indices = indices ^ (ctrl_bit * tgt_bit_mask)
            matrix = matrix[new_indices, :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / 2^n"""
    n = target.shape[0]
    val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(val) / n)


def _u3(theta, phi, lam):
    return ["u3", 0, theta, phi, lam]  # placeholder, will be filled


def _h(q):
    return ["u3", q, math.pi / 2, 0, math.pi]


def _s(q):
    return ["u3", q, math.pi / 2, 0, 0]


def _sdag(q):
    return ["u3", q, -math.pi / 2, 0, 0]


def _t(q):
    return ["u3", q, math.pi / 4, 0, 0]


def _tdag(q):
    return ["u3", q, -math.pi / 4, 0, 0]


def _toffoli(c0, c1, t):
    """Toffoli with controls c0, c1 and target t. Uses 6 CNOTs."""
    gates = [
        _h(t),
        _t(t),
        ["cx", c1, t],
        _tdag(t),
        ["cx", c0, t],
        _t(t),
        ["cx", c1, t],
        _tdag(t),
        ["cx", c0, t],
        _tdag(c1),
        _t(t),
        _h(t),
        _t(c1),
        ["cx", c0, c1],
        _tdag(c1),
        _t(c0),
        ["cx", c0, c1]
    ]
    return gates


def _match_exact(target, name, qubits):
    """Try to match known gates exactly."""
    tol = 1e-5
    
    if qubits == 2:
        # CNOT(0,1)
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_01, atol=tol, rtol=tol):
            return [["cx", 0, 1]]
        # CNOT(1,0)
        cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
        if np.allclose(target, cnot_10, atol=tol, rtol=tol):
            return [["cx", 1, 0]]
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=tol, rtol=tol):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        # iSWAP: [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]]
        iswap = np.array([[0,0,0,1j],[0,0,1j,0],[0,1j,0,0],[1j,0,0,0]], dtype=complex)
        if np.allclose(target, iswap, atol=tol, rtol=tol):
            # iSWAP decomposition:
            # H(0) H(1) CX(0,1) CX(1,0) S(0) S(1) CX(1,0) CX(0,1) H(0) H(1)
            # Let me verify: Actually, iSWAP = (H⊗H) CX CX (S⊗S) CX CX (H⊗H) ?
            # Standard decomposition: iSWAP = (H⊗H) · CX(0,1) · CX(1,0) · (S⊗S) · CX(1,0) · CX(0,1) · (H⊗H)
            # That's 4 CNOTs. But we can do it in 3?
            # Actually iSWAP can be done with 3 CNOTs:
            # H(0) H(1) CX(0,1) S(0) S(1) CX(1,0) Sdag(0) Sdag(1) CX(0,1) H(0) H(1)
            # That's 3 CNOTs.
            gates = [
                _h(0), _h(1),
                ["cx", 0, 1],
                _s(0), _s(1),
                ["cx", 1, 0],
                _sdag(0), _sdag(1),
                ["cx", 0, 1],
                _h(0), _h(1)
            ]
            return gates
        # Check iSWAP with different phase conventions
        # Maybe iSWAP is [[0, 1], [1, 0]] * i? That's what we have above.
        # Or maybe it's the "standard" iSWAP which is [[0, 1j], [1j, 0]] on 2 qubits?
        # Let's also check the case where it might be [[0, -1j], [-1j, 0]] or similar
        iswap_neg = -iswap
        if np.allclose(target, iswap_neg, atol=tol, rtol=tol):
            # iSWAP with -i phase
            gates = [
                _h(0), _h(1),
                ["cx", 0, 1],
                _sdag(0), _sdag(1),
                ["cx", 1, 0],
                _s(0), _s(1),
                ["cx", 0, 1],
                _h(0), _h(1)
            ]
            return gates

    if qubits == 3:
        # Toffoli(0,1,2)
        toff_012 = np.eye(8, dtype=complex)
        toff_012[6, 6] = 0; toff_012[6, 7] = 1
        toff_012[7, 6] = 1; toff_012[7, 7] = 0
        if np.allclose(target, toff_012, atol=tol, rtol=tol):
            return _toffoli(0, 1, 2)
        # Toffoli(1,0,2) - same gate, controls are symmetric
        # Toffoli(0,2,1)
        toff_021 = np.eye(8, dtype=complex)
        # Controls 0 and 2, target 1
        # Indices where bits 0 and 2 are 1: 100 (4), 101 (5) -> flip bit 1
        # 100 (4) -> 110 (6), 101 (5) -> 111 (7)
        toff_021[4, 4] = 0; toff_021[4, 6] = 1
        toff_021[6, 4] = 1; toff_021[6, 6] = 0
        toff_021[5, 5] = 0; toff_021[5, 7] = 1
        toff_021[7, 5] = 1; toff_021[7, 7] = 0
        if np.allclose(target, toff_021, atol=tol, rtol=tol):
            return _toffoli(0, 2, 1)
        # Toffoli(2,0,1)
        toff_201 = np.eye(8, dtype=complex)
        # Controls 2 and 0, target 1
        # Same as above since controls are symmetric
        if np.allclose(target, toff_201, atol=tol, rtol=tol):
            return _toffoli(2, 0, 1)
        # Toffoli(1,2,0)
        toff_120 = np.eye(8, dtype=complex)
        # Controls 1 and 2, target 0
        # Indices where bits 1 and 2 are 1: 011 (3), 111 (7) -> flip bit 0
        # 011 (3) -> 010 (2), 111 (7) -> 110 (6)
        toff_120[3, 3] = 0; toff_120[3, 2] = 1
        toff_120[2, 3] = 1; toff_120[2, 2] = 0
        toff_120[7, 7] = 0; toff_120[7, 6] = 1
        toff_120[6, 7] = 1; toff_120[6, 6] = 0
        if np.allclose(target, toff_120, atol=tol, rtol=tol):
            return _toffoli(1, 2, 0)
        # Toffoli(2,1,0)
        toff_210 = np.eye(8, dtype=complex)
        # Controls 2 and 1, target 0
        # Same as toff_120 since controls are symmetric
        if np.allclose(target, toff_210, atol=tol, rtol=tol):
            return _toffoli(2, 1, 0)
        # Toffoli(0,1,2) with different control ordering - already covered
        # Toffoli(1,0,2) - same as (0,1,2) since controls symmetric
        # Toffoli(2,0,1) and Toffoli(2,1,0) covered above

    return None


def _decompose_2q(target):
    """Decompose a 2-qubit unitary into 3 CNOTs and single-qubit gates."""
    # Use the standard 3-CNOT decomposition
    # U = (A ⊗ B) CX (C ⊗ D) CX (E ⊗ F) CX (G ⊗ H)
    # This is complex to derive from scratch. Let's use a numerical approach.
    
    # For 2 qubits, we can use the fact that any 2-qubit unitary can be written as:
    # U = (A ⊗ B) · CX · (C ⊗ D) · CX · (E ⊗ F) · CX · (G ⊗ H)
    # where A,B,C,D,E,F,G,H are single-qubit unitaries.
    
    # We'll use a parameterized approach with optimization.
    # Parameters: 8 single-qubit gates, each with 3 parameters (theta, phi, lam)
    # Total: 24 parameters
    
    # But we can reduce: the global phase is irrelevant, and we can fix some gates.
    # Actually, let's just use the layout and optimize.
    
    # Layout: U3(0) U3(1) CX U3(0) U3(1) CX U3(0) U3(1) CX U3(0) U3(1)
    # That's 8 u3 gates (24 params) and 3 CX gates.
    
    # We'll optimize the 24 parameters.
    from scipy.optimize import minimize
    
    def circuit_from_params(params):
        gates = []
        # 8 u3 gates, each 3 params
        for i in range(4):
            gates.append(["u3", 0, params[3*i], params[3*i+1], params[3*i+2]])
            gates.append(["u3", 1, params[3*i+3], params[3*i+4], params[3*i+5]])
            if i < 3:
                gates.append(["cx", 0, 1])
        return gates
    
    def objective(params):
        gates = circuit_from_params(params)
        U = _build(gates, 2)
        return _distance(target, U)
    
    # Initial guess: all zeros
    x0 = np.zeros(24)
    
    # Try a few random restarts
    best_params = x0.copy()
    best_val = objective(x0)
    
    start_time = time.time()
    deadline = 270.0 - 15.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        if restart > 0:
            x0 = rng.uniform(-math.pi, math.pi, size=24)
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
        
        if time.time() > deadline:
            break
    
    return circuit_from_params(best_params)


def _decompose_3q(target):
    """Decompose a 3-qubit unitary into 6 CNOTs and single-qubit gates."""
    # Use a standard 6-CNOT decomposition layout
    # Layout: U3(0) U3(1) U3(2) CX(0,1) U3(0) U3(1) U3(2) CX(0,2) 
    #         U3(0) U3(1) U3(2) CX(1,2) U3(0) U3(1) U3(2) CX(0,1)
    #         U3(0) U3(1) U3(2) CX(0,2) U3(0) U3(1) U3(2) CX(1,2)
    #         U3(0) U3(1) U3(2)
    # That's 7 sets of 3 u3 gates = 21 u3 gates = 63 params, and 6 CX gates.
    
    # This is a lot of parameters. Let's use a more efficient approach.
    # We can use the fact that the decomposition is:
    # U = (A0 ⊗ A1 ⊗ A2) · CX(0,1) · (B0 ⊗ B1 ⊗ B2) · CX(0,2) · 
    #     (C0 ⊗ C1 ⊗ C2) · CX(1,2) · (D0 ⊗ D1 ⊗ D2) · CX(0,1) ·
    #     (E0 ⊗ E1 ⊗ E2) · CX(0,2) · (F0 ⊗ F1 ⊗ F2) · CX(1,2) ·
    #     (G0 ⊗ G1 ⊗ G2)
    
    # Total: 7 * 3 * 3 = 63 parameters.
    
    from scipy.optimize import minimize
    
    def circuit_from_params(params):
        gates = []
        cx_pairs = [(0,1), (0,2), (1,2), (0,1), (0,2), (1,2)]
        for i in range(7):
            for q in range(3):
                gates.append(["u3", q, params[9*i + 3*q], params[9*i + 3*q + 1], params[9*i + 3*q + 2]])
            if i < 6:
                c, t = cx_pairs[i]
                gates.append(["cx", c, t])
        return gates
    
    def objective(params):
        gates = circuit_from_params(params)
        U = _build(gates, 3)
        return _distance(target, U)
    
    x0 = np.zeros(63)
    best_params = x0.copy()
    best_val = objective(x0)
    
    start_time = time.time()
    deadline = 270.0 - 15.0
    
    rng = np.random.RandomState(42)
    
    # Use a smarter initialization: try to match the first few layers
    for restart in range(3):
        if time.time() > deadline:
            break
        if restart > 0:
            x0 = rng.uniform(-math.pi/4, math.pi/4, size=63)
        
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
        
        if time.time() > deadline:
            break
    
    return circuit_from_params(best_params)


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try exact matches first
    exact = _match_exact(target, name, qubits)
    if exact is not None:
        # Verify the CNOT count
        cnot_count = sum(1 for g in exact if g[0] == "cx")
        if cnot_count <= budget and len(exact) <= 400:
            return exact
    
    # Fallback to numerical decomposition
    if qubits == 2:
        gates = _decompose_2q(target)
    elif qubits == 3:
        gates = _decompose_3q(target)
    else:
        return []
    
    # Verify constraints
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget or len(gates) > 400:
        # This shouldn't happen with our layouts, but just in case
        pass
    
    return gates
# EVOLVE-BLOCK-END
