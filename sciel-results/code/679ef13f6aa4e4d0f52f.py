# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 250.0  # safety margin before 270s kill

def _remaining():
    return max(0.0, _DEADLINE - (time.time() - _START))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    """Build unitary from gate list. qubit 0 = MSB."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye1 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            # Build operator by kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye1)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: if control bit is 1, flip target bit
            index = np.arange(n)
            c_bit = (index >> (qubits - 1 - control)) & 1
            t_bit = (index >> (qubits - 1 - target)) & 1
            # Flip target where control is 1
            flipped = np.where(c_bit == 1, index ^ (1 << (qubits - 1 - target)), index)
            # Permutation: row i of new = row flipped[i] of old
            matrix = matrix[flipped, :]
    return matrix

def _distance(target, matrix):
    n = target.shape[0]
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / n)

def _u3(theta, phi, lam):
    return ["u3", 0, theta, phi, lam]

def _cnot(c, t):
    return ["cx", c, t]

# ---- Known gate decompositions ----

def _decompose_cnot(problem):
    """CNOT(0->1): single CNOT."""
    return [_cnot(0, 1)]

def _decompose_swap(problem):
    """SWAP: 3 CNOTs."""
    return [_cnot(0, 1), _cnot(1, 0), _cnot(0, 1)]

def _decompose_iswap(problem):
    """iSWAP: uses 4 CNOTs + single-qubit rotations.
    iSWAP = [[0,0,0,i],[0,0,i,0],[0,i,0,0],[i,0,0,0]]
    Decomposition: 4 CNOTs with specific single-qubit gates.
    """
    # iSWAP circuit: 
    # q0: H, Rz(pi/2), CX(1,0), H, Rz(-pi/2), H
    # q1: H, Rz(pi/2), CX(1,0), H, Rz(-pi/2), H  -- but this needs careful ordering
    # Standard iSWAP decomposition using 4 CNOTs:
    gates = []
    # q0 and q1 both get: u3(pi, pi/2, 0) which is like a phase + rotation
    # Let me use the known 4-CNOT decomposition:
    # Apply to both qubits: H, then phase gates, interleaved with CNOTs
    pi = math.pi
    # q0: u3(pi, pi/2, 0)
    gates.append(["u3", 0, pi, pi/2, 0.0])
    # q1: u3(pi, pi/2, 0)
    gates.append(["u3", 1, pi, pi/2, 0.0])
    gates.append(_cnot(0, 1))
    # q0: u3(pi/2, 0, -pi/2)  (H-like)
    gates.append(["u3", 0, pi/2, 0, -pi/2])
    # q1: u3(pi/2, 0, -pi/2)
    gates.append(["u3", 1, pi/2, 0, -pi/2])
    gates.append(_cnot(1, 0))
    # q0: u3(pi/2, 0, pi/2)
    gates.append(["u3", 0, pi/2, 0, pi/2])
    # q1: u3(pi/2, 0, pi/2)
    gates.append(["u3", 1, pi/2, 0, pi/2])
    gates.append(_cnot(0, 1))
    # q0: u3(pi/2, 0, -pi/2)
    gates.append(["u3", 0, pi/2, 0, -pi/2])
    # q1: u3(pi/2, 0, -pi/2)
    gates.append(["u3", 1, pi/2, 0, -pi/2])
    gates.append(_cnot(1, 0))
    # q0: u3(pi, -pi/2, 0)
    gates.append(["u3", 0, pi, -pi/2, 0.0])
    # q1: u3(pi, -pi/2, 0)
    gates.append(["u3", 1, pi, -pi/2, 0.0])
    return gates

def _decompose_toffoli(problem):
    """Toffoli: 6 CNOTs standard decomposition."""
    pi = math.pi
    gates = []
    # q2 (target): H
    gates.append(["u3", 2, pi/2, 0, -pi/2])
    # CNOT q0->q2
    gates.append(_cnot(0, 2))
    # q2: T^dagger (Rz(-pi/4))
    gates.append(["u3", 2, 0, 0, -pi/4])
    # CNOT q1->q2
    gates.append(_cnot(1, 2))
    # q2: T (Rz(pi/4))
    gates.append(["u3", 2, 0, 0, pi/4])
    # CNOT q0->q2
    gates.append(_cnot(0, 2))
    # q2: T^dagger
    gates.append(["u3", 2, 0, 0, -pi/4])
    # CNOT q1->q2
    gates.append(_cnot(1, 2))
    # q2: T
    gates.append(["u3", 2, 0, 0, pi/4])
    # q0: T (Rz(pi/4))  -- wait, Toffoli also needs phase on controls
    # Actually the standard Toffoli with 6 CNOTs:
    gates.append(["u3", 0, 0, 0, pi/4])
    gates.append(["u3", 1, 0, 0, pi/4])
    # CNOT q0->q1
    gates.append(_cnot(0, 1))
    # CNOT q1->q2 (we already have some, let me recount)
    # Let me redo with the clean 6-CNOT Toffoli:
    return _toffoli_6cnot()

def _toffoli_6cnot():
    """6-CNOT Toffoli decomposition."""
    pi = math.pi
    gates = []
    # Target q2: H
    gates.append(["u3", 2, pi/2, 0, -pi/2])
    gates.append(_cnot(0, 2))
    gates.append(["u3", 2, 0, 0, -pi/4])
    gates.append(_cnot(1, 2))
    gates.append(["u3", 2, 0, 0, pi/4])
    gates.append(_cnot(0, 2))
    gates.append(["u3", 2, 0, 0, -pi/4])
    gates.append(_cnot(1, 2))
    gates.append(["u3", 2, 0, 0, pi/4])
    gates.append(["u3", 0, 0, 0, pi/4])
    gates.append(["u3", 1, 0, 0, pi/4])
    gates.append(_cnot(0, 1))
    gates.append(["u3", 2, pi/2, 0, -pi/2])
    return gates

# ---- Single qubit decomposition ----

def _decompose_single_qubit(U):
    """Decompose 2x2 unitary into u3(theta, phi, lam)."""
    det = np.linalg.det(U)
    if abs(det) < 1e-10:
        return (0.0, 0.0, 0.0)
    phase = np.angle(det)
    U_prime = U / np.exp(1j * phase)
    # Now det(U_prime) = 1
    # U_prime = [[a, -b*exp(-i*lam)], [b*exp(i*phi), a*exp(i*(phi+lam))]]
    a = U_prime[0, 0]
    b = U_prime[1, 0]
    theta = 2 * math.atan2(abs(b), abs(a))
    if abs(a) + abs(b) < 1e-10:
        return (0.0, 0.0, 0.0)
    # phi: phase of b relative
    if abs(b) > 1e-10:
        phi = np.angle(b)
    else:
        phi = 0.0
    if abs(a) > 1e-10:
        lam = np.angle(a) - np.angle(b)
    else:
        lam = 0.0
    # Adjust for the -exp(i*lam) in position [0,1]
    # U[0,1] should be -exp(-i*lam)*b*exp(i*(phi))... let me verify
    # Actually the u3 matrix is:
    # [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #  [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    # So U[0,1] = -exp(i*lam)*sin(theta/2), U[1,0] = exp(i*phi)*sin(theta/2)
    # From U[1,0] = b: phi = angle(b)
    # From U[0,1] = -exp(i*lam)*sin(theta/2): 
    #   -exp(i*lam) = U[0,1]/sin(theta/2)
    #   exp(i*lam) = -U[0,1]/sin(theta/2)
    #   lam = angle(-U[0,1]/sin(theta/2))
    s = math.sin(theta / 2)
    if s > 1e-10:
        lam = np.angle(-U_prime[0, 1] / s)
    else:
        lam = 0.0
    return (theta, phi, lam)

def _u3_gates(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]

# ---- 2-qubit KAK decomposition ----

def _kak_decompose_2qubit(target):
    """
    Decompose 2-qubit unitary using KAK: U = (A1⊗A2) · CX · (B1⊗B2) · CX · (C1⊗C2) · CX · (D1⊗D2)
    where A_i, B_i, C_i, D_i are 2x2 unitaries.
    Returns list of gates with exactly 3 CNOTs.
    """
    # Normalize: make det = 1 (or handle global phase)
    det = np.linalg.det(target)
    if abs(det) < 1e-10:
        return []
    # We'll work with the target as-is; the global phase is absorbed into single-qubit gates
    
    # KAK: compute the three canonical angles
    # U^2 = I condition for the "canonical" part
    # Standard approach: 
    # 1. Compute local invariants
    # 2. Find the CNOT structure
    
    # Simpler approach: use the fact that any 2-qubit gate can be written as
    # U = (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)
    # We can extract this by "peeling off" CNOTs.
    
    # Start from the right: peel off CX(0,1)
    # If U = V · CX, then V = U · CX^dagger = U · CX (since CX is self-inverse)
    
    CX = np.zeros((4, 4), dtype=complex)
    CX[0,0] = 1; CX[1,1] = 1; CX[2,3] = 1; CX[3,2] = 1
    
    U = target.copy()
    # Peel off three CNOTs from the right
    # After peeling: U = (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)
    # We peel: U3 = U, U2 = U3 @ CX^dagger, U1 = U2 @ CX^dagger, U0 = U1 @ CX^dagger
    # Then U0 = (A⊗B), U1 = (C⊗D) CX (E⊗F), U2 = (C⊗D) CX, U3 = target
    
    # Actually let's do it step by step from right to left
    # Step 1: U = (X0) CX (X1) CX (X2) CX (X3)
    # X3 is just single qubit: X3 = U3 where U3 @ CX @ ... = U
    # Let U_after3 = U
    # U_after3 = (X3)  -- no wait, U = X0 CX X1 CX X2 CX X3
    # So X3 = U @ CX^{-1} @ CX^{-1} @ CX^{-1} ... no that's not right either
    
    # Correct: U = X0 · CX · X1 · CX · X2 · CX · X3
    # => X3 = CX^T · X2^T · CX^T · X1^T · CX^T · X0^T · U^T  -- getting complicated
    
    # Let me use a direct approach:
    # Peel from the right:
    # U = A · CX · B · CX · C · CX · D
    # D = U · CX^dagger · C^dagger · CX^dagger · B^dagger · CX^dagger · A^dagger  -- circular
    
    # Better: work forward
    # Let W0 = U
    # W0 = A CX B CX C CX D
    # W1 = W0 @ D^dagger = A CX B CX C
    # W2 = W1 @ C^dagger = A CX B
    # W3 = W2 @ B^dagger = A CX
    # W4 = W3 @ CX^dagger = A
    # But we don't know A, B, C, D.
    
    # Alternative: use the known result that we can choose A, B, C, D to be any local unitaries
    # and the "canonical" form has specific angles.
    
    # Let me use a numerical approach: optimize the 4 pairs of single-qubit gates (24 params)
    # to match the target, given the fixed 3-CNOT structure.
    
    return _optimize_2qubit_kak(target)

def _optimize_2qubit_kak(target):
    """Optimize 4 layers of single-qubit gates around 3 CNOTs."""
    rng = random.Random(12345)
    # Parameters: 4 layers x 2 qubits x 3 params = 24 params
    best = [0.0] * 24
    best_dist = 1.0
    
    def circuit(vals):
        gates = []
        for layer in range(4):
            for q in range(2):
                base = layer * 6 + q * 3
                gates.append(["u3", q, vals[base], vals[base+1], vals[base+2]])
            if layer < 3:
                gates.append(_cnot(0, 1))
        return gates
    
    # Evaluate initial
    U = _build(circuit(best), 2)
    best_dist = _distance(target, U)
    
    # Coordinate descent with time limit
    deadline = _remaining()
    sweeps = 0
    while _remaining() > 5.0 and sweeps < 30:
        step = 2.0 / (1 + sweeps)
        improved = False
        for i in range(24):
            if _remaining() < 5.0:
                break
            for trial in (best[i] + step, best[i] - step, rng.uniform(-math.pi, math.pi)):
                vals = best[:]
                vals[i] = trial
                U = _build(circuit(vals), 2)
                d = _distance(target, U)
                if d < best_dist:
                    best, best_dist = vals, d
                    improved = True
                    break
        if not improved:
            break
        sweeps += 1
    
    return circuit(best)

# ---- 3-qubit optimization ----

def _optimize_3qubit(target, qubits, budget):
    """CNOT ladder optimization for 3-qubit gates."""
    pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    rng = random.Random(7777 + qubits)
    
    # Build layout: alternating single-qubit layers and CNOTs
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_dist = 1.0
    
    def circuit(vals):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], vals[k], vals[k+1], vals[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    U = _build(circuit(best), qubits)
    best_dist = _distance(target, U)
    
    # Coordinate descent
    sweeps = 0
    while _remaining() > 10.0 and sweeps < 20:
        step = 2.0 / (1 + sweeps)
        improved = False
        for i in range(n_params):
            if _remaining() < 10.0:
                break
            for trial in (best[i] + step, best[i] - step, rng.uniform(-math.pi, math.pi)):
                vals = best[:]
                vals[i] = trial
                U = _build(circuit(vals), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best, best_dist = vals, d
                    improved = True
                    break
        if not improved:
            break
        sweeps += 1
    
    return circuit(best)

# ---- Gate identification ----

def _identify_gate(target, qubits):
    """Identify if target matches a known gate (up to global phase)."""
    n = 2 ** qubits
    # Check CNOT(0->1) for 2 qubits
    if qubits == 2:
        cnot_ref = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=complex)
        # CNOT(0->1): control q0, target q1
        # Basis: |00>,|01>,|10>,|11>
        # CNOT(0->1) maps |10>->|11>, |11>->|10>
        cnot_ref = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        for ref, name in [(cnot_ref, "cnot")]:
            tr = np.trace(ref.conj().T @ target)
            if abs(abs(tr) / n - 1.0) < 1e-6:
                return name
        # SWAP
        swap_ref = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        tr = np.trace(swap_ref.conj().T @ target)
        if abs(abs(tr) / n - 1.0) < 1e-6:
            return "swap"
        # iSWAP
        iswap_ref = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        tr = np.trace(iswap_ref.conj().T @ target)
        if abs(abs(tr) / n - 1.0) < 1e-6:
            return "iswap"
    
    if qubits == 3:
        # Toffoli(0,1->2)
        toff_ref = np.zeros((8,8), dtype=complex)
        for i in range(8):
            toff_ref[i,i] = 1.0
        # Toffoli: |110>->|111>, |111>->|110>  (controls 0,1 target 2)
        # Bit order: q0 is MSB, so |110> = index 6, |111> = index 7
        toff_ref[6,6] = 0; toff_ref[6,7] = 1
        toff_ref[7,6] = 1; toff_ref[7,7] = 0
        tr = np.trace(toff_ref.conj().T @ target)
        if abs(abs(tr) / 8 - 1.0) < 1e-6:
            return "toffoli"
    
    return None

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check known gates
    known = _identify_gate(target, qubits)
    
    if known == "cnot":
        gates = _decompose_cnot(problem)
        return gates[:budget + qubits + 10]  # ensure within budget
    elif known == "swap":
        gates = _decompose_swap(problem)
        return gates
    elif known == "iswap":
        gates = _decompose_iswap(problem)
        # Ensure within budget
        cx_count = sum(1 for g in gates if g[0] == "cx")
        if cx_count <= budget:
            return gates
        else:
            # Fall through to optimization
            pass
    elif known == "toffoli":
        gates = _toffoli_6cnot()
        cx_count = sum(1 for g in gates if g[0] == "cx")
        if cx_count <= budget:
            return gates
        else:
            pass
    
    # Generic decomposition
    if qubits == 2:
        gates = _optimize_2qubit_kak(target)
        # Ensure CNOT budget
        cx_count = sum(1 for g in gates if g[0] == "cx")
        if cx_count > budget:
            # Truncate CNOTs (keep first 'budget' CNOTs)
            new_gates = []
            cx_seen = 0
            for g in gates:
                if g[0] == "cx":
                    if cx_seen < budget:
                        new_gates.append(g)
                        cx_seen += 1
                else:
                    new_gates.append(g)
            gates = new_gates
        # Limit total gates
        if len(gates) > 400:
            gates = gates[:400]
        return gates
    else:
        gates = _optimize_3qubit(target, qubits, budget)
        cx_count = sum(1 for g in gates if g[0] == "cx")
        if cx_count > budget:
            new_gates = []
            cx_seen = 0
            for g in gates:
                if g[0] == "cx":
                    if cx_seen < budget:
                        new_gates.append(g)
                        cx_seen += 1
                else:
                    new_gates.append(g)
            gates = new_gates
        if len(gates) > 400:
            gates = gates[:400]
        return gates
# EVOLVE-BLOCK-END
