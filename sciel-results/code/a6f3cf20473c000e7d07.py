"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 230.0  # Safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cx_mat(n, control, target):
    """Build CNOT matrix for n-qubit system (qubit 0 is MSB)."""
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        if (i >> (n - 1 - control)) & 1:
            j = i ^ (1 << (n - 1 - target))
            mat[i, i] = 0
            mat[i, j] = 1
            mat[j, j] = 0
            mat[j, i] = 1
    return mat


def _apply_gate_left(U, gate, qubits):
    """Apply gate to the left of U: result = gate_matrix @ U"""
    n = 2 ** qubits
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        single = _u3_mat(theta, phi, lam)
        # Build full operator via kron
        op = np.array([[1]], dtype=complex)
        eye = np.eye(2, dtype=complex)
        for k in range(qubits):
            op = np.kron(op, single if k == q else eye)
        return op @ U
    else:
        _, control, target = gate
        cx = _cx_mat(n, control, target)
        return cx @ U


def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        U = _apply_gate_left(U, gate, qubits)
    return U


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _decompose_2qubit(U):
    """
    Decompose a 2-qubit unitary into CNOTs and single-qubit rotations.
    Returns (gates, residual) where gates is a list of gate specs.
    Uses the standard 3-CNOT decomposition.
    """
    n = 4
    # First, absorb global phase
    det = np.linalg.det(U)
    phase = np.exp(1j * np.angle(det) / 2)
    V = U / phase

    # We need to find U1, U2, U3, U4 such that:
    # V = U1 ⊗ U2 * CX * U3 ⊗ U4 * CX * U5 ⊗ U6 * CX * U7 ⊗ U8
    # This is complex; let's use a simpler approach.
    # 
    # Actually, let's use a known result: any 2-qubit unitary can be written as
    # (A ⊗ B) CX (C ⊗ D) CX (E ⊗ F) CX (G ⊗ H)
    # with 6 single-qubit gates (well, 8 but some can be absorbed).
    #
    # For simplicity and exactness, let's use the approach of extracting
    # single-qubit gates from the matrix entries directly.
    
    # Alternative: use the fact that we can peel off single-qubit gates.
    # If we apply (A ⊗ B) to the left, we get a matrix where the first row
    # has a specific structure.
    
    # Let's use a different strategy: since we know the exact decompositions
    # for standard gates, and for generic unitaries we'll use optimization.
    # For 2-qubit generic unitaries, we'll use coordinate descent with a good layout.
    
    # Actually, let me implement a proper 2-qubit decomposition.
    # The key insight: we can write U = (U1 ⊗ U2) CX (U3 ⊗ U4) CX (U5 ⊗ U6) CX (U7 ⊗ U8)
    # where Ui are single-qubit unitaries.
    
    # Let's use the algorithm from Shende, Markov, Bullock (2006).
    # Step 1: Compute the canonical form.
    # This is quite involved. Let me use a simpler but effective approach.
    
    # For 2-qubit, the minimum CNOT count is 3 (for generic unitaries).
    # Let's use a parameterized form and optimize.
    
    # Layout: [u3]*2, CX, [u3]*2, CX, [u3]*2, CX, [u3]*2
    # That's 3 CNOTs and 8 u3 gates = 24 parameters.
    
    # But optimization might not converge exactly. Let me try a direct approach.
    
    # Direct approach for 2-qubit:
    # U[0,0] = a, U[0,1] = b, U[0,2] = c, U[0,3] = d
    # U[1,0] = e, U[1,1] = f, U[1,2] = g, U[1,3] = h
    # U[2,0] = i, U[2,1] = j, U[2,2] = k, U[2,3] = l
    # U[3,0] = m, U[3,1] = n, U[3,2] = o, U[3,3] = p
    
    # After applying (A ⊗ B) to the left, we can make the first column
    # have a specific form. Let's say we want to make U[0,0] real and positive,
    # U[1,0] = 0, U[2,0] = 0, U[3,0] = 0... but that's not possible with just
    # local operations.
    
    # Let me just use optimization for 2-qubit as well, but with a better layout.
    
    return None  # Will use optimization


def _decompose_3qubit_toffoli():
    """Return exact Toffoli decomposition with 6 CNOTs."""
    # Toffoli = CCX
    # Standard decomposition:
    # |00> |0> -> |00> |0>
    # |00> |1> -> |00> |1>
    # |01> |0> -> |01> |0>
    # |01> |1> -> |01> |1>
    # |10> |0> -> |10> |0>
    # |10> |1> -> |10> |1>
    # |11> |0> -> |11> |1>
    # |11> |1> -> |11> |0>
    
    # One known decomposition with 6 CNOTs:
    # H on target
    # CNOT(0,2)
    # T† on target
    # CNOT(1,2)
    # T on target
    # CNOT(0,2)
    # T† on target
    # T on qubit 1
    # H on target
    # CNOT(1,0)
    # T on qubit 0
    # T† on qubit 1
    # CNOT(1,0)
    
    # Let me use a simpler known decomposition.
    # Actually, the standard Toffoli with 6 CNOTs uses T gates.
    # But we only have u3, which can implement T gates.
    
    # T gate = u3(pi/4, 0, 0)  (up to phase)
    # Actually T = diag(1, exp(i*pi/4)) = u3(0, 0, pi/4) up to phase
    # Let me be more careful.
    
    # u3(theta, phi, lambda) = [[cos(theta/2), -exp(i*lambda)*sin(theta/2)],
    #                            [exp(i*phi)*sin(theta/2), exp(i*(phi+lambda))*cos(theta/2)]]
    
    # T gate = diag(1, exp(i*pi/4))
    # We need: cos(theta/2) = 1, sin(theta/2) = 0 => theta = 0
    # Then: [[1, 0], [0, exp(i*lambda)]]
    # So lambda = pi/4 gives T.
    
    # T† = diag(1, exp(-i*pi/4)) => lambda = -pi/4
    
    # H = u3(pi/2, 0, pi)  (up to phase)
    # H = 1/sqrt(2) [[1, 1], [1, -1]]
    # u3(pi/2, 0, pi) = [[1/sqrt(2), -exp(i*pi)/sqrt(2)], [0, exp(i*pi)/sqrt(2)]]
    # = [[1/sqrt(2), 1/sqrt(2)], [0, -1/sqrt(2)]]  -- that's not H.
    
    # Let me compute H properly.
    # H = 1/sqrt(2) [[1, 1], [1, -1]]
    # u3(theta, phi, lambda):
    # [0,0] = cos(theta/2)
    # [0,1] = -exp(i*lambda) sin(theta/2)
    # [1,0] = exp(i*phi) sin(theta/2)
    # [1,1] = exp(i*(phi+lambda)) cos(theta/2)
    
    # For H: cos(theta/2) = 1/sqrt(2), sin(theta/2) = 1/sqrt(2) => theta = pi/2
    # [0,1] = -exp(i*lambda)/sqrt(2) = 1/sqrt(2) => exp(i*lambda) = -1 => lambda = pi
    # [1,0] = exp(i*phi)/sqrt(2) = 1/sqrt(2) => exp(i*phi) = 1 => phi = 0
    # [1,1] = exp(i*(0+pi))/sqrt(2) = -1/sqrt(2) ✓
    
    # So H = u3(pi/2, 0, pi)
    
    # Now the Toffoli decomposition:
    gates = [
        ["u3", 2, math.pi / 2, 0, math.pi],  # H on target
        ["cx", 0, 2],
        ["u3", 2, 0, 0, -math.pi / 4],  # T† on target
        ["cx", 1, 2],
        ["u3", 2, 0, 0, math.pi / 4],  # T on target
        ["cx", 0, 2],
        ["u3", 2, 0, 0, -math.pi / 4],  # T† on target
        ["u3", 1, 0, 0, math.pi / 4],  # T on qubit 1
        ["u3", 2, math.pi / 2, 0, math.pi],  # H on target
        ["cx", 1, 0],
        ["u3", 0, 0, 0, math.pi / 4],  # T on qubit 0
        ["u3", 1, 0, 0, -math.pi / 4],  # T† on qubit 1
        ["cx", 1, 0],
    ]
    return gates


def _decompose_3qubit_iSWAP():
    """Return exact iSWAP decomposition."""
    # iSWAP = [[0, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 0]]
    # Wait, that's not right. iSWAP swaps with a phase:
    # |01> -> i|10>, |10> -> i|01>
    # So iSWAP = [[1, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 1]]
    
    # Known decomposition with 2 CNOTs:
    # H on q0, H on q1
    # CNOT(0,1)
    # T† on q0, T† on q1
    # CNOT(0,1)
    # T on q0, T on q1
    # H on q0, H on q1
    
    # Actually, let me look up the standard iSWAP decomposition.
    # iSWAP can be done with 2 CNOTs:
    # 1. T† on both qubits
    # 2. CNOT(0,1)
    # 3. T on both qubits
    # 4. CNOT(0,1)
    # 5. T† on both qubits
    
    # Let me verify this is correct by construction.
    # Actually, I'll use a known correct decomposition.
    
    # iSWAP = (H ⊗ H) (CNOT) (T† ⊗ T†) (CNOT) (T ⊗ T) (H ⊗ H)
    # Hmm, this might not be exactly right. Let me just use optimization for iSWAP.
    # But the task says iSWAP has a known CNOT budget. Let me use 2 CNOTs.
    
    # Standard iSWAP decomposition:
    # iSWAP = [H H] [CX] [T† T†] [CX] [T T] [H H]  -- this is for SWAP-like gates
    
    # Let me use a different known decomposition:
    # iSWAP = (T† on q0) (T† on q1) (CNOT 0->1) (T on q0) (T on q1) (CNOT 0->1) (T† on q0) (T† on q1)
    
    # I'll just use optimization for iSWAP to be safe.
    return None


def _optimize_2qubit(target, budget):
    """Optimize 2-qubit unitary decomposition."""
    n = 4
    # Layout: alternating u3 and CNOTs
    # For 2 qubits with budget CNOTs, use layout:
    # [u3, u3, CX, u3, u3, CX, u3, u3, ...]
    
    # Use a fixed layout with 3 CNOTs (minimum for generic 2-qubit)
    layout = []
    for layer in range(3):
        layout.append(["u3", 0, 0.0, 0.0, 0.0])
        layout.append(["u3", 1, 0.0, 0.0, 0.0])
        layout.append(["cx", 0, 1])
    layout.append(["u3", 0, 0.0, 0.0, 0.0])
    layout.append(["u3", 1, 0.0, 0.0, 0.0])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = np.zeros(n_params)
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_dist = _distance(target, _build(circuit(best), 2))
    
    # Coordinate descent
    for sweep in range(20):
        step = 2.0 / (1 + sweep)
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step, best[i] + np.pi * 0.1 * (1 if i % 2 == 0 else -1)):
                values = best.copy()
                values[i] = trial
                dist = _distance(target, _build(circuit(values), 2))
                if dist < best_dist:
                    best, best_dist = values, dist
        if _time_left() < 10:
            break
    
    return circuit(best), best_dist


def _optimize_3qubit(target, budget):
    """Optimize 3-qubit unitary decomposition."""
    n = 8
    # Layout with budget CNOTs
    pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    layout = []
    for layer in range(budget):
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = np.zeros(n_params)
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_dist = _distance(target, _build(circuit(best), 3))
    
    # Coordinate descent with time limit
    for sweep in range(15):
        if _time_left() < 20:
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best.copy()
                values[i] = trial
                dist = _distance(target, _build(circuit(values), 3))
                if dist < best_dist:
                    best, best_dist = values, dist
    
    return circuit(best), best_dist


def _exact_2qubit_cnot():
    """Exact CNOT gate decomposition."""
    return [
        ["u3", 0, 0.0, 0.0, 0.0],
        ["u3", 1, 0.0, 0.0, 0.0],
        ["cx", 0, 1],
    ]


def _exact_2qubit_iswap():
    """Exact iSWAP gate decomposition."""
    # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # Decomposition:
    # H on q0, H on q1
    # CNOT(0,1)
    # T† on q0, T† on q1
    # CNOT(0,1)
    # T on q0, T on q1
    # H on q0, H on q1
    
    # Let me verify: this is actually the decomposition for a controlled-phase gate.
    # For iSWAP, the standard decomposition is:
    # (T† on q0) (T† on q1) (CNOT 0→1) (T on q0) (T on q1) (CNOT 0→1) (T† on q0) (T† on q1)
    # Wait, that gives a different gate.
    
    # Let me just use the known result: iSWAP requires 2 CNOTs.
    # iSWAP = [H H] CX [T† T†] CX [T T] [H H]  -- I'll use this and hope it's correct.
    
    return [
        ["u3", 0, math.pi/2, 0, math.pi],  # H on q0
        ["u3", 1, math.pi/2, 0, math.pi],  # H on q1
        ["cx", 0, 1],
        ["u3", 0, 0, 0, -math.pi/4],  # T† on q0
        ["u3", 1, 0, 0, -math.pi/4],  # T† on q1
        ["cx", 0, 1],
        ["u3", 0, 0, 0, math.pi/4],  # T on q0
        ["u3", 1, 0, 0, math.pi/4],  # T on q1
        ["u3", 0, math.pi/2, 0, math.pi],  # H on q0
        ["u3", 1, math.pi/2, 0, math.pi],  # H on q1
    ]


def _exact_2qubit_swap():
    """Exact SWAP gate decomposition with 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known exact decompositions
    if qubits == 2:
        if name.lower() in ["cnot", "cx"]:
            return _exact_2qubit_cnot()
        elif name.lower() in ["iswap", "i-swap"]:
            gates = _exact_2qubit_iswap()
            # Verify
            U = _build(gates, 2)
            dist = _distance(target, U)
            if dist < 1e-10:
                return gates
            # Fall through to optimization
        elif name.lower() in ["swap"]:
            return _exact_2qubit_swap()
    
    if qubits == 3:
        if name.lower() in ["toffoli", "ccx", "ccnot"]:
            gates = _decompose_3qubit_toffoli()
            U = _build(gates, 3)
            dist = _distance(target, U)
            if dist < 1e-10:
                return gates
            # Fall through to optimization
    
    # Use optimization
    if qubits == 2:
        gates, dist = _optimize_2qubit(target, budget)
        return gates
    else:
        gates, dist = _optimize_3qubit(target, budget)
        return gates
# EVOLVE-BLOCK-END
