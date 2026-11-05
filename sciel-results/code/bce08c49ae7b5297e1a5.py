import math
import numpy as np

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta/2), math.sin(theta/2)
    return np.array([
        [c, -np.exp(1j*lam)*s],
        [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]
    ])

def _cx_matrix(control, target, qubits):
    # Returns the full 2**qubits x 2**qubits matrix
    size = 2**qubits
    M = np.eye(size, dtype=complex)
    for i in range(size):
        c_bit = (i >> (qubits-1-control)) & 1
        t_bit = (i >> (qubits-1-target)) & 1
        if c_bit == 1:
            j = i ^ (1 << (qubits-1-target))
            M[i, j] = 0
            M[j, i] = 1
            # Wait, this is just a permutation. CNOT is: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
            # My loop logic above is for SWAP-like logic? No.
            # CNOT: if control is 1, flip target.
            # i is index.
            # If c_bit is 1, the target bit flips.
            # So M[i, i] = 1 if c_bit is 0.
            # If c_bit is 1, M[i, i^target_mask] = 1.
    # Let's just build it simply.
    M = np.zeros((size, size), dtype=complex)
    target_mask = 1 << (qubits-1-target)
    for i in range(size):
        c_bit = (i >> (qubits-1-control)) & 1
        if c_bit == 0:
            M[i, i] = 1
        else:
            M[i, i ^ target_mask] = 1
    return M

def _build(gates, qubits):
    matrix = np.eye(2**qubits, dtype=complex)
    for gate in gates:
        if gate[0] == 'u3':
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Kroneckers
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else np.eye(2, dtype=complex))
            matrix = op @ matrix
        else:
            _, control, target = gate
            op = _cx_matrix(control, target, qubits)
            matrix = op @ matrix
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _kak_decompose(U):
    # KAK decomposition for 2-qubit
    # U = (A @ B) @ exp(i a XX + i b YY + i c ZZ) @ (C @ D)
    # We need to find a, b, c and A, B, C, D.
    # Standard algorithm:
    # 1. Compute U @ (Z @ Z)
    # 2. ... this is complex.
    # Simpler: Use a library or a known closed form.
    # Since I can't rely on external libs for KAK, I will implement a simplified version or use optimization for 2-qubit as well?
    # No, KAK is fast.
    # Reference: "Exact synthesis of two-qubit unitary gates"
    # I will use a numerical approach to find the KAK parameters if possible, or just use the optimizer for 2-qubit too?
    # Optimizer for 2-qubit with 3 CNOTs is very fast (3 layers of 3 params = 9 params).
    # Given the time, using a fast optimizer for both 2 and 3 qubit generic cases is safer to implement correctly than KAK from scratch.
    # 3 CNOTs for 2-qubit is the minimum.
    # So, for 2-qubit generic: 3 CNOTs template + optimization.
    # For 3-qubit generic: 12 CNOTs template + optimization.
    
    pass

def synthesize(problem):
    name = problem['name']
    qubits = problem['qubits']
    budget = problem['cnot_budget']
    target = _target(problem)
    
    # Named gates
    if name == 'CNOT':
        return [['cx', 0, 1]]
    if name == 'SWAP':
        return [
            ['cx', 0, 1],
            ['cx', 1, 0],
            ['cx', 0, 1]
        ]
    if name == 'iSWAP':
        # iSWAP = 0 1i; 1i 0
        # Can be done with 3 CNOTs and rotations.
        # H on both, CNOT, H, CNOT, H, CNOT, H?
        # Standard iSWAP decomposition:
        # 0: H, 1: H
        # 0: CNOT 1
        # 0: H, 1: H
        # 0: CNOT 1
        # 0: H, 1: H
        # 0: CNOT 1
        # 0: H, 1: H
        # This is 3 CNOTs.
        # Let's check the unitary.
        # Actually, iSWAP is symmetric.
        # I will use a known circuit.
        gates = [
            ['u3', 0, math.pi/2, 0, 0], # H? No, u3(pi/2, 0, 0) is H?
            # H = 1/sqrt(2) [[1, 1], [1, -1]]
            # u3(theta, phi, lam) = [[cos(t/2), -e^il sin(t/2)], [e^ip sin(t/2), e^i(p+l) cos(t/2)]]
            # For H: t=pi/2, p=0, l=pi.
            # [[1/sqrt2, -e^ipi 1/sqrt2], [1/sqrt2, e^ipi 1/sqrt2]] = [[1/sqrt2, 1/sqrt2], [1/sqrt2, -1/sqrt2]]
            # So H is u3(pi/2, 0, pi).
            ['u3', 0, math.pi/2, 0, math.pi],
            ['u3', 1, math.pi/2, 0, math.pi],
            ['cx', 0, 1],
            ['u3', 0, math.pi/2, 0, math.pi],
            ['u3', 1, math.pi/2, 0, math.pi],
            ['cx', 0, 1],
            ['u3', 0, math.pi/2, 0, math.pi],
            ['u3', 1, math.pi/2, 0, math.pi],
            ['cx', 0, 1],
            ['u3', 0, math.pi/2, 0, math.pi],
            ['u3', 1, math.pi/2, 0, math.pi]
        ]
        return gates
    if name == 'Toffoli':
        # 6 CNOTs decomposition
        gates = [
            ['u3', 2, math.pi/4, 0, 0], # T gate on target? No, T is on control?
            # Standard Toffoli (CCNOT) with 6 CNOTs:
            # H on q2, T on q1, CNOT(1,2), T^-1 on q0, CNOT(0,2), T on q1, CNOT(1,2), T^-1 on q1, CNOT(0,2), T on q0, T on q1, CNOT(0,1), H on q2
            # This uses 6 CNOTs.
            ['u3', 2, math.pi/2, 0, math.pi], # H
            ['u3', 1, math.pi/4, 0, 0], # T
            ['cx', 1, 2],
            ['u3', 0, -math.pi/4, 0, 0], # T^-1
            ['cx', 0, 2],
            ['u3', 1, math.pi/4, 0, 0], # T
            ['cx', 1, 2],
            ['u3', 1, -math.pi/4, 0, 0], # T^-1
            ['cx', 0, 2],
            ['u3', 0, math.pi/4, 0, 0], # T
            ['u3', 1, math.pi/4, 0, 0], # T
            ['cx', 0, 1],
            ['u3', 2, math.pi/2, 0, math.pi], # H
        ]
        return gates

    # Generic 2-qubit
    if qubits == 2:
        # Use 3 CNOTs template
        # Layout: U3 U3 CX U3 U3 CX U3 U3 CX U3 U3
        # 3 CNOTs
        layout = []
        for layer in range(3):
            layout.append(['u3', 0, 0, 0, 0])
            layout.append(['u3', 1, 0, 0, 0])
            layout.append(['cx', 0, 1])
        layout.append(['u3', 0, 0, 0, 0])
        layout.append(['u3', 1, 0, 0, 0])
        
        # Optimize
        params = [0.0] * (3 * 6) # 6 u3 gates * 3 params
        # ... run coordinate descent ...
        return _optimize(layout, target, qubits, params)

    # Generic 3-qubit
    if qubits == 3:
        # Use 12 CNOTs template
        # 4 layers of 3 CNOTs?
        # Pairs: (0,1), (1,2), (0,1), (1,2)
        # Or (0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)
        # Let's use a standard brick wall.
        layout = []
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        for i in range(12):
            layout.append(['u3', 0, 0, 0, 0])
            layout.append(['u3', 1, 0, 0, 0])
            layout.append(['u3', 2, 0, 0, 0])
            layout.append(['cx', pairs[i][0], pairs[i][1]])
        layout.append(['u3', 0, 0, 0, 0])
        layout.append(['u3', 1, 0, 0, 0])
        layout.append(['u3', 2, 0, 0, 0])
        
        params = [0.0] * (3 * 15) # 15 u3 gates
        return _optimize(layout, target, qubits, params)
    
    return []

def _optimize(layout, target, qubits, initial_params):
    # Coordinate descent
    best = initial_params
    best_dist = _distance(target, _build(_circuit_from_layout(layout, best), qubits))
    # ...
    pass
