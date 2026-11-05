"""Rewritten program for quantum circuit synthesis."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Constants
TIME_LIMIT = 250  # Safety margin for 270s limit

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _is_close(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol, rtol=tol)

def _identify_and_synth(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    
    # 1. Check for Standard Gates
    if qubits == 2:
        # CNOT
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        # iSWAP
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        
        # Check CNOT
        # Note: Global phase might differ. We check distance to identity for phase-insensitive match?
        # Actually, just check if target is close to these matrices up to phase.
        # dist = 1 - |tr(T^dagger U)|/4. If U is CNOT, dist should be 0.
        
        # Try CNOT(0,1)
        gates_cnot_01 = [['cx', 0, 1]]
        if _distance(target, _build(gates_cnot_01, 2)) < 1e-5:
            return gates_cnot_01
            
        # Try CNOT(1,0)
        gates_cnot_10 = [['cx', 1, 0]]
        if _distance(target, _build(gates_cnot_10, 2)) < 1e-5:
            return gates_cnot_10

        # Try SWAP
        gates_swap = [['cx', 0, 1], ['cx', 1, 0], ['cx', 0, 1]]
        if _distance(target, _build(gates_swap, 2)) < 1e-5:
            return gates_swap

        # Try iSWAP
        # Decomposition: H0, H1, CX(0,1), Tdag(0), H0, CX(0,1), T(1), H1
        # T = u3(0, 0, pi/4), Tdag = u3(0, -pi/4, 0), H = u3(pi/2, 0, pi)
        gates_iswap = [
            ['u3', 0, math.pi/2, 0, math.pi],
            ['u3', 1, math.pi/2, 0, math.pi],
            ['cx', 0, 1],
            ['u3', 0, 0, -math.pi/4, 0],
            ['u3', 0, math.pi/2, 0, math.pi],
            ['cx', 0, 1],
            ['u3', 1, 0, 0, math.pi/4],
            ['u3', 1, math.pi/2, 0, math.pi]
        ]
        if _distance(target, _build(gates_iswap, 2)) < 1e-5:
            return gates_iswap

    elif qubits == 3:
        # Toffoli (0,1,2)
        # Standard 6-CNOT decomposition
        pi = math.pi
        H = [pi/2, 0, pi]
        T = [0, 0, pi/4]
        Tdag = [0, -pi/4, 0]
        
        gates_toffoli = [
            ['u3', 2, H[0], H[1], H[2]],
            ['cx', 1, 2],
            ['u3', 2, Tdag[0], Tdag[1], Tdag[2]],
            ['cx', 0, 2],
            ['u3', 2, T[0], T[1], T[2]],
            ['u3', 1, Tdag[0], Tdag[1], Tdag[2]],
            ['cx', 0, 1],
            ['u3', 1, T[0], T[1], T[2]],
            ['u3', 2, T[0], T[1], T[2]],
            ['cx', 1, 2],
            ['u3', 2, Tdag[0], Tdag[1], Tdag[2]],
            ['u3', 2, H[0], H[1], H[2]],
            ['u3', 1, Tdag[0], Tdag[1], Tdag[2]],
            ['u3', 0, Tdag[0], Tdag[1], Tdag[2]]
        ]
        if _distance(target, _build(gates_toffoli, 3)) < 1e-5:
            return gates_toffoli

    # 2. Generic Synthesis using Optimization
    start_time = time.time()
    
    if qubits == 2:
        # 3 CNOTs
        # Structure: U3(0) U3(1) CX(0,1) U3(0) U3(1) CX(0,1) U3(0) U3(1) CX(0,1) U3(0) U3(1)
        # 12 U3 gates, 3 params each = 36 params
        # Wait, 4 layers of U3.
        # Layer 0: U3(0), U3(1)
        # CX(0,1)
        # Layer 1: U3(0), U3(1)
        # CX(0,1)
        # Layer 2: U3(0), U3(1)
        # CX(0,1)
        # Layer 3: U3(0), U3(1)
        
        def circuit_2q(params):
            gates = []
            idx = 0
            for layer in range(4):
                for q in range(2):
                    gates.append(['u3', q, params[idx], params[idx+1], params[idx+2]])
                    idx += 3
                if layer < 3:
                    gates.append(['cx', 0, 1])
            return gates

        def obj_2q(params):
            if time.time() - start_time > TIME_LIMIT:
                return 1.0
            gates = circuit_2q(params)
            return _distance(target, _build(gates, 2))

        x0 = np.zeros(36)
        # Use Nelder-Mead
        from scipy.optimize import minimize
        res = minimize(obj_2q, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() - start_time > TIME_LIMIT:
             # Return best found so far? minimize might not return best if killed.
             # We assume it finishes or we check res.
             pass
        return circuit_2q(res.x)

    elif qubits == 3:
        # 15 CNOTs
        # Structure: 16 layers of U3 (3 qubits each)
        # CNOTs: 5 blocks of 3 CNOTs (0,1), (1,2), (0,2)
        # Total params: 16 * 3 * 3 = 144
        
        def circuit_3q(params):
            gates = []
            idx = 0
            cnot_seq = [(0,1), (1,2), (0,2)]
            for layer in range(16):
                for q in range(3):
                    gates.append(['u3', q, params[idx], params[idx+1], params[idx+2]])
                    idx += 3
                if layer < 15:
                    # Add CNOTs based on layer index
                    # We need 15 CNOTs total.
                    # 5 blocks of 3.
                    # So in layers 0, 3, 6, 9, 12 we add 3 CNOTs?
                    # No, the structure is:
                    # U3s, CNOTs, U3s, CNOTs...
                    # Let's place CNOTs after every 3rd layer of U3s?
                    # 16 layers of U3.
                    # If we put 3 CNOTs after layer 2, 5, 8, 11, 14?
                    # That's 5 * 3 = 15 CNOTs.
                    # Layers: 0,1,2 -> CNOTs -> 3,4,5 -> CNOTs -> 6,7,8 -> CNOTs -> 9,10,11 -> CNOTs -> 12,13,14 -> CNOTs -> 15
                    if layer in [2, 5, 8, 11, 14]:
                        for c, t in cnot_seq:
                            gates.append(['cx', c, t])
            return gates

        def obj_3q(params):
            if time.time() - start_time > TIME_LIMIT:
                return 1.0
            gates = circuit_3q(params)
            return _distance(target, _build(gates, 3))

        x0 = np.zeros(144)
        from scipy.optimize import minimize
        res = minimize(obj_3q, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        return circuit_3q(res.x)
        
    return []

def synthesize(problem):
    try:
        gates = _identify_and_synth(problem)
        # Validate gate count
        if len(gates) > 400:
            return []
        cnot_count = sum(1 for g in gates if g[0] == 'cx')
        if cnot_count > problem["cnot_budget"]:
            return []
        return gates
    except Exception:
        return []
# EVOLVE-BLOCK-END
