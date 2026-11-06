"""Exact decompositions for named gates + optimized synthesis for generic unitaries."""
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


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _build(gates, qubits):
    """Build the full unitary from a gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
            ], dtype=complex)
            # Apply single-qubit gate on qubit q
            # Build operator by kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target when control is 1
            # Permute rows
            index = np.arange(dim)
            ctrl_bit = (index >> (qubits - 1 - control)) & 1
            tgt_bit = (index >> (qubits - 1 - target)) & 1
            # For rows where control=1, flip target bit
            new_index = index.copy()
            mask = ctrl_bit == 1
            new_index[mask] = index[mask] ^ (1 << (qubits - 1 - target))
            matrix = matrix[new_index, :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dagger @ matrix)| / dim"""
    dim = target.shape[0]
    tr_val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr_val) / dim)


def _exact_cnot_2qubit():
    """CNOT gate: 1 CNOT."""
    return [_cx(0, 1)]


def _exact_iswap_2qubit():
    """iSWAP: iSWAP = (1/2)[[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]] * i
    Actually iSWAP = [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]]
    Standard decomposition: iSWAP can be done with 1 CNOT + single-qubit gates.
    iSWAP = (H ⊗ H) * iSWAP_ideal... let me use a known decomposition.
    
    iSWAP = [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]]
    
    Known: iSWAP = diag(1,i) on both qubits * (1/2)(I + i*XX) ... 
    Actually a simpler approach:
    iSWAP = (S ⊗ S) * (1/2)(I + i*XX) where XX = X⊗X
    
    Let me use the standard 1-CNOT decomposition:
    iSWAP = (H⊗H) * CNOT(0,1) * (Rz(pi/4)⊗Rz(-pi/4)) * CNOT(0,1) * (H⊗H)
    
    Actually, let me just use a known exact decomposition.
    
    iSWAP = [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]]
    
    One known decomposition with 1 CNOT:
    Apply: 
    - u3 on q0: theta=pi/2, phi=0, lam=pi/2  (this is S gate... let me think more carefully)
    
    Let me use a different approach. The iSWAP gate:
    iSWAP = i * (1/2) * [[0,0,0,0],[0,0,1,1],[0,1,1,0],[0,0,0,0]] ... no.
    
    iSWAP matrix:
    |00> -> 0
    |01> -> i|10>
    |10> -> i|01>
    |11> -> 0
    
    So iSWAP|01> = i|10>, iSWAP|10> = i|01>
    
    This is equivalent to: (S ⊗ S) * SWAP * (S† ⊗ S†) ... let me check.
    S = diag(1, i)
    S⊗S = diag(1, i, i, -1)
    SWAP = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
    (S⊗S) SWAP (S†⊗S†):
    First SWAP maps |01>->|10>, |10>->|01>
    Then S†⊗S† = diag(1, -i, -i, 1)
    Then S⊗S = diag(1, i, i, -1)
    
    For |01>: SWAP gives |10>, S†⊗S† gives -i|10>, S⊗S gives i*(-i)|10> = |10>... that's not right.
    
    Let me just use a direct construction.
    iSWAP = (1/2) * [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]] * 2 = [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]]
    
    Actually the standard iSWAP is:
    iSWAP = [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]]
    
    A known 1-CNOT decomposition:
    iSWAP = (Rz(pi/4) ⊗ Rz(-pi/4)) * CNOT(0,1) * (Rz(-pi/4) ⊗ Rz(pi/4)) * CNOT(0,1)
    
    Hmm, that uses 2 CNOTs. Let me think again.
    
    Actually, for a 2-qubit gate with 1 CNOT budget, we can write:
    U = (A ⊗ B) * CNOT * (C ⊗ D)
    
    For iSWAP, this is possible. Let me look up the exact parameters.
    
    iSWAP = (1/2)[[0,0,0,0],[0,0,2,2i],[0,2,2i,0],[0,0,0,0]] ... no.
    
    Let me just use the fact that:
    iSWAP = (S ⊗ S) * (1/2)(I + i*XX)
    where XX = X⊗X = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
    
    (1/2)(I + i*XX) = (1/2)([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] + i*[[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
    = (1/2)[[1,0,0,i],[0,1,i,0],[0,i,1,0],[i,0,0,1]]
    
    (S⊗S) * that = diag(1,i,i,-1) * (1/2)[[1,0,0,i],[0,1,i,0],[0,i,1,0],[i,0,0,1]]
    = (1/2)[[1,0,0,i],[0,i,i,0],[0,-i,1,0],[-i,0,0,-1]]
    
    That doesn't look like iSWAP. Let me try a different approach.
    
    I'll use a known decomposition. The iSWAP gate can be decomposed as:
    iSWAP = (H ⊗ H) * CNOT(0,1) * (Rz(π/4) ⊗ Rz(π/4)) * CNOT(0,1) * (H ⊗ H)
    
    Wait, that's 2 CNOTs. For 1 CNOT, I need a different approach.
    
    Actually, let me reconsider. The CNOT budget for iSWAP might be 2, not 1. Let me check the problem statement - it says "within the CNOT counts known to suffice for each." For iSWAP, the known minimum is 1 CNOT (it's a 2-qubit gate with 1 CNOT equivalent).
    
    For a 2-qubit gate with 1 CNOT: U = (A⊗B) CNOT (C⊗D)
    
    Let me just compute the decomposition numerically for each named gate and hardcode it.
    """
    # I'll compute this numerically in the synthesize function for now
    # For iSWAP with 1 CNOT:
    # U = (A⊗B) CNOT (C⊗D)
    # CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    # We need A⊗B = U CNOT (C⊗D)†
    # This is a standard 2-qubit decomposition problem.
    # For now, let me use a known exact decomposition.
    
    # iSWAP decomposition with 1 CNOT (from literature):
    # iSWAP = (S ⊗ S) * (1/2)(I + i XX)
    # And (1/2)(I + i XX) can be written with 1 CNOT.
    # (1/2)(I + i XX) = (H⊗H) CNOT (H⊗H) ... let me verify.
    # H⊗H CNOT (H⊗H):
    # H = (1/sqrt(2))[[1,1],[1,-1]]
    # H⊗H = (1/2)[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    # CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    # H⊗H * CNOT = (1/2)[[1,1,0,1],[1,-1,0,-1],[1,1,1,-1],[1,-1,1,1]]... this is getting complex.
    
    # Let me just use a numerical approach in synthesize for all gates.
    # For the named gates, I'll provide known exact decompositions.
    
    # CNOT: just 1 CNOT
    # SWAP: 3 CNOTs
    # iSWAP: 1 CNOT (need to find exact single-qubit params)
    # Toffoli: 6 CNOTs (standard)
    
    # For now, return a placeholder and compute in synthesize
    raise NotImplementedError("Use synthesize for actual decomposition")


def _synthesize_generic(problem, qubits, budget, target):
    """Optimize single-qubit rotations around a fixed CNOT ladder."""
    dim = 2 ** qubits
    
    # Build CNOT ladder: use a fixed pattern
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Layout: alternating single-qubit rotations and CNOTs
    layout = []
    # Initial layer of single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    cnot_count = 0
    for layer in range(budget):
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
        cnot_count += 1
        # Add single-qubit gates after each CNOT layer
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Final layer of single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # Count u3 parameters
    u3_indices = []
    param_count = 0
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            u3_indices.append(i)
            param_count += 3
    
    # Initialize parameters
    params = np.zeros(param_count)
    
    def build_from_params(p):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], float(p[k]), float(p[k+1]), float(p[k+2])])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def compute_distance(p):
        gates = build_from_params(p)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Coordinate descent with adaptive step sizes
    best_params = params.copy()
    best_dist = compute_distance(best_params)
    
    # Time check
    if _elapsed() > _DEADLINE - 30:
        return build_from_params(best_params)
    
    # Multi-start coordinate descent
    n_starts = 5
    for start in range(n_starts):
        if _elapsed() > _DEADLINE - 30:
            break
        
        # Random initialization
        p = np.random.uniform(-math.pi, math.pi, size=param_count)
        dist = compute_distance(p)
        
        # Coordinate descent
        for sweep in range(8):
            if _elapsed() > _DEADLINE - 30:
                break
            
            step = 1.0 / (1 + sweep)
            improved = False
            for i in range(param_count):
                if _elapsed() > _DEADLINE - 30:
                    break
                
                for trial in (p[i] + step, p[i] - step):
                    new_p = p.copy()
                    new_p[i] = trial
                    new_dist = compute_distance(new_p)
                    if new_dist < dist:
                        p = new_p
                        dist = new_dist
                        improved = True
            
            if not improved:
                break
        
        if dist < best_dist:
            best_dist = dist
            best_params = p.copy()
    
    return build_from_params(best_params)


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    dim = 2 ** qubits
    
    # Check if this is a named gate with known exact decomposition
    name_lower = name.lower() if name else ""
    
    if qubits == 2 and budget == 1:
        # Could be CNOT or iSWAP
        # Check if target is CNOT
        cnot_target = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        # Check phase-insensitive equivalence
        tr_val = np.trace(cnot_target.conj().T @ target)
        if abs(abs(tr_val) / dim - 1.0) < 1e-6:
            return _exact_cnot_2qubit()
        
        # Check if target is iSWAP
        iswap_target = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 1j],
            [0, 1j, 0, 0],
            [0, 0, 0, 0]
        ], dtype=complex)
        
        tr_val = np.trace(iswap_target.conj().T @ target)
        if abs(abs(tr_val) / dim - 1.0) < 1e-6:
            # iSWAP decomposition with 1 CNOT
            # iSWAP = (S⊗S) * (1/2)(I + i*XX)
            # (1/2)(I + i*XX) = (H⊗H) * CNOT * (H⊗H) ... let me verify numerically
            # Actually, let me just use a known decomposition.
            # 
            # iSWAP with 1 CNOT:
            # U = (A⊗B) CNOT (C⊗D)
            # 
            # Known decomposition:
            # iSWAP = (Rz(π/4) ⊗ Rz(-π/4)) * CNOT * (Rz(-π/4) ⊗ Rz(π/4)) * CNOT ... no, that's 2 CNOTs.
            #
            # For 1 CNOT: 
            # iSWAP = (S ⊗ I) * CNOT * (H ⊗ H) ... let me check.
            #
            # Let me just compute it numerically.
            # iSWAP = (A⊗B) CNOT (C⊗D)
            # => A⊗B = iSWAP CNOT (C⊗D)†
            #
            # Choose C⊗D = I, then A⊗B = iSWAP CNOT
            # iSWAP CNOT = [[0,0,0,0],[0,0,1,i],[0,i,0,0],[0,0,0,0]] * [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
            # = [[0,0,0,0],[0,0,0,1],[0,0,i,0],[0,0,0,0]]... this doesn't factor nicely.
            #
            # Let me try C⊗D = (H⊗H):
            # iSWAP CNOT (H⊗H)† = iSWAP CNOT (H⊗H)
            # This is getting complex. Let me just use a numerical search for the single-qubit params.
            
            # For iSWAP with 1 CNOT, use optimization
            return _synthesize_generic(problem, qubits, budget, target)
    
    if qubits == 2 and budget == 3:
        # Could be SWAP
        swap_target = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        tr_val = np.trace(swap_target.conj().T @ target)
        if abs(abs(tr_val) / dim - 1.0) < 1e-6:
            # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
            return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    
    if qubits == 3 and budget == 6:
        # Could be Toffoli
        toffoli_target = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            toffoli_target[i, i] = 1.0
        # Flip |110> -> |111> and |111> -> |110>
        # Actually Toffoli: |11z> -> |11(1-z)>
        toffoli_target[6, 6] = 0  # |110>
        toffoli_target[6, 7] = 1  # |110> -> |111>
        toffoli_target[7, 6] = 1  # |111> -> |110>
        toffoli_target[7, 7] = 0
        
        tr_val = np.trace(toffoli_target.conj().T @ target)
        if abs(abs(tr_val) / dim - 1.0) < 1e-6:
            # Toffoli decomposition with 6 CNOTs
            # Standard decomposition:
            # Toffoli = 
            #   H(2)
            #   CNOT(1,2)
            #   T†(2)
            #   CNOT(0,2)
            #   T(1) T†(2)
            #   CNOT(0,2)
            #   T†(1) T(2)
            #   CNOT(1,2)
            #   H(2)
            #   T(0) T†(1)
            #   CNOT(0,1)
            #   T†(0) T(1)
            #   CNOT(0,1)
            #
            # This uses 6 CNOTs and single-qubit gates.
            # T = diag(1, e^{iπ/4}) = u3(0, 0, 0, π/4)
            # T† = diag(1, e^{-iπ/4}) = u3(0, 0, 0, -π/4)
            # H = u3(π/2, 0, π)
            
            gates = []
            # H on qubit 2
            gates.append(_u3(2, math.pi / 2, 0, math.pi))
            # CNOT(1,2)
            gates.append(_cx(1, 2))
            # T† on qubit 2
            gates.append(_u3(2, 0, 0, -math.pi / 4))
            # CNOT(0,2)
            gates.append(_cx(0, 2))
            # T on qubit 1
            gates.append(_u3(1, 0, 0, math.pi / 4))
            # T† on qubit 2
            gates.append(_u3(2, 0, 0, -math.pi / 4))
            # CNOT(0,2)
            gates.append(_cx(0, 2))
            # T† on qubit 1
            gates.append(_u3(1, 0, 0, -math.pi / 4))
            # T on qubit 2
            gates.append(_u3(2, 0, 0, math.pi / 4))
            # CNOT(1,2)
            gates.append(_cx(1, 2))
            # H on qubit 2
            gates.append(_u3(2, math.pi / 2, 0, math.pi))
            # T on qubit 0
            gates.append(_u3(0, 0, 0, math.pi / 4))
            # T† on qubit 1
            gates.append(_u3(1, 0, 0, -math.pi / 4))
            # CNOT(0,1)
            gates.append(_cx(0, 1))
            # T† on qubit 0
            gates.append(_u3(0, 0, 0, -math.pi / 4))
            # T on qubit 1
            gates.append(_u3(1, 0, 0, math.pi / 4))
            # CNOT(0,1)
            gates.append(_cx(0, 1))
            
            return gates
    
    # Generic unitary: use optimization
    return _synthesize_generic(problem, qubits, budget, target)
# EVOLVE-BLOCK-END
