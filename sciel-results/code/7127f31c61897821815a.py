"""Analytical synthesis for standard gates + optimized search for generic unitaries."""
import time
import math
import random

import numpy as np

_DEADLINE = 250.0
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


def _cnot_matrix(control, target, qubits):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _solve_sandwich(U, qubits, c, t):
    """
    Solve U = A CNOT B for single-qubit A, B.
    Returns (A_params, B_params) where params are (theta, phi, lam).
    """
    C = _cnot_matrix(c, t, qubits)
    # We need U = A C B => B = C^{-1} A^{-1} U => B = C A^{-1} U
    # This is not linear in A.
    # Instead, use the structure:
    # For 2 qubits, let A = A0 ⊗ A1 (if c=0, t=1) or A1 ⊗ A0 (if c=1, t=0)
    # Actually, A is a product of single-qubit gates.
    # Let's try a direct ansatz for 2-qubit gates.
    
    if qubits == 2:
        # U is 4x4.
        # CNOT(0,1) structure:
        # C = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        # If c=0, t=1:
        # A = A0 ⊗ A1, B = B0 ⊗ B1
        # U = (A0⊗A1) C (B0⊗B1)
        # C = |0><0|⊗I + |1><1|⊗X
        # C (B0⊗B1) = |0><0|⊗B1 + |1><1|⊗(X B1)
        # (A0⊗A1) [ ... ] = A0|0><0|A1⊗B1 + A0|1><1|A1⊗(X B1)
        # This is getting complicated. Let's use a known decomposition method.
        # For any 2-qubit gate with CNOT count 1, it must be of the form A CNOT B.
        # We can find A and B by:
        # 1. Pick a basis state |i> such that CNOT maps it to |j>.
        # 2. U|i> = A|j> => A = U|i> <j| ... no, A is unitary.
        
        # Alternative: Use the fact that we can always decompose any 2-qubit gate into CNOTs.
        # But we want exactly 1 CNOT for iSWAP.
        # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
        # Let's verify if iSWAP = A CNOT B.
        # CNOT(0,1) = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        # Let A = a0 ⊗ a1, B = b0 ⊗ b1.
        # iSWAP = (a0⊗a1) CNOT (b0⊗b1)
        # Column 0 of iSWAP: [1,0,0,0]^T.
        # CNOT (b0⊗b1) col 0: b0[0] * (b1[0], b1[1], 0, 0)^T = (b0[0]b1[0], b0[0]b1[1], 0, 0)^T
        # (a0⊗a1) * that:
        # Row 0: a0[0,0]*b0[0]b1[0] * a1[0,0] + a0[0,1]*0 * a1[1,0] ...
        # This is too tedious manually. Let's use a numerical solver for the parameters.
        
        pass
        
    # Fallback: Just return a reasonable guess or None.
    return None


def _iswap_gates_exact():
    """
    iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    Decomposition:
    iSWAP = (Rz(π/4) ⊗ Rz(-π/4)) · (Ry(π/4) ⊗ Ry(-π/4)) · CNOT(0,1) · (Ry(-π/4) ⊗ Ry(π/4)) · (Rz(-π/4) ⊗ Rz(π/4))
    Wait, that's 1 CNOT but many single-qubit gates.
    Let's verify this specific decomposition.
    """
    # Let's try a different known 1-CNOT decomposition for iSWAP.
    # From literature: iSWAP = (H ⊗ H) CNOT (Ry(π/4) ⊗ Ry(-π/4)) CNOT (H ⊗ H) -- this is 2 CNOTs.
    # Actually, iSWAP can be done with 1 CNOT.
    # Reference: "Synthesis of quantum logic circuits"
    # iSWAP = (Ry(π/4) ⊗ Ry(-π/4)) CNOT (Ry(-π/4) ⊗ Ry(π/4)) with phase corrections?
    
    # Let's just use a verified circuit.
    # iSWAP = 
    # |00> -> |00>
    # |01> -> i|10>
    # |10> -> i|01>
    # |11> -> |11>
    
    # Try:
    # 1. Ry(π/4) on q0, Ry(-π/4) on q1
    # 2. CNOT(0,1)
    # 3. Ry(-π/4) on q0, Ry(π/4) on q1
    # 4. Phase corrections: S on q0? S on q1?
    
    s = 1.0 / math.sqrt(2)
    # Let's construct it numerically and solve for phases.
    # For now, return a circuit that is close.
    gates = [
        ["u3", 0, math.pi / 4, 0, 0],   # Ry(π/4)
        ["u3", 1, -math.pi / 4, 0, 0],  # Ry(-π/4)
        ["cx", 0, 1],
        ["u3", 0, -math.pi / 4, 0, 0],  # Ry(-π/4)
        ["u3", 1, math.pi / 4, 0, 0],   # Ry(π/4)
        ["u3", 0, 0, 0, math.pi / 2],   # S on q0
        ["u3", 1, 0, 0, -math.pi / 2],  # S† on q1
    ]
    return gates


def _toffoli_gates_exact():
    """
    Toffoli with 6 CNOTs.
    Standard decomposition from Barenco et al.
    """
    t = math.pi / 4
    gates = [
        ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
        ["cx", 1, 2],                         # CNOT(1,2)
        ["u3", 2, 0, 0, -t],                 # T† on q2
        ["cx", 0, 2],                         # CNOT(0,2)
        ["u3", 2, 0, 0, t],                  # T on q2
        ["cx", 1, 2],                         # CNOT(1,2)
        ["u3", 2, 0, 0, -t],                 # T† on q2
        ["cx", 0, 2],                         # CNOT(0,2)
        ["u3", 2, 0, 0, t],                  # T on q2
        ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
        ["cx", 0, 1],                         # CNOT(0,1)
        ["u3", 1, 0, 0, t],                  # T on q1
        ["u3", 0, 0, 0, -t],                 # T† on q0
        ["cx", 0, 1],                         # CNOT(0,1)
        ["u3", 0, 0, 0, t],                  # T on q0
        ["u3", 1, 0, 0, t],                  # T on q1
    ]
    return gates


def _known_decompositions(problem):
    """Try to return known exact decompositions for standard gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # CNOT
    if name == "CNOT" and qubits == 2 and budget >= 1:
        gates = [["cx", 0, 1]]
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates
        # Try reversed CNOT if qubit order is different
        gates = [["cx", 1, 0]]
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates
        return None
    
    # SWAP: 3 CNOTs
    if name == "SWAP" and qubits == 2 and budget >= 3:
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates
        return None
    
    # iSWAP: 1 CNOT
    if name == "iSWAP" and qubits == 2 and budget >= 1:
        gates = _iswap_gates_exact()
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates
        # Try to fix iSWAP with a quick local search
        # For now, return None to fall back to optimization
        return None
    
    # Toffoli: 6 CNOTs
    if name == "Toffoli" and qubits == 3 and budget >= 6:
        gates = _toffoli_gates_exact()
        if _distance(target, _build_fast(gates, qubits)) < 1e-8:
            return gates
        return None
    
    return None


def _build_gates(cnots_layout, values):
    """Build gate list from CNOT layout and parameter values."""
    gates = []
    k = 0
    for gate in cnots_layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _nelder_mead(target, qubits, cnots_layout, initial_values, time_limit, max_iter=200):
    """Nelder-Mead optimization on single-qubit parameters."""
    n = len(initial_values)
    if n == 0:
        return list(initial_values), _distance(target, _build_fast(_build_gates(cnots_layout, initial_values), qubits))
    
    # Initialize simplex
    simplex = [list(initial_values)]
    for i in range(n):
        perturbed = list(initial_values)
        step = 0.1 if abs(perturbed[i]) < 1e-6 else 0.1 * abs(perturbed[i])
        perturbed[i] += step
        simplex.append(perturbed)
    
    def objective(vals):
        gates = _build_gates(cnots_layout, vals)
        return _distance(target, _build_fast(gates, qubits))
    
    # Evaluate initial simplex
    f_values = [objective(s) for s in simplex]
    
    for iteration in range(max_iter):
        if time.time() - _start_time > time_limit:
            break
        
        # Sort simplex by objective value
        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Check convergence
        if f_values[0] < 1e-12:
            break
        if abs(f_values[0] - f_values[-1]) < 1e-12:
            break
        
        # Centroid of all but worst point
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        alpha = 1.0
        reflected = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
        f_reflected = objective(reflected)
        
        if f_values[0] <= f_reflected < f_values[n - 1]:
            simplex[n] = reflected
            f_values[n] = f_reflected
        elif f_reflected < f_values[0]:
            # Expansion
            gamma = 2.0
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n)]
            f_expanded = objective(expanded)
            if f_expanded < f_reflected:
                simplex[n] = expanded
                f_values[n] = f_expanded
            else:
                simplex[n] = reflected
                f_values[n] = f_reflected
        else:
            # Contraction
            rho = 0.5
            contracted = [centroid[j] + rho * (simplex[n][j] - centroid[j]) for j in range(n)]
            f_contracted = objective(contracted)
            if f_contracted < f_values[n]:
                simplex[n] = contracted
                f_values[n] = f_contracted
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    f_values[i] = objective(simplex[i])
    
    # Find best
    best_idx = min(range(n + 1), key=lambda i: f_values[i])
    return simplex[best_idx], f_values[best_idx]


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
    
    # First try known decompositions
    known = _known_decompositions(problem)
    if known is not None:
        dist = _distance(target, _build_fast(known, qubits))
        if dist < 1e-8:
            return known
    
    # Search approach: generate varied candidate circuits with Nelder-Mead
    best_gates = None
    best_dist = 1.0  # worst case
    
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Try multiple layouts with Nelder-Mead
    num_layouts = 6 if qubits <= 2 else 3
    rng = random.Random(12345 + qubits)
    
    for layout_idx in range(num_layouts):
        if time.time() - _start_time > time_limit:
            break
        
        # Generate a CNOT layout
        layout = []
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        for i in range(budget):
            if i < len(pairs):
                c, t = pairs[i]
            else:
                c, t = rng.choice(pairs)
            layout.append(["cx", c, t])
            if i < budget - 1:
                layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        n_u3 = sum(1 for g in layout if g[0] == "u3")
        n_params = 3 * n_u3
        
        # Try multiple initializations
        for init_idx in range(3):
            if time.time() - _start_time > time_limit:
                break
            
            if init_idx == 0:
                initial = [0.0] * n_params
            elif init_idx == 1:
                initial = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            else:
                initial = [rng.uniform(-0.5, 0.5) for _ in range(n_params)]
            
            values, dist = _nelder_mead(target, qubits, layout, initial, time_limit, max_iter=150)
            
            if dist < best_dist:
                best_dist = dist
                best_gates = _build_gates(layout, values)
    
    if best_gates is None:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates
