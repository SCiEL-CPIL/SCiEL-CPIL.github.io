"""Direct synthesis: exact decompositions for standard gates + fast Nelder-Mead for generic unitaries."""
import time
import math
import random

import numpy as np

_DEADLINE = 240.0
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


def _iswap_gates_exact():
    """iSWAP on qubits (0,1): |01> -> i|10>, |10> -> i|01>. 1-CNOT decomposition."""
    # iSWAP matrix: [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # Known 1-CNOT decomposition:
    # Apply Ry(-pi/4) on q0, Ry(pi/4) on q1
    # CNOT(0,1)
    # Apply Ry(pi/4) on q0, Ry(-pi/4) on q1
    # Then add phase i to the |01> and |10> states via u3 gates
    # Actually, let me construct this properly.
    # iSWAP = (Ry(pi/4)_0 ⊗ Ry(-pi/4)_1) · CNOT(0,1) · (Ry(-pi/4)_0 ⊗ Ry(pi/4)_1) · CNOT(0,1)
    # Wait, that's 2 CNOTs. For 1 CNOT:
    # iSWAP can be decomposed as:
    # U1 · CNOT · U0 where U0, U1 are single-qubit gates
    # Let me just use a verified decomposition.
    # After some research, iSWAP with 1 CNOT:
    # q0: u3(pi/2, 0, -pi/2)  # This is like a phase gate
    # q1: u3(pi/2, 0, pi/2)
    # CNOT(0,1)
    # q0: u3(pi/2, 0, -pi/2)
    # q1: u3(pi/2, 0, pi/2)
    # Hmm, let me just compute it numerically.
    # Actually, the simplest approach: use the known fact that
    # iSWAP = diag(1, i, -i, 1) · SWAP
    # And SWAP needs 3 CNOTs, but iSWAP needs only 1 CNOT + single-qubit gates.
    # Let me use this decomposition:
    # iSWAP = (H ⊗ I) · CNOT · (Ry(π/4) ⊗ Ry(-π/4)) · CNOT · (H ⊗ I)
    # No, that's 2 CNOTs.
    # For 1 CNOT, I'll use:
    # iSWAP = U1 · CNOT(0,1) · U0
    # where U0 = Ry(-π/4)_0 ⊗ Ry(π/4)_1
    # and U1 = Ry(π/4)_0 ⊗ Ry(-π/4)_1
    # But this gives SWAP-like behavior without the i phase.
    # Let me just search for it numerically with the optimizer.
    # For now, return a layout that the optimizer can refine.
    return None


def _toffoli_gates_exact():
    """Toffoli gate with 6 CNOTs. Controls: q0, q1; Target: q2."""
    # Standard 6-CNOT Toffoli decomposition
    # This is a well-known decomposition. Let me use the one from Nielsen & Chuang.
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
        ["u3", 0, math.pi / 2, 0, 0],
        ["u3", 1, math.pi / 2, 0, 0],
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
    # Count CNOTs: positions 1,3,5,7,11,13,15,19,21,23,25 = 11 CNOTs. Too many!
    # I need the 6-CNOT version. Let me use the correct one.
    # The 6-CNOT Toffoli uses a different structure.
    # Actually, the minimal CNOT count for Toffoli is 6, but the decomposition is non-trivial.
    # Let me just return None and let the optimizer find it.
    return None


def _known_decompositions(problem):
    """Try to return known exact decompositions for standard gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # CNOT
    if name == "CNOT" and qubits == 2 and budget >= 1:
        return [["cx", 0, 1]]
    
    # SWAP: 3 CNOTs
    if name == "SWAP" and qubits == 2 and budget >= 3:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    
    # iSWAP: 1 CNOT - let the optimizer handle it
    # Toffoli: 6 CNOTs - let the optimizer handle it
    
    return None


def _build_gates_from_layout(layout, values):
    """Build gate list from CNOT layout and parameter values."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _make_objective(target, qubits, layout):
    """Create objective function for Nelder-Mead."""
    def objective(values):
        gates = _build_gates_from_layout(layout, values)
        if len(gates) > 400:
            return 1.0
        matrix = _build_fast(gates, qubits)
        return _distance(target, matrix)
    return objective


def _nelder_mead(objective, x0, max_iter=500, tol=1e-10, time_limit=None):
    """Fast Nelder-Mead optimizer."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] += 0.5 if point[i] == 0 else point[i] * 1.05
        simplex.append(point)
    
    f_values = [objective(p) for p in simplex]
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - _start_time > time_limit:
            break
        
        # Sort by function value
        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        if f_values[0] < tol:
            break
        
        # Centroid of all but worst point
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        alpha = 1.0
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = objective(xr)
        
        if fr < f_values[0]:
            # Expansion
            gamma = 2.0
            xe = [centroid[j] + gamma * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = objective(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    f_values[i] = objective(simplex[i])
    
    order = sorted(range(n + 1), key=lambda i: f_values[i])
    best_point = simplex[order[0]]
    best_val = f_values[order[0]]
    return best_point, best_val


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    dim = 2 ** qubits
    time_limit = _DEADLINE - 5.0
    
    # First try known decompositions
    known = _known_decompositions(problem)
    if known is not None:
        dist = _distance(target, _build_fast(known, qubits))
        if dist < 1e-8:
            return known
    
    # For iSWAP and Toffoli, use specific layouts
    if name == "iSWAP" and qubits == 2:
        # Layout with 1 CNOT
        layout = [
            ["u3", 0, 0.0, 0.0, 0.0],
            ["u3", 1, 0.0, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, 0.0, 0.0, 0.0],
            ["u3", 1, 0.0, 0.0, 0.0],
        ]
    elif name == "Toffoli" and qubits == 3:
        # Layout with 6 CNOTs
        layout = []
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        for i in range(6):
            if i == 0:
                layout.append(["cx", 1, 2])
            elif i == 1:
                layout.append(["cx", 0, 2])
            elif i == 2:
                layout.append(["cx", 1, 2])
            elif i == 3:
                layout.append(["cx", 0, 2])
            elif i == 4:
                layout.append(["cx", 0, 1])
            else:
                layout.append(["cx", 2, 1])
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    else:
        # Generic layout
        pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
        layout = []
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        for i in range(budget):
            c, t = pairs[i % len(pairs)]
            layout.append(["cx", c, t])
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Initial values: all zeros
    initial = [0.0] * n_params
    
    # Try multiple random initializations
    rng = random.Random(42)
    best_gates = None
    best_dist = 1.0
    
    num_starts = 10 if qubits <= 2 else 5
    
    for start_idx in range(num_starts):
        if time.time() - _start_time > time_limit:
            break
        
        if start_idx == 0:
            x0 = initial[:]
        else:
            x0 = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        objective = _make_objective(target, qubits, layout)
        values, dist = _nelder_mead(objective, x0, max_iter=300, tol=1e-12, time_limit=time_limit)
        
        if dist < best_dist:
            best_dist = dist
            best_gates = _build_gates_from_layout(layout, values)
        
        if best_dist < 1e-10:
            break
    
    if best_gates is None:
        # Fallback: identity
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates
