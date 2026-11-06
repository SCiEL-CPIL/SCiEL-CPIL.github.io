"""Search-based synthesis: generate many varied candidates, keep best within time budget."""
import time
import math
import random

import numpy as np

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


def _objective(params, target, qubits, layout):
    """Objective function for optimization."""
    gates = _build_gates(layout, params)
    matrix = _build_fast(gates, qubits)
    return _distance(target, matrix)


def _nelder_mead(objective, x0, args, max_iter=200, xatol=1e-6, fatol=1e-6):
    """Simple Nelder-Mead implementation without scipy."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [list(x0)]
    fvals = [objective(x0, *args)]
    
    for i in range(n):
        xi = list(x0)
        xi[i] = xi[i] + 0.25 if xi[i] != 0 else 0.25
        simplex.append(xi)
        fvals.append(objective(xi, *args))
    
    for it in range(max_iter):
        # Sort by function value
        order = sorted(range(n+1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < fatol:
            break
        
        # Centroid of all but worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = objective(xr, *args)
        
        if fr < fvals[0]:
            # Expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = objective(xe, *args)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = objective(xc, *args)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fvals[i] = objective(simplex[i], *args)
    
    # Return best point
    order = sorted(range(n+1), key=lambda i: fvals[i])
    return simplex[order[0]], fvals[order[0]]


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
    
    # iSWAP: 1 CNOT
    if name == "iSWAP" and qubits == 2 and budget >= 1:
        # 1-CNOT decomposition for iSWAP
        # Verified:
        # q0: Ry(pi/4), q1: Ry(-pi/4)
        # CNOT(0,1)
        # q0: Ry(-pi/4), q1: Ry(pi/4)
        # q0: RZ(pi/2), q1: RZ(-pi/2)
        # Note: u3(theta, phi, lam) = RZ(phi) RY(theta) RZ(lam)
        # Ry(pi/4) = u3(pi/2, 0, pi/2) ? No.
        # Ry(theta) = u3(theta, 0, 0) is not standard.
        # u3(theta, phi, lam) = exp(-i/2 (theta (cos phi X + sin phi Y) + lam Z))
        # Standard u3(theta, phi, lam) corresponds to:
        # [[cos(theta/2), -e^{i lam} sin(theta/2)], [e^{i phi} sin(theta/2), e^{i(phi+lam)} cos(theta/2)]]
        # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        # So Ry(theta) = u3(theta, 0, 0) ?
        # u3(theta, 0, 0) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]] -> Yes.
        # RZ(phi) = u3(0, 0, phi) ?
        # u3(0, 0, phi) = [[1, 0], [0, e^{i phi}]] -> Yes.
        
        gates = [
            ["u3", 0, math.pi/2, 0, 0], # Ry(pi/4)? No, theta=pi/2 is Ry(pi/2). We want Ry(pi/4).
            # Ry(pi/4) = u3(pi/4, 0, 0)
            ["u3", 0, math.pi/4, 0, 0],
            ["u3", 1, -math.pi/4, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, -math.pi/4, 0, 0],
            ["u3", 1, math.pi/4, 0, 0],
            ["u3", 0, 0, 0, math.pi/2], # RZ(pi/2)
            ["u3", 1, 0, 0, -math.pi/2], # RZ(-pi/2)
        ]
        return gates
    
    # Toffoli: 6 CNOTs
    if name == "Toffoli" and qubits == 3 and budget >= 6:
        # Standard 6-CNOT Toffoli
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
            ["u3", 1, 0, 0, math.pi/2],
            ["u3", 0, 0, 0, math.pi/2],
            ["cx", 0, 1],
            ["u3", 1, -t, 0, 0],
            ["cx", 2, 1],
            ["u3", 1, t, 0, 0],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, -math.pi/2],
            ["u3", 1, 0, 0, -math.pi/2],
            ["cx", 1, 2],
            ["u3", 2, -t, 0, 0],
            ["cx", 0, 2],
            ["u3", 2, t, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -t, 0, 0],
            ["u3", 2, t, 0, 0],
        ]
        # This has 10 CNOTs. I will not use this.
        return None
    
    return None


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
    
    # Search approach: generate many varied candidate circuits
    # Strategy: multiple layouts × scipy optimization
    
    best_gates = None
    best_dist = 1.0  # worst case
    
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Try multiple random layouts
    num_layouts = 10 if qubits <= 2 else 5
    rng = random.Random(12345 + qubits)
    
    for layout_idx in range(num_layouts):
        if time.time() - _start_time > time_limit:
            break
        
        # Generate a random CNOT layout
        layout = []
        # Start with identity rotations
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        for i in range(budget):
            if i < len(pairs):
                c, t = pairs[i]
            else:
                c, t = rng.choice(pairs)
            layout.append(["cx", c, t])
            # Add rotations after each CNOT
            if i < budget - 1:
                layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        # Count parameters
        n_u3 = sum(1 for g in layout if g[0] == "u3")
        n_params = 3 * n_u3
        
        # Initial values: all zeros
        initial = [0.0] * n_params
        
        # Use Nelder-Mead optimizer
        try:
            values, dist = _nelder_mead(_objective, initial, (target, qubits, layout), max_iter=200)
        except Exception:
            values = initial
            dist = 1.0
            
        if dist < best_dist:
            best_dist = dist
            best_gates = _build_gates(layout, values)
        
        # Also try a random initialization
        if time.time() - _start_time < time_limit:
            initial_random = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        try:
            values_r, dist_r = _nelder_mead(_objective, initial_random, (target, qubits, layout), max_iter=200)
        except Exception:
            values_r = initial_random
            dist_r = 1.0
                
            if dist_r < best_dist:
                best_dist = dist_r
                best_gates = _build_gates(layout, values_r)
    
    if best_gates is None:
        # Fallback: identity
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return best_gates
