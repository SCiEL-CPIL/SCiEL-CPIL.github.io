"""Quantum circuit synthesis: exact decompositions + Nelder-Mead for generic gates."""
import math
import time

import numpy as np

_DEADLINE = 230.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _build(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator via kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: swap rows where control bit is 1
            idx = np.arange(dim)
            cbit = (idx >> (qubits - 1 - control)) & 1
            tshift = 1 << (qubits - 1 - target)
            # For each row index, if control bit is 1, flip target bit
            new_idx = idx.copy()
            mask = cbit == 1
            new_idx[mask] = idx[mask] ^ tshift
            matrix = matrix[new_idx, :]
    return matrix


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _exact_cnot(gates):
    """Return gates for a single CNOT on qubits 0,1."""
    return [["cx", 0, 1]]


def _exact_iswap(gates):
    """iSWAP on qubits 0,1: |01><10| + |10><01| (swap with i phase).
    iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    Decomposition: H on both, then CNOT 0->1, CNOT 1->0, H on both, then phase gates.
    Actually iSWAP = (H⊗H) · CNOT(1,0) · CNOT(0,1) · (H⊗H) · (S⊗S)
    Let me use a known decomposition:
    iSWAP = (Rz(π/2)⊗Rz(π/2)) · (H⊗H) · CNOT(0,1) · CNOT(1,0) · (H⊗H)
    Actually let me just use: iSWAP can be done with 2 CNOTs and single-qubit gates.
    
    iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    
    A known 2-CNOT decomposition:
    Apply S to q0, S to q1, H to q0, H to q1, CNOT(0,1), CNOT(1,0), H to q0, H to q1
    Then adjust phases. Let me just use a standard one.
    
    iSWAP = (Rz(π/2) ⊗ Rz(π/2)) · (H ⊗ H) · CNOT(0,1) · CNOT(1,0) · (H ⊗ H)
    """
    # u3(theta, phi, lam) = exp(-i(theta/2)(...))
    # H = u3(π/2, 0, π)
    # S = u3(π/2, 0, π/2)  -- actually S = diag(1, i) = u3(π/2, 0, π/2)? Let's check.
    # u3(π/2, 0, π/2) = [[cos(π/4), -e^{iπ/2}sin(π/4)], [0, e^{iπ/2}cos(π/4)]]
    # = [[1/√2, -i/√2], [0, i/√2]] -- that's not S.
    # S = [[1,0],[0,i]]. In u3: theta=π, phi=0, lam=π/2 gives [[0, -e^{iπ/2}], [e^{i0}, e^{iπ/2}]] = [[0,-i],[1,i]] -- no.
    # Let me just use: S = u3(π/2, 0, π/2) doesn't work directly.
    # Actually for our purposes, let me just compute the correct u3 parameters.
    # For S = diag(1, i): we need c=1, s=0 → theta=0, then it's identity. That's wrong.
    # S = [[1,0],[0,i]]. u3(θ,φ,λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
    # For S: cos(θ/2)=1, sin(θ/2)=0 → θ=0. Then it's identity. So S can't be done with θ=0.
    # We need: c=1, s=0 doesn't give i. Let's use θ=π: c=0, s=1.
    # u3(π, 0, π/2) = [[0, -e^{iπ/2}], [1, e^{iπ/2}]] = [[0,-i],[1,i]] -- not S.
    # u3(π, 0, 0) = [[0,-1],[1,0]] -- that's X-like.
    # For S=diag(1,i): we need [[1,0],[0,i]].
    # c=1, s=0 won't work. Let's try: c=cos(θ/2)=1/√2, s=sin(θ/2)=1/√2, θ=π/2.
    # u3(π/2, φ, λ) = [[1/√2, -e^{iλ}/√2], [e^{iφ}/√2, e^{i(φ+λ)}/√2]]
    # For this to equal [[1,0],[0,i]]: 1/√2=1? No.
    # S is a phase gate, not a rotation. It can be written as e^{iπ/4}·u3(π/2, 0, π/2)? 
    # u3(π/2, 0, π/2) = [[1/√2, -i/√2], [0, i/√2]]. Multiply by e^{iπ/4}: [[e^{iπ/4}/√2, -ie^{iπ/4}/√2], [0, ie^{iπ/4}/√2]]
    # = [[(1+i)/2, -(i+1)/2·i... ]] this is getting complicated.
    # Let me just use a simpler approach: use the known fact that iSWAP needs 2 CNOTs.
    # I'll just return a placeholder and let the optimizer handle it, OR use a direct construction.
    
    # Actually, for the task, the targets are fixed known gates. Let me just return empty 
    # and let the general optimizer handle all cases. The exact decompositions are complex 
    # to get right without verification, so I'll focus on a robust general approach.
    return []


def _exact_swap(gates):
    """SWAP on qubits 0,1 = CNOT(0,1)·CNOT(1,0)·CNOT(0,1)."""
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _exact_toffoli(gates):
    """Toffoli on 3 qubits. Standard decomposition uses 6 CNOTs."""
    # Toffoli(0,1,2) where 0,1 are controls and 2 is target
    # Standard 6-CNOT decomposition
    H = lambda q: ["u3", q, math.pi / 2, 0, math.pi]
    S = lambda q: ["u3", q, math.pi / 2, 0, math.pi / 2]  # approximate, may need adjustment
    T = lambda q: ["u3", q, math.pi / 4, 0, math.pi / 4]  # approximate
    
    # This is a known decomposition but getting exact u3 params is tricky.
    # Let me just return a 6-CNOT structure and let optimizer tune.
    return []


def _nelder_mead(f, x0, max_iter=200, tol=1e-10, time_limit=None):
    """Nelder-Mead optimization."""
    n = len(x0)
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        step = 0.5 if pt[i] == 0 else 0.5 * abs(pt[i])
        pt[i] += step
        simplex.append(pt)
    
    values = [f(pt) for pt in simplex]
    
    start = time.time()
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start) > time_limit:
            break
        # Sort simplex
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if abs(values[-1] - values[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - np.array(simplex[-1]))
        fr = f(xr)
        
        if fr < values[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - np.array(simplex[-1]))
            fe = f(xe)
            if fe < fr:
                simplex[-1], values[-1] = xe, fe
            else:
                simplex[-1], values[-1] = xr, fr
        elif fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (np.array(simplex[-1]) - centroid)
            fc = f(xc)
            if fc < values[-1]:
                simplex[-1], values[-1] = xc, fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (np.array(simplex[i]) - np.array(simplex[0]))
                    values[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        # Not enough time, return identity
        return []
    
    # Try exact decompositions for known gates
    if qubits == 2 and budget >= 2:
        if "iswap" in name.lower() or "iSWAP" in name:
            # iSWAP: 2 CNOTs
            # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
            # Decomposition: (Rz(π/2)⊗Rz(π/2)) · (H⊗H) · CNOT(0,1) · CNOT(1,0) · (H⊗H)
            # H = u3(π/2, 0, π)
            # Rz(π/2) = u3(0, 0, π/2) -- actually Rz(φ) = u3(0, -φ/2, φ/2)
            # Let me verify: u3(0, φ, -φ) = [[1, 0], [0, e^{i(φ-φ)}]] = [[1,0],[0,1]] -- no.
            # u3(0, 0, λ) = [[1, 0], [0, e^{iλ}]]
            # So Rz(φ) = u3(0, 0, φ) up to global phase.
            # iSWAP = (Rz(π/2)⊗Rz(π/2)) · (H⊗H) · CNOT(0,1) · CNOT(1,0) · (H⊗H)
            gates = []
            gates += [["u3", 0, 0, 0, math.pi/2], ["u3", 1, 0, 0, math.pi/2]]
            gates += [["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi]]
            gates += [["cx", 0, 1], ["cx", 1, 0]]
            gates += [["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi]]
            # Verify
            U = _build(gates, 2)
            d = _distance(target, U)
            if d < 1e-6:
                return gates
            # Try with different phase conventions
            gates2 = []
            gates2 += [["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi]]
            gates2 += [["cx", 0, 1], ["cx", 1, 0]]
            gates2 += [["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi]]
            gates2 += [["u3", 0, 0, 0, math.pi/2], ["u3", 1, 0, 0, math.pi/2]]
            U2 = _build(gates2, 2)
            d2 = _distance(target, U2)
            if d2 < 1e-6:
                return gates2
    
    if qubits == 2 and budget >= 3:
        if "swap" in name.lower() and "iswap" not in name.lower() and "iSWAP" not in name:
            gates = [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
            U = _build(gates, 2)
            d = _distance(target, U)
            if d < 1e-6:
                return gates
    
    if qubits == 1 and budget >= 0:
        if "cnot" in name.lower():
            return []  # CNOT is 2-qubit
    
    # For CNOT target (2 qubits, 1 CNOT budget)
    if qubits == 2 and budget >= 1:
        if "cnot" in name.lower() or "cx" in name.lower():
            gates = [["cx", 0, 1]]
            U = _build(gates, 2)
            d = _distance(target, U)
            if d < 1e-6:
                return gates
            # Try reversed
            gates2 = [["cx", 1, 0]]
            U2 = _build(gates2, 2)
            d2 = _distance(target, U2)
            if d2 < 1e-6:
                return gates2
    
    # For Toffoli (3 qubits)
    if qubits == 3 and budget >= 6:
        if "toffoli" in name.lower() or "ccnot" in name.lower():
            # Standard 6-CNOT Toffoli decomposition
            # Controls: q0, q1; Target: q2
            H = lambda q: ["u3", q, math.pi/2, 0, math.pi]
            # T gate approximation
            T = lambda q: ["u3", q, math.pi/4, 0, math.pi/4]
            Tdag = lambda q: ["u3", q, math.pi/4, 0, -math.pi/4]
            # This is a heuristic; let's just use the optimizer
            pass
    
    # General approach: build a circuit with the CNOT budget and optimize single-qubit params
    # Layout: interleave CNOTs with layers of u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout: CNOTs with u3 layers between
    layout = []
    cnot_count = 0
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
        cnot_count += 1
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Count parameters
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
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
    
    def objective(values):
        gates = circuit(values)
        if len(gates) > 400:
            return 1.0
        U = _build(gates, qubits)
        return _distance(target, U)
    
    # Initial guess: all zeros
    x0 = [0.0] * n_params
    
    # Check time remaining
    remaining = _DEADLINE - (time.time() - _start_time)
    if remaining < 5:
        return circuit(x0)
    
    # Use Nelder-Mead with time limit
    time_limit = remaining - 10
    x_best, f_best = _nelder_mead(objective, x0, max_iter=500, tol=1e-10, time_limit=time_limit)
    
    # Try a few random restarts if time permits
    best_gates = circuit(x_best)
    best_dist = f_best
    
    remaining = _DEADLINE - (time.time() - _start_time)
    if remaining > 20:
        rng = np.random.RandomState(42)
        for restart in range(3):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            x_rand = rng.uniform(-math.pi, math.pi, size=n_params)
            t_lim = (_DEADLINE - 10) - (time.time() - _start_time)
            if t_lim < 5:
                break
            x_r, f_r = _nelder_mead(objective, x_rand, max_iter=200, tol=1e-10, time_limit=t_lim)
            if f_r < best_dist:
                best_dist = f_r
                best_gates = circuit(x_r)
    
    return best_gates
