import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0

def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - control)) & 1
            matrix = matrix[idx ^ (bit << (qubits - 1 - target)), :]
    return matrix

def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)

def _count_cnots(gates):
    return sum(1 for g in gates if g[0] == "cx")

def _detect_known_gate(target, qubits):
    n = target.shape[0]
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"
        iswap = 0.5 * np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"
    if qubits == 3:
        toffoli = np.zeros((8,8), dtype=complex)
        for i in range(8):
            b2 = (i >> 2) & 1
            b1 = (i >> 1) & 1
            if b2 == 1 and b1 == 1:
                toffoli[i, i ^ 1] = 1
            else:
                toffoli[i, i] = 1
        if np.allclose(target, toffoli, atol=1e-10):
            return "toffoli"
    return "generic"

def _matrix_to_u3(M):
    """Extract u3(theta, phi, lam) from a 2x2 unitary matrix M."""
    M = np.array(M, dtype=complex)
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    
    # Handle global phase: u3 has det = e^{i(phi+lam)}
    # We need to find theta, phi, lam such that u3(theta,phi,lam) = M
    # |a|^2 = cos^2(theta/2), |b|^2 = sin^2(theta/2)
    
    abs_a = abs(a)
    abs_b = abs(b)
    
    if abs_a > 1e-12:
        theta = 2.0 * math.acos(min(1.0, max(-1.0, abs_a)))
        c_val = abs_a
        s_val = math.sqrt(max(0.0, 1.0 - c_val*c_val))
        
        if s_val < 1e-12:
            # theta ~ 0
            phi = 0.0
            lam = math.atan2(d.imag, d.real) if abs(d) > 1e-12 else 0.0
            lam = (lam + math.pi) % (2 * math.pi) - math.pi
            return [theta, phi, lam]
        
        # e^{i phi} = c / s_val
        # e^{i lam} = -b / s_val
        phi = math.atan2(c.imag, c.real) - math.atan2(s_val, 0)
        lam = math.atan2(-b.imag, -b.real) - math.atan2(s_val, 0)
        
        # More robust: use the actual values
        if abs(c) > 1e-12:
            phi = math.atan2(c.imag, c.real) - math.atan2(s_val, 0.0)
        elif abs(b) > 1e-12:
            phi = math.atan2(-b.imag, -b.real) + math.atan2(d.imag, d.real) - math.atan2(c_val, 0.0)
        else:
            phi = 0.0
            
        if abs(b) > 1e-12:
            lam = math.atan2(-b.imag, -b.real)
        elif abs(c) > 1e-12:
            lam = math.atan2(d.imag, d.real) - math.atan2(c.imag, c.real)
        else:
            lam = 0.0
            
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        lam = (lam + math.pi) % (2 * math.pi) - math.pi
        return [theta, phi, lam]
    else:
        # theta = pi
        theta = math.pi
        if abs(b) > 1e-12:
            lam = math.atan2(-b.imag, -b.real)
        else:
            lam = 0.0
        if abs(c) > 1e-12:
            phi = math.atan2(c.imag, c.real)
        else:
            phi = 0.0
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        lam = (lam + math.pi) % (2 * math.pi) - math.pi
        return [theta, phi, lam]

def _u3_to_matrix(theta, phi, lam):
    return _u3_matrix(theta, phi, lam)

def _decompose_local_2qubit(V):
    """
    Decompose a 4x4 local unitary V = (A ⊗ B) (C ⊗ D)
    into 4 u3 gates: [A0, A1, C0, C1] where A0,A1 act on qubit 0, C1,C0 act on qubit 1.
    Actually, we want to find A, B, C, D such that V = (A ⊗ B)(C ⊗ D).
    We can absorb: let P = AC, Q = BD, then V = P ⊗ Q.
    So we need to decompose V into P ⊗ Q, then decompose P and Q.
    But P and Q are only determined up to a unitary: if V = P ⊗ Q, then V = (P U) ⊗ (Q U^{-1}) for any unitary U.
    We can fix U = I by choosing a specific decomposition.
    
    Method:
    V = [[V11, V12], [V21, V22]] where each Vij is 2x2.
    V = P ⊗ Q => V11 = p11 Q, V12 = p12 Q, V21 = p21 Q, V22 = p22 Q.
    
    From V11 and V21: V11 V21^† = p11 p21^† (Q Q^†) = p11 p21^† I (since Q is unitary).
    So V11 V21^† = s1 I for some scalar s1.
    Similarly V12 V22^† = p12 p22^† I.
    
    We can recover Q up to a scalar from V11: V11 = p11 Q => Q = V11 / p11.
    But we don't know p11.
    
    Alternative: Use the fact that V is a Kronecker product.
    We can use SVD or eigenvalue decomposition.
    
    Simpler: Since V = (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD), let P = AC, Q = BD.
    We need V = P ⊗ Q.
    
    From the structure:
    V[0,0] = P[0,0] * Q[0,0]
    V[0,1] = P[0,0] * Q[0,1]
    V[1,0] = P[1,0] * Q[0,0]
    V[1,1] = P[1,0] * Q[0,1]
    V[2,0] = P[0,1] * Q[0,0]
    V[2,1] = P[0,1] * Q[0,1]
    V[3,0] = P[1,1] * Q[0,0]  -- wait, this indexing is wrong.
    
    Let me use block notation. V is 4x4, partitioned into 2x2 blocks:
    V = [[V00, V01], [V10, V11]] where each block is 2x2.
    P ⊗ Q = [[p00 Q, p01 Q], [p10 Q, p11 Q]]
    So V00 = p00 Q, V01 = p01 Q, V10 = p10 Q, V11 = p11 Q.
    
    From V00 and V10: V00 V10^† = p00 p10^† (Q Q^†) = p00 p10^† I.
    Let S = V00 V10^†. If S is a scalar times identity, then s = p00 p10^†.
    
    We can find Q by: Q = V00 / p00. But we need p00.
    Note that |p00|^2 + |p10|^2 = 1 (column norm of P).
    And |V00[0,0]|^2 + |V00[1,0]|^2 = |p00|^2 (|Q[0,0]|^2 + |Q[1,0]|^2) = |p00|^2.
    So |p00| = sqrt(|V00[0,0]|^2 + |V00[1,0]|^2).
    
    Similarly, we can find the phase of p00 from the consistency condition.
    
    Let's implement this carefully.
    """
    V = np.array(V, dtype=complex)
    
    # Extract 2x2 blocks
    V00 = V[0:2, 0:2]
    V01 = V[0:2, 2:4]
    V10 = V[2:4, 0:2]
    V11 = V[2:4, 2:4]
    
    # Check if V00 V10^† is scalar * I
    S = V00 @ V10.conj().T
    # S should be s * I
    s = (S[0,0] + S[1,1]) / 2.0
    # Check consistency
    if abs(S[0,1]) > 1e-8 or abs(S[1,0]) > 1e-8:
        # Not a perfect Kronecker product, use least squares
        # For now, proceed with best effort
        pass
    
    # |p00|^2 = trace(V00 V00^†) = ||V00||_F^2
    norm_V00_sq = np.sum(np.abs(V00)**2)
    norm_V10_sq = np.sum(np.abs(V10)**2)
    
    if norm_V00_sq < 1e-12:
        # Degenerate case
        P = np.eye(2, dtype=complex)
        Q = V @ P.conj().T
    else:
        abs_p00 = math.sqrt(norm_V00_sq)
        # p00 = V00[0,0] / Q[0,0], but we don't know Q yet.
        # Use: V00 = p00 Q => Q = V00 / p00
        # We need the phase of p00.
        # From S = p00 p10^† I, and |p00|^2 + |p10|^2 = 1.
        # Also, V10 = p10 Q => p10 = V10[0,0] / Q[0,0].
        # This is circular.
        
        # Alternative: Use the fact that Q is unitary.
        # Q = V00 / p00 must be unitary.
        # So V00 V00^† / |p00|^2 = I.
        # This gives us |p00| but not the phase.
        
        # The phase of p00 can be determined from the off-diagonal consistency.
        # V01 = p01 Q, V11 = p11 Q.
        # V01 V11^† = p01 p11^† I.
        # Let T = V01 @ V11.conj().T
        # t = (T[0,0] + T[1,1]) / 2
        # p01 p11^† = t
        
        # We have:
        # p00 p10^† = s
        # p01 p11^† = t
        # |p00|^2 + |p10|^2 = 1
        # |p01|^2 + |p11|^2 = 1
        # p00 p01^† + p10 p11^† = 0 (orthogonality of rows of P)
        
        # This is getting complex. Let's use a different approach:
        # Directly solve for P and Q using the Kronecker product structure.
        
        # Use SVD of the reshaped matrix? No.
        
        # Let's just use numerical optimization for the local part.
        # Given V, find 4 u3 gates (8 params) such that (A⊗B)(C⊗D) ≈ V.
        
        return None  # Signal to use optimization
    
    # For now, return None to indicate we should use optimization
    return None

def _optimize_local_2qubit(V, time_budget=5.0):
    """Optimize 4 u3 gates to match a 4x4 local unitary V."""
    start = time.time()
    
    def build_and_dist(params):
        # params: 8 values = [theta0, phi0, lam0, theta1, phi1, lam1, theta2, phi2, lam2, theta3, phi3, lam3]
        # Wait, 4 gates * 3 params = 12 params
        # Gate order: A0, A1, C0, C1 (or similar)
        # Circuit: u3(q0, A0), u3(q1, A1), u3(q0, C0), u3(q1, C1)
        # This gives (C0 A0) ⊗ (C1 A1)
        # We want this to equal V.
        
        gates = [
            ["u3", 0, params[0], params[1], params[2]],
            ["u3", 1, params[3], params[4], params[5]],
            ["u3", 0, params[6], params[7], params[8]],
            ["u3", 1, params[9], params[10], params[11]],
        ]
        U = _build(gates, 2)
        return _distance(V, U)
    
    # Use Nelder-Mead with multiple restarts
    def _nelder_mead(f, x0, max_iter=200, tol=1e-10):
        n = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        simplex = [x0.copy()]
        for i in range(n):
            p = x0.copy()
            p[i] += 0.1 if p[i] == 0 else 0.1 * p[i]
            simplex.append(p)
        fvals = [f(p) for p in simplex]
        
        for it in range(max_iter):
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
        return simplex[0], fvals[0]
    
    best_params = None
    best_dist = 1.0
    
    restarts = [
        np.zeros(12),
        np.array([math.pi/2]*12),
        np.array([math.pi/4]*12),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    ]
    
    for x0 in restarts:
        if time.time() - start > time_budget:
            break
        try:
            res_x, res_f = _nelder_mead(build_and_dist, x0, max_iter=300)
            if res_f < best_dist:
                best_dist = res_f
                best_params = res_x
        except:
            pass
    
    if best_params is None:
        best_params = np.zeros(12)
    
    return best_params, best_dist

def _kak_2qubit(target, budget):
    """
    Analytic KAK decomposition for 2-qubit unitary.
    Returns list of gates with at most 3 CNOTs.
    """
    target = np.array(target, dtype=complex)
    
    # Remove global phase
    det = np.linalg.det(target)
    if abs(det) > 1e-15:
        target = target / (det / abs(det))
    
    # Compute KAK angles using the standard method
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    M = np.kron(target, target.T) @ np.kron(sy, sy)
    eigvals, eigvecs = np.linalg.eig(M)
    
    # The eigenvalues are e^{i(theta1+theta2+theta3+theta4)} etc.
    # For a 2-qubit gate, the KAK angles a, b, c satisfy:
    # The four eigenvalues are e^{i(a+b+c)}, e^{i(a+b-c)}, e^{i(a-b+c)}, e^{i(-a+b+c)}
    
    # We need to find a, b, c from the eigenvalues.
    # Let the eigenvalues be λ1, λ2, λ3, λ4.
    # Then:
    # a = (arg(λ1) + arg(λ2) - arg(λ3) + arg(λ4)) / 4  -- but we need to assign correctly.
    
    # A robust method: try all 24 permutations and check consistency.
    # But for now, let's use a simpler approach:
    # The product of all eigenvalues is 1 (det M = 1).
    # Sum of args = 0 mod 2π.
    
    # Let's try the first permutation and verify.
    args = np.angle(eigvals)
    
    # Try: a = (args[0] + args[1] - args[2] + args[3]) / 4
    #       b = (args[0] + args[2] - args[1] + args[3]) / 4
    #       c = (args[0] - args[1] + args[2] + args[3]) / 4
    
    # This assumes a specific ordering. Let's just use optimization for the full circuit.
    # Given the time, I'll use a hybrid: estimate the entangling power and use optimization.
    
    # For now, fall back to optimization with good initialization.
    return _optimize_2qubit(target, budget, time_budget=min(20.0, _remaining_time() - 5.0))

def _optimize_2qubit(target, budget, time_budget=20.0):
    start = time.time()
    num_cnots = min(budget, 3)
    if num_cnots == 0:
        # Local unitary only
        # Decompose target into u3 gates
        # This is a local unitary, so we can use _optimize_local_2qubit
        params, dist = _optimize_local_2qubit(target, time_budget)
        gates = [
            ["u3", 0, params[0], params[1], params[2]],
            ["u3", 1, params[3], params[4], params[5]],
            ["u3", 0, params[6], params[7], params[8]],
            ["u3", 1, params[9], params[10], params[11]],
        ]
        return gates, dist
    
    cnot_pairs = [(0, 1)] * num_cnots
    if num_cnots >= 2:
        cnot_pairs[1] = (1, 0)
    if num_cnots >= 3:
        cnot_pairs[2] = (0, 1)
    
    num_u3 = 2 * (num_cnots + 1)
    n_params = 3 * num_u3
    
    def build_circuit(params):
        gates = []
        k = 0
        for i in range(num_cnots + 1):
            gates.append(["u3", 0, params[k], params[k+1], params[k+2]])
            k += 3
            gates.append(["u3", 1, params[k], params[k+1], params[k+2]])
            k += 3
            if i < num_cnots:
                gates.append(["cx", cnot_pairs[i][0], cnot_pairs[i][1]])
        return gates
    
    def dist_fn(params):
        gates = build_circuit(params)
        U = _build(gates, 2)
        return _distance(target, U)
    
    def _nelder_mead(f, x0, max_iter=500, tol=1e-10):
        n = len(x0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        simplex = [x0.copy()]
        for i in range(n):
            p = x0.copy()
            p[i] += 0.1 if p[i] == 0 else 0.1 * p[i]
            simplex.append(p)
        fvals = [f(p) for p in simplex]
        
        for it in range(max_iter):
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
        return simplex[0], fvals[0]
    
    best_params = None
    best_dist = 1.0
    
    # Multiple restarts
    restarts = [
        np.zeros(n_params),
        np.array([math.pi/2] * n_params),
        np.array([math.pi/4] * n_params),
        np.array([0.7] * n_params),
    ]
    
    for x0 in restarts:
        if time.time() - start > time_budget:
            break
        try:
            res_x, res_f = _nelder_mead(dist_fn, x0, max_iter=1000)
            if res_f < best_dist:
                best_dist = res_f
                best_params = res_x
        except:
            pass
    
    if best_params is None:
        best_params = np.zeros(n_params)
    
    gates = build_circuit(best_params)
    return gates, best_dist

def _optimize_3qubit(target, budget, time_budget=30.0):
    start = time.time()
    num_cnots = min(budget, 14)
    
    if num_cnots == 0:
        # Local only
        gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        return gates, 1.0
    
    # Use a standard 14-CNOT layout
    cnots = [
        (0, 1), (1, 2), (0, 1), (0, 2),
        (1, 0), (2, 1), (0, 2), (1, 0),
        (0, 1), (2, 1), (0, 2), (1, 0),
        (0, 1), (1, 2)
    ][:num_cnots]
    
    num_layers = num_cnots + 1
    n_params = 3 * 3 * num_layers
    
    def build_circuit(params):
        gates = []
        k = 0
        for i in range(num_layers):
            for q in range(3):
                gates.append(["u3", q, params[k], params[k+1], params[k+2]])
                k += 3
            if i < num_cnots:
                gates.append(["cx", cnots[i][0], cnots[i][1]])
        return gates
    
    def dist_fn(params):
        gates = build_circuit(params)
        U = _build(gates, 3)
        return _distance(target, U)
    
    # Use coordinate descent for 135+ params
    # This is slow, so use a limited number of iterations
    
    best_params = np.zeros(n_params)
    best_dist = dist_fn(best_params)
    
    # Simple gradient-free coordinate descent
    step = 0.5
    for iteration in range(50):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(n_params):
            for delta in [step, -step]:
                trial = best_params.copy()
                trial[i] += delta
                d = dist_fn(trial)
                if d < best_dist:
                    best_params = trial
                    best_dist = d
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    
    gates = build_circuit(best_params)
    return gates, best_dist

def _exact_decomposition(gate_name, qubits, budget):
    if gate_name == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_name == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_name == "iswap" and qubits == 2:
        # iSWAP = 0.5 * [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
        # 2-CNOT decomposition:
        # CNOT(0,1), H(1), CNOT(0,1), H(1), S(0), S(1)
        # H = u3(pi/2, 0, pi)
        # S = diag(1, i) = u3(pi/2, 0, 0) * u3(0, 0, pi/2) * u3(pi/2, 0, 0)?
        # Actually, S = e^{i pi/4} * u3(pi/2, 0, -pi/2) * e^{-i pi/4}?
        # Let's use the 3-CNOT version for simplicity if budget allows.
        if budget >= 3:
            return [["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 0, 1],
                    ["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 1, 0],
                    ["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 0, 1]]
        else:
            # 2 CNOT
            return [["cx", 0, 1],
                    ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 0, 1],
                    ["u3", 1, math.pi/2, 0, math.pi],
                    ["u3", 0, math.pi/2, 0, 0],
                    ["u3", 1, math.pi/2, 0, 0]]
    elif gate_name == "toffoli" and qubits == 3:
        # Toffoli = 6 CNOTs
        gates = [
            ["cx", 2, 1],
            ["cx", 1, 0],
            ["cx", 2, 1],
            ["cx", 1, 0],
            ["cx", 2, 1],
            ["cx", 0, 1],
            ["cx", 2, 1],
        ]
        return gates
    return None

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    
    gate_type = _detect_known_gate(target, qubits)
    
    if gate_type != "generic":
        exact = _exact_decomposition(gate_type, qubits, budget)
        if exact is not None and _count_cnots(exact) <= budget and len(exact) <= 400:
            dist = _distance(target, _build(exact, qubits))
            if dist < 1e-6:
                return exact
    
    time_budget = min(40.0, _remaining_time() - 10.0)
    if time_budget < 2.0:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    if qubits == 2:
        gates, dist = _optimize_2qubit(target, budget, time_budget)
    else:
        gates, dist = _optimize_3qubit(target, budget, time_budget)
    
    # Enforce budget
    while _count_cnots(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break
    
    return gates
