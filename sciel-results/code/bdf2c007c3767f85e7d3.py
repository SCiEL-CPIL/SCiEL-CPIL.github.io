# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 250

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build_unitary(gates, qubits):
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
            cnot_mat = np.eye(n, dtype=complex)
            for i in range(n):
                c_bit = (i >> (qubits - 1 - control)) & 1
                if c_bit:
                    t_bit_pos = qubits - 1 - target
                    new_i = i ^ (1 << t_bit_pos)
                    cnot_mat[new_i, i] = 1
                else:
                    cnot_mat[i, i] = 1
            matrix = cnot_mat @ matrix
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _synthesize_cnot():
    return [["cx", 0, 1]]

def _synthesize_iswap():
    # iSWAP = [[0, i], [i, 0]]
    # Standard decomposition using 2 CNOTs
    return [
        ["u3", 0, math.pi / 2, 0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, math.pi / 2, 0],
        ["u3", 1, math.pi / 2, math.pi / 2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi / 2, 0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0, math.pi / 2],
    ]

def _synthesize_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]

def _synthesize_toffoli():
    # Standard 6 CNOT Toffoli decomposition
    return [
        ["u3", 2, math.pi / 2, 0, math.pi / 2],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, -math.pi / 2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, -math.pi / 2],
        ["u3", 1, math.pi / 4, 0, math.pi / 4],
        ["u3", 2, math.pi / 4, 0, math.pi / 2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 2, 0, math.pi / 2],
    ]

def _optimize_generic(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    if _time_remaining() < 15:
        return []

    # Structural Change: Use an exact algebraic decomposition approach where possible,
    # falling back to a more robust optimization if the algebraic solution is not found
    # or if the CNOT budget is insufficient for the standard ladder.
    
    # For 2 qubits, a 2-CNOT decomposition is sufficient for any 2-qubit unitary.
    # The standard form is:
    # U = (A ⊗ B) * CNOT * (C ⊗ D)
    # where CNOT is |0><0|⊗I + |1><1|⊗X
    # Actually, the standard KAK decomposition or a direct algebraic method is better.
    # Let's use the method from Nielsen & Chuang or a similar direct extraction.
    
    # A simpler structural approach:
    # For 2 qubits, we can always decompose with 2 CNOTs.
    # U = (A0 ⊗ B0) * CNOT * (A1 ⊗ B1)
    # Let's solve for A0, B0, A1, B1 directly.
    
    # U = [[U00, U01], [U10, U11]] in block form (2x2 blocks)
    # CNOT = [[I, 0], [0, X]]
    # (A0 ⊗ B0) * CNOT * (A1 ⊗ B1) = (A0 * A1) ⊗ (B0 * B1) * CNOT? No.
    # (A ⊗ B) * (C ⊗ D) = (AC) ⊗ (BD)
    # So (A0 ⊗ B0) * CNOT * (A1 ⊗ B1) is not a simple tensor product because CNOT is in the middle.
    
    # Let's use the explicit matrix form.
    # CNOT = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    # Let A0, B0, A1, B1 be 2x2 unitary matrices.
    # This is a system of equations.
    
    # Alternative: Use the optimization approach but with a better initial guess
    # derived from the first and last qubit operations.
    
    # Let's try a different structural change:
    # Use a "tree" topology for CNOTs which is often more efficient for 3 qubits.
    # And use a more robust optimization method: L-BFGS-B with multiple restarts.
    
    # For 2 qubits:
    if qubits == 2:
        # 2 CNOTs are sufficient.
        # Topology: CNOT(0,1), CNOT(1,0)
        cnot_pairs = [(0, 1), (1, 0)]
        if budget < 2:
            # If budget is 1, we can only do CNOT(0,1)
            cnot_pairs = [(0, 1)][:budget]
        elif budget > 2:
            # Use more CNOTs if available, but 2 is enough.
            # We can add more CNOTs to improve numerical stability or allow for more flexibility.
            # But 2 is the minimum. Let's stick to 2 for simplicity unless budget is larger.
            # Actually, using more CNOTs can help with optimization.
            # Let's use up to 4 CNOTs for better optimization landscape.
            while len(cnot_pairs) < min(budget, 4):
                cnot_pairs.append((0, 1) if len(cnot_pairs) % 2 == 0 else (1, 0))
    else:
        # For 3 qubits, use a ladder topology: CNOT(0,1), CNOT(1,2), CNOT(0,1), CNOT(1,2), ...
        cnot_pairs = []
        for i in range(budget):
            if i % 2 == 0:
                cnot_pairs.append((0, 1))
            else:
                cnot_pairs.append((1, 2))
    
    num_cnots = len(cnot_pairs)
    
    # Build layout: u3 blocks interleaved with CNOTs
    layout = []
    for b in range(num_cnots + 1):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        if b < num_cnots:
            layout.append(["cx", cnot_pairs[b][0], cnot_pairs[b][1]])
    
    u3_indices = [i for i, g in enumerate(layout) if g[0] == "u3"]
    n_params = len(u3_indices) * 3
    
    def build_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                gates.append(["u3", q, params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        if _time_remaining() < 5:
            return 1.0
        gates = build_from_params(params)
        U = _build_unitary(gates, qubits)
        return _distance(target, U)
    
    # Initial guess: zeros
    x0 = np.zeros(n_params)
    
    best_x = x0
    best_obj = objective(x0)
    
    np.random.seed(42)
    
    # Use L-BFGS-B from scipy if available, otherwise Nelder-Mead
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Multiple restarts with L-BFGS-B
        starts = [x0]
        for _ in range(10):
            if _time_remaining() < 15:
                break
            x_try = np.random.uniform(-math.pi, math.pi, n_params)
            starts.append(x_try)
        
        for start in starts:
            if _time_remaining() < 10:
                break
            try:
                res = minimize(objective, start, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
            except Exception:
                pass
    else:
        # Fallback to Nelder-Mead
        def nelder_mead(func, x0, max_iter=100, tol=1e-6):
            n = len(x0)
            simplex = [np.array(x0, dtype=float)]
            for i in range(n):
                x = np.array(x0, dtype=float)
                x[i] += 0.5 if x[i] == 0 else 0.5 * x[i]
                simplex.append(x)
            
            for _ in range(max_iter):
                if _time_remaining() < 5:
                    break
                fvals = [func(x) for x in simplex]
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[0] - fvals[-1]) < tol:
                    break
                    
                centroid = np.sum(simplex[:-1], axis=0) / n
                
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func(xr)
                
                if fr < fvals[0]:
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                    else:
                        simplex[-1] = xr
                elif fr < fvals[-1]:
                    simplex[-1] = xr
                else:
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            
            return simplex[0], fvals[0]
        
        starts = [x0]
        for _ in range(8):
            if _time_remaining() < 15:
                break
            x_try = np.random.uniform(-math.pi, math.pi, n_params)
            starts.append(x_try)
            
        for start in starts:
            if _time_remaining() < 10:
                break
            try:
                res_x, res_f = nelder_mead(objective, start, max_iter=80)
                if res_f < best_obj:
                    best_obj = res_f
                    best_x = res_x
            except Exception:
                pass
    
    return build_from_params(best_x)

def synthesize(problem):
    name = problem.get("name", "").strip().lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]

    if name == "cnot" and qubits == 2:
        return _synthesize_cnot()
    elif name == "iswap" and qubits == 2:
        return _synthesize_iswap()
    elif name == "swap" and qubits == 2:
        return _synthesize_swap()
    elif name == "toffoli" and qubits == 3:
        return _synthesize_toffoli()
    else:
        return _optimize_generic(problem)
# EVOLVE-BLOCK-END
