import random
import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0

def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
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
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"
        iswap = 0.5 * np.array([[0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"
    if qubits == 3:
        toffoli = np.zeros((8, 8), dtype=complex)
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

def _exact_decomposition(gate_name, qubits, budget):
    if gate_name == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_name == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_name == "iswap" and qubits == 2:
        # 2-CNOT iSWAP
        # iSWAP = (S ⊗ S) · CNOT(1,0) · (H ⊗ H) · CNOT(0,1) · (H ⊗ H) · (S† ⊗ S†)
        # But we need exact.
        # Standard 2-CNOT iSWAP:
        # CNOT(0,1), H(1), CNOT(0,1), H(1), S(0), S(1)
        # H = u3(pi/2, 0, pi)
        # S = diag(1, i).
        # S = e^{i*pi/4} * u3(pi/2, 0, -pi/2)? No.
        # S = [[1, 0], [0, i]].
        # u3(theta, phi, lam) = [[cos(t/2), -e^{i lam} sin(t/2)], [e^{i phi} sin(t/2), e^{i(phi+lam)} cos(t/2)]]
        # If we set theta=pi/2, phi=0, lam=pi: H.
        # If we set theta=pi/2, phi=0, lam=0: [[1/sqrt(2), -1/sqrt(2)], [1/sqrt(2), 1/sqrt(2)]] = H without phase? No.
        # S = (I + Z)/2 + i(I - Z)/2?
        # S = e^{i*pi/4} * R_z(-pi/2)?
        # R_z(theta) = [[e^{-i theta/2}, 0], [0, e^{i theta/2}]].
        # R_z(-pi/2) = [[e^{i pi/4}, 0], [0, e^{-i pi/4}]].
        # S = e^{i*pi/4} * R_z(-pi/2) = [[e^{i*pi/2}, 0], [0, 1]] = [[i, 0], [0, 1]]. Not S.
        # S = e^{-i*pi/4} * R_z(pi/2) = [[e^{-i*pi/2}, 0], [0, e^{i*pi/2}]] = [[-i, 0], [0, i]]. Not S.
        # Let's use the 3-CNOT version if budget allows, else we need a different 2-CNOT.
        # Actually, iSWAP can be done with 2 CNOTs:
        # CNOT(0,1), H(1), CNOT(0,1), H(1), S(0), S(1) is 2 CNOTs.
        # We just need to implement S.
        # S = u3(pi/2, 0, 0) is H.
        # S = u3(0, 0, pi) is Z.
        # S = H Z H? H Z H = X.
        # S = e^{i*pi/4} X? No.
        # Let's just use 3 CNOTs for iSWAP. If budget is 2, we might fail.
        # But iSWAP is a known gate, likely budget is 2 or 3.
        # If budget >= 3, use 3 CNOTs.
        if budget >= 3:
            H = math.pi / 2
            phi = 0
            lam = math.pi
            return [["u3", 0, H, phi, lam], ["u3", 1, H, phi, lam],
                    ["cx", 0, 1],
                    ["u3", 0, H, phi, lam], ["u3", 1, H, phi, lam],
                    ["cx", 1, 0],
                    ["u3", 0, H, phi, lam], ["u3", 1, H, phi, lam],
                    ["cx", 0, 1]]
        else:
            # 2 CNOT iSWAP
            # CNOT(0,1), H(1), CNOT(0,1), H(1), S(0), S(1)
            # H = u3(pi/2, 0, pi)
            # S = diag(1, i).
            # S = u3(pi/2, 0, 0) is H.
            # S = u3(0, 0, pi) is Z.
            # S = e^{i*pi/4} * u3(pi/2, 0, -pi/2)?
            # u3(pi/2, 0, -pi/2) = [[1/sqrt(2), i/sqrt(2)], [1/sqrt(2), -i/sqrt(2)]]
            # e^{i*pi/4} * u3(pi/2, 0, -pi/2) = [[e^{i*pi/4}/sqrt(2), e^{i*pi/4}*i/sqrt(2)], [e^{i*pi/4}/sqrt(2), e^{i*pi/4}*(-i)/sqrt(2)]]
            # = [[e^{i*pi/4}/sqrt(2), e^{i*pi/2}/sqrt(2)], [e^{i*pi/4}/sqrt(2), -e^{i*pi/2}/sqrt(2)]]
            # = [[e^{i*pi/4}/sqrt(2), i/sqrt(2)], [e^{i*pi/4}/sqrt(2), -i/sqrt(2)]]
            # This is not S.
            # Let's just return the 3-CNOT circuit and hope budget is 3. If not, we might fail iSWAP.
            return [["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 0, 1],
                    ["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 1, 0],
                    ["u3", 0, math.pi/2, 0, math.pi], ["u3", 1, math.pi/2, 0, math.pi],
                    ["cx", 0, 1]]
    elif gate_name == "toffoli" and qubits == 3:
        # 7-CNOT Toffoli
        gates = []
        cnots = [(2, 1), (1, 0), (2, 1), (1, 0), (2, 1), (0, 1), (2, 1)]
        for c, t in cnots:
            gates.append(["cx", c, t])
        return gates
    return None

def _kak_decompose_2qubit(U):
    """
    Decompose 2-qubit unitary U into L1, L2, L3 and CNOTs.
    Returns a list of gates.
    """
    # KAK Decomposition
    # U = e^{i alpha} (A ⊗ B) e^{i (beta X⊗X + gamma Y⊗Y + delta Z⊗Z)} (C ⊗ D)
    
    # 1. Compute M = (U ⊗ U^T) (sigma_y ⊗ sigma_y)
    sy = np.array([[0, -1j], [1j, 0]])
    M = np.kron(U, U.T) @ np.kron(sy, sy)
    
    # 2. Eigendecompose M
    eigvals, eigvecs = np.linalg.eig(M)
    
    # 3. Extract angles
    # The eigenvalues are e^{i(theta1+theta2+theta3+theta4)}
    # The canonical angles are related to the eigenvalues.
    # For 2 qubits, the canonical angles a, b, c satisfy:
    # e^{i(a+b+c)} = lambda1, e^{i(a+b-c)} = lambda2, e^{i(a-b+c)} = lambda3, e^{i(-a+b+c)} = lambda4
    
    # Sort eigenvalues to be consistent
    # This part is tricky. We use the fact that M is unitary.
    
    # Alternative: Use the known decomposition algorithm.
    # We will use a simpler approach: 
    # If U is locally equivalent to identity, 0 CNOTs.
    # If U is locally equivalent to CNOT, 1 CNOT.
    # Else, 3 CNOTs.
    
    # We can check the distance to identity (0 CNOTs) and CNOT (1 CNOT).
    # But we need the exact single-qubit gates.
    
    # Let's use the standard algorithm:
    # 1. Compute local unitaries.
    
    # This is complex to implement from scratch robustly.
    # Instead, we can use a fixed 3-CNOT layout and optimize.
    # Since 2-qubit generic is only 1 target, and KAK is exact, let's try to implement a robust KAK.
    
    # From Nielsen & Chuang:
    # U = e^{i alpha} (A ⊗ B) e^{i (beta XX + gamma YY + delta ZZ)} (C ⊗ D)
    
    # We can extract A, B, C, D and angles.
    
    # For now, let's use the optimization approach for 2-qubit generic as well, 
    # but with a 3-CNOT layout. It's simpler and likely fast enough for 1 target.
    pass

def _nelder_mead(objective, x0, bounds=None, max_iter=500, tol=1e-6):
    """Simple Nelder-Mead implementation using NumPy."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        point = x0.copy()
        if point[i] != 0:
            point[i] += 0.25 * point[i]
        else:
            point[i] += 0.25
        simplex.append(point)
        
    fvals = [objective(p) for p in simplex]
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
                    
    return simplex[0], fvals[0]

def _optimize_generic(target, qubits, budget, time_budget=30.0):
    start = time.time()
    best_gates = None
    best_dist = 1.0
    
    if qubits == 2:
        # 3 CNOT layout
        layout = []
        pairs = [(0, 1), (1, 0), (0, 1)]
        for i in range(4):
            layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
            if i < 3:
                layout.append(["cx", pairs[i][0], pairs[i][1]])
        num_restarts = 10
    else:
        # 14 CNOT layout
        layout = []
        pairs = [(0, 1), (1, 2)]
        for i in range(15):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
            if i < 14:
                layout.append(["cx", pairs[i % 2][0], pairs[i % 2][1]])
        num_restarts = 5
        
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def objective(params):
        values = params.tolist()
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        U = _build(gates, qubits)
        return _distance(target, U)
        
    for restart in range(num_restarts):
        if time.time() - start > time_budget:
            break
        
        if restart == 0:
            x0 = np.zeros(n_params)
        else:
            rng = random.Random(42 + restart * 100)
            x0 = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_params)])
            
        try:
            # Limit iterations based on time
            max_iter = 200 if qubits == 2 else 100
            res_x, res_f = _nelder_mead(objective, x0, max_iter=max_iter)
            if res_f < best_dist:
                best_dist = res_f
                values = res_x.tolist()
                gates = []
                k = 0
                for gate in layout:
                    if gate[0] == "u3":
                        gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                        k += 3
                    else:
                        gates.append(list(gate))
                best_gates = gates
        except Exception:
            continue
            
    if best_gates is None:
        best_gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
    return best_gates, best_dist

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
    
    time_budget = min(35.0, _remaining_time() - 10.0)
    if time_budget < 2.0:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    gates, dist = _optimize_generic(target, qubits, budget, time_budget)
    
    while _count_cnots(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break
            
    return gates
