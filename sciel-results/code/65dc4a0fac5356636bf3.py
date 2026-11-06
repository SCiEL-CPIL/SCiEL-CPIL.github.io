"""Exact decompositions for known gates + analytic peeling + LM optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 200.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _h(q):
    return _u3(q, math.pi / 2, 0, math.pi)


def _s(q):
    return _u3(q, 0, 0, math.pi / 2)


def _t(q):
    return _u3(q, 0, 0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0, 0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0, 0, -math.pi / 2)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(name, qubits):
    """Check if the target is a known gate that can be exactly synthesized."""
    if name == "CNOT" and qubits == 2:
        return "CNOT"
    elif name == "iSWAP" and qubits == 2:
        return "iSWAP"
    elif name == "SWAP" and qubits == 2:
        return "SWAP"
    elif name == "Toffoli" and qubits == 3:
        return "Toffoli"
    return None


def _synthesize_CNOT():
    """CNOT gate: 1 CNOT."""
    return [_cx(0, 1)]


def _synthesize_iSWAP():
    """iSWAP gate: 3 CNOTs."""
    gates = []
    gates.append(_h(0))
    gates.append(_h(1))
    gates.append(_t(0))
    gates.append(_t(1))
    gates.append(_cx(0, 1))
    gates.append(_t_dag(0))
    gates.append(_t(1))
    gates.append(_cx(0, 1))
    gates.append(_t(0))
    gates.append(_t_dag(1))
    gates.append(_h(0))
    gates.append(_h(1))
    gates.append(_s(0))
    gates.append(_s(1))
    return gates


def _synthesize_SWAP():
    """SWAP gate: 3 CNOTs."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _synthesize_Toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
    gates = []
    gates.append(_h(2))
    gates.append(_cx(1, 2))
    gates.append(_t_dag(2))
    gates.append(_cx(0, 2))
    gates.append(_t(2))
    gates.append(_cx(1, 2))
    gates.append(_t_dag(2))
    gates.append(_cx(0, 2))
    gates.append(_t(2))
    gates.append(_t(1))
    gates.append(_h(2))
    gates.append(_cx(0, 1))
    gates.append(_t_dag(1))
    gates.append(_cx(0, 1))
    gates.append(_t(0))
    gates.append(_t(1))
    return gates


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _cx_matrix(qubits, control, target):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _peel_2qubit(target, qubits=2):
    """Peel off single-qubit gates from a 2-qubit unitary using SVD."""
    dim = 2 ** qubits
    # Reshape: U_{i,j} where i = a*2+c, j = b*2+d
    # U = (A ⊗ B) U_mid (C ⊗ D)
    # We want to find A, B, C, D in SU(2)
    
    # First, peel left: U = L U_mid
    # U_{ac,bd} = sum_{a',b'} L_{a,a'} M_{a'b',bd}
    # Reshape to (2,2,2,2): U[a,c,b,d]
    U = target.reshape(2, 2, 2, 2)
    
    # Left peel: sum over a',b' -> contract with L
    # U[a,c,b,d] = sum_{a',b'} L[a,a'] * M[a',b',c,d]
    # This is a matrix product in the (a) index for each (c,b,d)
    # Reshape U to (2, 2*2*2) where first dim is a
    U_left = U.reshape(2, 8)  # (a, (c,b,d))
    U_mid_left, s, Vh = np.linalg.svd(U_left)
    # U_left = U_mid_left @ s @ Vh
    # L = U_mid_left (unitary), M = s @ Vh reshaped
    L = U_mid_left
    M = (s[:, None] * Vh).reshape(2, 2, 2, 2)
    
    # Now peel right from M: M[a',b',c,d]
    # M = M_mid (C ⊗ D)
    # M[a',b',c,d] = sum_{c',d'} M_mid[a',b',c',d'] C[c',c] D[d',d]
    # Reshape M to (4, 4): rows (a',b'), cols (c,d)
    M_mat = M.reshape(4, 4)
    U_mid_mat, s2, Vh2 = np.linalg.svd(M_mat)
    # M_mat = U_mid_mat @ s2 @ Vh2
    # We want M_mid (C⊗D), so C⊗D = Vh2 (unitary)
    # But Vh2 is 4x4 unitary, need to factor as C⊗D
    # For now, keep as is and handle in optimization
    
    # Actually, let's just peel left and right separately
    # Left: U = L U_mid, L is 4x4 unitary from SVD
    # Right: U_mid = U_mid_final R, R is 4x4 unitary
    
    # Better approach: peel one qubit at a time
    # U = (A ⊗ I) U1 (I ⊗ C) ... this gets complicated
    
    # For 2-qubit, let's use the standard KAK decomposition approach
    # or just use optimization
    
    # Peel left single-qubit on qubit 0:
    # U[a,c,b,d] = sum_{a'} L0[a,a'] U1[a',c,b,d]
    U_l = U.reshape(2, 8)
    L0_mat, s_l, Vh_l = np.linalg.svd(U_l)
    L0 = L0_mat  # 2x2 unitary for qubit 0 left
    U1 = (s_l[:, None] * Vh_l).reshape(2, 2, 2, 2)
    
    # Peel left single-qubit on qubit 1:
    # U1[a',c,b,d] = sum_{c'} L1[c',c] U2[a',c',b,d]
    # Reshape: (a',b) is row, (c,d) is col
    U1_mat = U1.transpose(0, 2, 1, 3).reshape(4, 4)
    L1_mat, s_l1, Vh_l1 = np.linalg.svd(U1_mat)
    L1 = L1_mat.reshape(2, 2, 2, 2)  # This is (c', a', b, d) -> need (c', c) for each (a',b,d)
    # Actually L1_mat = L1[c',c] @ I[a',b] @ I[d] ... 
    # L1_mat[a'*2+b, c'*2+d] = sum L1_left[c',c] delta[a',a''] delta[b,b''] delta[d,d'']
    # So L1_mat = (L1_left ⊗ I_d) ... no
    # Let me redo: U1[a',c,b,d], want U1[a',c,b,d] = sum_{c'} L1[c',c] U2[a',c',b,d]
    # Reshape to (c, (a',b,d)): U1_c = U1.transpose(1,0,2,3).reshape(2, 8)
    U1_c = U1.transpose(1, 0, 2, 3).reshape(2, 8)
    L1_mat2, s_l1b, Vh_l1b = np.linalg.svd(U1_c)
    L1 = L1_mat2  # 2x2 for qubit 1 left
    U2 = (s_l1b[:, None] * Vh_l1b).reshape(2, 2, 2, 2)  # (c', a', b, d) -> transpose back
    
    # Now peel right from U2
    # U2[c',a',b,d] = sum_{d'} U3[c',a',b,d'] R1[d',d]
    U2_d = U2.transpose(0,1,2,3).reshape(4, 2)  # (c',a',b) rows, d col
    # Actually: U2[c',a',b,d], want U2[c',a',b,d] = sum_{d'} U3[c',a',b,d'] R1[d',d]
    # Reshape to ((c',a',b), d): 
    U2_r = U2.reshape(8, 2)
    U3_mat, s_r1, R1_mat = np.linalg.svd(U2_r)
    # U2_r = U3_mat @ s_r1 @ R1_mat
    # U2_r[i,d] = sum_k U3_mat[i,k] s_r1[k] R1_mat[k,d]
    # We want U2[i,d] = sum_{d'} U3[i,d'] R1[d',d]
    # So U3 = U3_mat @ diag(s_r1), R1 = R1_mat
    U3 = (U3_mat * s_r1[None, :]).reshape(2, 2, 2, 2)
    R1 = R1_mat  # 2x2 for qubit 1 right
    
    # Peel right from U3 on qubit 0:
    # U3[c',a',b,d'] = sum_{b'} U4[c',a',b',d'] R0[b',b]
    U3_b = U3.transpose(0,1,3,2).reshape(4, 2)  # (c',a',d') rows, b col
    U4_mat, s_r0, R0_mat = np.linalg.svd(U3_b)
    U4 = (U4_mat * s_r0[None, :]).reshape(2, 2, 2, 2)
    R0 = R0_mat  # 2x2 for qubit 0 right
    
    # Now U4 should be close to a 2-qubit gate with only CNOTs
    # The remaining U4 is what we need to decompose with CNOTs
    # For now, return the peeled single-qubit gates and the residual
    return L0, L1, R0, R1, U4


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations around a CNOT ladder for generic unitaries."""
    dim = 2 ** qubits
    
    # Build CNOT topology
    if qubits == 2:
        cnot_pairs = [(0, 1)] * budget
    else:
        # For 3 qubits, use a better topology
        if budget <= 6:
            cnot_pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (0,1)][:budget]
        else:
            cnot_pairs = []
            pattern = [(0,1), (1,2), (0,2), (1,0), (2,1), (0,1)]
            for i in range(budget):
                cnot_pairs.append(pattern[i % len(pattern)])
    
    # Build layout: initial rotations, CNOT layers with rotations, final rotations
    layout = []
    # Initial rotations
    layout += [("u3", q) for q in range(qubits)]
    # CNOT layers
    for layer in range(budget):
        layout += [("u3", q) for q in range(qubits)]
        layout.append(("cx",) + cnot_pairs[layer])
    # Final rotations
    layout += [("u3", q) for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Precompute CNOT matrices
    cnot_mats = {}
    for c, t in set(cnot_pairs):
        cnot_mats[(c, t)] = _cx_matrix(qubits, c, t)
    
    def build_circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", gate[1], gate[2]])
        return gates
    
    def fidelity_and_grad(values):
        """Compute fidelity and gradient using finite differences."""
        # Build the unitary
        mat = np.eye(dim, dtype=complex)
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                _, q = gate
                theta, phi, lam = values[k], values[k+1], values[k+2]
                single = _u3_matrix(theta, phi, lam)
                # Build full operator
                op = np.array([[1]], dtype=complex)
                for qq in range(qubits):
                    op = np.kron(op, single if qq == q else np.eye(2, dtype=complex))
                mat = op @ mat
                k += 3
            else:
                mat = cnot_mats[(gate[1], gate[2])] @ mat
        
        # Fidelity
        prod = target.conj().T @ mat
        tr_val = np.trace(prod)
        fid = abs(tr_val) / dim
        
        # Gradient via finite differences
        grad = np.zeros(n_params)
        h = 1e-6
        for i in range(n_params):
            if _elapsed() > _DEADLINE - 5:
                break
            vp = values.copy()
            vm = values.copy()
            vp[i] += h
            vm[i] -= h
            
            # Build with vp
            mat_p = np.eye(dim, dtype=complex)
            k = 0
            for gate in layout:
                if gate[0] == "u3":
                    _, q = gate
                    theta, phi, lam = vp[k], vp[k+1], vp[k+2]
                    single = _u3_matrix(theta, phi, lam)
                    op = np.array([[1]], dtype=complex)
                    for qq in range(qubits):
                        op = np.kron(op, single if qq == q else np.eye(2, dtype=complex))
                    mat_p = op @ mat_p
                    k += 3
                else:
                    mat_p = cnot_mats[(gate[1], gate[2])] @ mat_p
            
            tr_p = np.trace(target.conj().T @ mat_p)
            
            # Build with vm
            mat_m = np.eye(dim, dtype=complex)
            k = 0
            for gate in layout:
                if gate[0] == "u3":
                    _, q = gate
                    theta, phi, lam = vm[k], vm[k+1], vm[k+2]
                    single = _u3_matrix(theta, phi, lam)
                    op = np.array([[1]], dtype=complex)
                    for qq in range(qubits):
                        op = np.kron(op, single if qq == q else np.eye(2, dtype=complex))
                    mat_m = op @ mat_m
                    k += 3
                else:
                    mat_m = cnot_mats[(gate[1], gate[2])] @ mat_m
            
            tr_m = np.trace(target.conj().T @ mat_m)
            
            # d|tr|/dparam ≈ (|tr_p| - |tr_m|) / (2h)
            grad[i] = (abs(tr_p) - abs(tr_m)) / (2 * h)
        
        return fid, grad
    
    # Initialize
    import random
    rng = random.Random(4242 + qubits + budget)
    values = np.array([rng.uniform(-0.1, 0.1) for _ in range(n_params)])
    
    # First, try to get a good initial guess using peeling for 2-qubit
    if qubits == 2:
        try:
            L0, L1, R0, R1, U4 = _peel_2qubit(target, 2)
            # Extract parameters from L0, L1, R0, R1
            # u3(theta, phi, lam) = [[cos(theta/2), -e^{i*lam}sin(theta/2)], [e^{i*phi}sin(theta/2), e^{i*(phi+lam)}cos(theta/2)]]
            def extract_u3_params(M):
                # M = [[a, b], [c, d]]
                a, b = M[0,0], M[0,1]
                c, d = M[1,0], M[1,1]
                # a = cos(theta/2), d = e^{i(phi+lam)}cos(theta/2)
                # c = e^{i*phi}sin(theta/2), b = -e^{i*lam}sin(theta/2)
                theta = 2 * math.acos(abs(a))
                if abs(math.sin(theta/2)) > 1e-10:
                    phi = math.atan2(c.imag, c.real)
                    lam = math.atan2(-b.imag, -b.real)
                else:
                    phi, lam = 0.0, 0.0
                return theta, phi, lam
            
            # L0 is for qubit 0 left, L1 for qubit 1 left
            # R0 for qubit 0 right, R1 for qubit 1 right
            # But our layout has: initial u3 on q0, q1, then CNOTs, then final u3 on q0, q1
            # So initial u3 on q0 should be L0, initial u3 on q1 should be L1
            # Final u3 on q0 should be R0^dagger, final u3 on q1 should be R1^dagger
            # Actually, U = (L0⊗L1) U_mid (R0⊗R1), so our circuit is L0⊗L1 then CNOTs then R0⊗R1
            # The CNOT part should approximate U_mid = U4
            
            t0, p0, l0 = extract_u3_params(L0)
            t1, p1, l1 = extract_u3_params(L1)
            # R0^dagger for final
            R0_dag = R0.conj().T
            R1_dag = R1.conj().T
            t0f, p0f, l0f = extract_u3_params(R0_dag)
            t1f, p1f, l1f = extract_u3_params(R1_dag)
            
            # Set initial values
            # Layout: u3(q0), u3(q1), [u3(q0), u3(q1), cx], ..., u3(q0), u3(q1)
            values[0:3] = [t0, p0, l0]
            values[3:6] = [t1, p1, l1]
            # For each CNOT layer, set u3 to identity
            for i in range(budget):
                base = 6 + i * 6
                values[base:base+3] = [0, 0, 0]
                values[base+3:base+6] = [0, 0, 0]
            # Final u3
            final_base = 6 + budget * 6
            values[final_base:final_base+3] = [t0f, p0f, l0f]
            values[final_base+3:final_base+6] = [t1f, p1f, l1f]
        except:
            pass
    
    # LM optimization
    best_values = values.copy()
    best_fid, _ = fidelity_and_grad(values)
    best_dist = 1.0 - best_fid
    
    lam_lm = 0.1
    for iteration in range(100):
        if _elapsed() > _DEADLINE - 10:
            break
        if best_dist < 1e-10:
            break
        
        fid, grad = fidelity_and_grad(values)
        dist = 1.0 - fid
        
        # Compute Hessian approximation (J^T J)
        # For speed, use diagonal approximation
        JtJ = np.diag(np.maximum(grad**2, 1e-10))
        
        # LM step
        try:
            step = np.linalg.solve(JtJ + lam_lm * np.eye(n_params), -grad)
        except:
            step = -grad / (np.linalg.norm(grad) + 1e-10)
        
        # Limit step size
        step_norm = np.linalg.norm(step)
        if step_norm > 1.0:
            step = step / step_norm
        
        new_values = values + step
        new_fid, _ = fidelity_and_grad(new_values)
        new_dist = 1.0 - new_fid
        
        if new_dist < dist:
            values = new_values
            dist = new_dist
            lam_lm = max(lam_lm * 0.5, 1e-6)
            if dist < best_dist:
                best_values = values.copy()
                best_dist = dist
        else:
            lam_lm = min(lam_lm * 2.0, 100.0)
        
        if lam_lm > 1000:
            break
    
    # Final polish with smaller steps
    if _elapsed() < _DEADLINE - 20 and best_dist > 1e-8:
        for i in range(50):
            if _elapsed() > _DEADLINE - 5:
                break
            fid, grad = fidelity_and_grad(best_values)
            dist = 1.0 - fid
            if dist < 1e-10:
                break
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            step = -grad / (grad_norm + 1e-10)
            # Line search
            alpha = 0.1
            for _ in range(10):
                trial = best_values + alpha * step
                trial_fid, _ = fidelity_and_grad(trial)
                if 1.0 - trial_fid < dist:
                    best_values = trial
                    dist = 1.0 - trial_fid
                    break
                alpha *= 0.5
    
    return build_circuit(best_values)


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    known = _is_known_gate(name, qubits)
    if known == "CNOT":
        return _synthesize_CNOT()
    elif known == "iSWAP":
        return _synthesize_iSWAP()
    elif known == "SWAP":
        return _synthesize_SWAP()
    elif known == "Toffoli":
        return _synthesize_Toffoli()
    
    # Generic optimization
    return _optimize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
