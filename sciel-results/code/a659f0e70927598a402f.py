"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 110.0
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    N = 6
    Jz = 0.8
    h = 0.3
    dim = 2**N
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_op(i, j):
        ops_x = [I2] * N
        ops_y = [I2] * N
        ops_z = [I2] * N
        ops_x[i] = sx; ops_x[j] = sx
        ops_y[i] = sy; ops_y[j] = sy
        ops_z[i] = sz; ops_z[j] = sz
        return kron_all(ops_x) + kron_all(ops_y) + Jz * kron_all(ops_z)
    
    def field_op(half=True):
        h_eff = h / 2.0 if half else h
        ops = [I2] * N
        for i in range(N):
            ops[i] = sz
        return -h_eff * kron_all(ops)
    
    A = bond_op(1, 2) + bond_op(3, 4) + field_op(half=True)
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5) + field_op(half=True)
    H = A + B
    
    eigA, vecA = np.linalg.eigh(A)
    eigB, vecB = np.linalg.eigh(B)
    eigH, vecH = np.linalg.eigh(H)
    
    invA = vecA.conj().T
    invB = vecB.conj().T
    invH = vecH.conj().T
    
    def expm_A(c, t):
        if abs(c * t) < 1e-15:
            return np.eye(dim, dtype=complex)
        phase = np.exp(-1j * c * t * eigA)
        return vecA @ (phase * invA)
    
    def expm_B(c, t):
        if abs(c * t) < 1e-15:
            return np.eye(dim, dtype=complex)
        phase = np.exp(-1j * c * t * eigB)
        return vecB @ (phase * invB)
    
    def expm_H(t):
        phase = np.exp(-1j * t * eigH)
        return vecH @ (phase * invH)
    
    t_vals = [0.25, 0.5, 1.0]
    U_exact = [expm_H(t) for t in t_vals]
    
    def evaluate(a, b):
        total_err = 0.0
        for t, Ue in zip(t_vals, U_exact):
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                U = expm_B(b[k], t) @ U
                U = expm_A(a[k], t) @ U
            err = np.linalg.norm(U - Ue, 2)
            total_err += err
        return total_err / 3.0
    
    # Try several candidate patterns
    candidates = []
    
    # 1. Uniform
    candidates.append(([1.0/12.0]*12, [1.0/12.0]*12))
    
    # 2. Strang-like: a=(1/2,1/2,...) b=(1,0,...)
    a_strang = [0.5/6.0]*12
    b_strang = []
    for _ in range(6):
        b_strang += [1.0/6.0, 0.0]
    candidates.append((a_strang, b_strang))
    
    # 3. 4th-order Suzuki scaled
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = 1.0 - c1 - c2
    # B(c1/2) A(c1/2) B(c2/2) A(c2/2) B(c3/2) A(c3/2) B(c2/2) A(c2/2) B(c1/2) A(c1/2)
    # 5 B, 5 A. Pad with 1 more each.
    a_suz = [c1/2.0, c2/2.0, c3/2.0, c2/2.0, c1/2.0, 0.0]
    b_suz = [c1/2.0, c2/2.0, c3/2.0, c2/2.0, c1/2.0, 0.0]
    # Normalize to sum=1
    a_suz = [x / sum(a_suz) for x in a_suz] * 2  # Repeat to get 12
    b_suz = [x / sum(b_suz) for x in b_suz] * 2
    candidates.append((a_suz, b_suz))
    
    best_a, best_b = candidates[0]
    best_val = evaluate(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        if time.time() > deadline:
            break
        v = evaluate(a_c, b_c)
        if v < best_val:
            best_val = v
            best_a, best_b = a_c, b_c
    
    # Local search from best
    if time.time() < deadline - 10.0:
        best_params = list(best_a[:11]) + list(best_b[:11])
        
        def params_to_coefficients(params):
            a_free = params[:11]
            b_free = params[11:22]
            a = list(a_free) + [1.0 - sum(a_free)]
            b = list(b_free) + [1.0 - sum(b_free)]
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            return a, b
        
        def objective(params):
            a, b = params_to_coefficients(params)
            return evaluate(a, b)
        
        # Coordinate descent
        step = 0.01
        for coord in range(22):
            if time.time() > deadline:
                break
            for delta in [-step, step]:
                p = list(best_params)
                p[coord] += delta
                v = objective(p)
                if v < best_val:
                    best_val = v
                    best_params = p
        
        # A few more passes
        for _ in range(3):
            if time.time() > deadline:
                break
            step *= 0.5
            for coord in range(22):
                if time.time() > deadline:
                    break
                for delta in [-step, step]:
                    p = list(best_params)
                    p[coord] += delta
                    v = objective(p)
                    if v < best_val:
                        best_val = v
                        best_params = p
        
        best_a, best_b = params_to_coefficients(best_params)
    
    return best_a, best_b
# EVOLVE-BLOCK-END
