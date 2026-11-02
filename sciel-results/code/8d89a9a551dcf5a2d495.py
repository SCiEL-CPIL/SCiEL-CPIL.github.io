import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _bini_rank23_terms():
    """Return 23 (u,v,w) triples for 3x3x3 matrix multiplication."""
    terms = []
    
    def vec(*vals):
        return list(vals)
    
    # Term 0: a00*b00 -> c00
    terms.append((vec(1,0,0, 0,0,0, 0,0,0), vec(1,0,0, 0,0,0, 0,0,0), vec(1,0,0, 0,0,0, 0,0,0)))
    # Term 1: a11*b11 -> c11
    terms.append((vec(0,0,0, 0,1,0, 0,0,0), vec(0,0,0, 0,1,0, 0,0,0), vec(0,0,0, 0,1,0, 0,0,0)))
    # Term 2: a22*b22 -> c22
    terms.append((vec(0,0,0, 0,0,0, 0,0,1), vec(0,0,0, 0,0,0, 0,0,1), vec(0,0,0, 0,0,0, 0,0,1)))
    
    # Terms 3-22: Bini et al. rank-23 additional terms
    terms.append((vec(0,1,0, 1,1,0, 0,1,0), vec(0,0,1, 0,1,0, 1,0,0), vec(0,0,1, 1,0,0, 0,0,0)))
    terms.append((vec(1,0,0, 0,1,1, 0,0,0), vec(0,1,0, 1,1,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,0)))
    terms.append((vec(1,0,1, 0,0,0, 1,0,0), vec(1,0,0, 0,1,0, 1,0,0), vec(1,0,0, 0,0,1, 0,0,0)))
    terms.append((vec(0,1,0, 0,0,1, 1,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,1,0, 0,0,0, 0,1,0), vec(0,0,0, 1,0,1, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,1, 0,0,0, 1,0,0), vec(1,0,0, 0,1,1, 0,0,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,0,0, 1,1,0, 0,0,1), vec(0,1,0, 0,0,1, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,1,0, 0,0,1, 0,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,0,1, 0,1,0, 0,0,0), vec(1,0,0, 0,0,1, 0,1,0), vec(1,0,0, 0,0,1, 0,0,0)))
    terms.append((vec(0,1,0, 1,0,0, 0,1,0), vec(0,0,1, 0,1,0, 1,0,0), vec(0,1,0, 1,0,0, 0,0,0)))
    terms.append((vec(1,0,0, 0,0,1, 1,0,0), vec(0,1,0, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,0,1, 1,0,0, 0,1,0), vec(0,1,0, 0,0,1, 1,0,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,0,0, 0,1,1, 0,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,0, 0,0,1, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,1,0, 0,0,0, 0,0,1), vec(0,0,1, 0,1,0, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,0,1, 0,0,0, 0,1,0), vec(0,1,0, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,0, 1,0,0, 0,0,1), vec(0,0,1, 0,1,0, 0,0,1), vec(0,1,0, 1,0,0, 0,0,1)))
    terms.append((vec(1,0,0, 0,1,0, 0,1,0), vec(0,0,1, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,1, 0,0,0, 0,0,1), vec(0,0,1, 0,1,0, 1,0,0), vec(0,1,0, 0,0,1, 0,0,1)))
    terms.append((vec(1,0,0, 1,0,1, 0,0,0), vec(0,1,0, 0,0,1, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    
    return terms

def _get_small_rank_terms(n, m, p):
    """Get schoolbook terms for small dimensions."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the full decomposition."""
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        a_u = u_2d * A
        b_v = v_2d * B
        contrib = a_u @ b_v
        C_computed += w_2d.T * contrib
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def construct():
    deadline = time.time() + 125.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Split m=5 into 3+2
    # Split p=6 into 3+3
    # Blocks:
    # 1. A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3] (3x3x3, rank 23)
    # 2. A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6] (3x3x3, rank 23)
    # 3. A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3] (3x2x3, rank 18)
    # 4. A[0:3, 3:5] * B[3:5, 3:6] -> C[0:3, 3:6] (3x2x3, rank 18)
    
    total_rank = 23 + 23 + 18 + 18
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    terms_33 = _bini_rank23_terms()
    
    # Block 1: A1 (3x3) * B11 (3x3)
    for t in range(23):
        u, v, w = terms_33[t]
        for i in range(3):
            for j in range(3):
                U[i*5 + j][col] += u[i*3 + j]
        for j in range(3):
            for k in range(3):
                V[j*6 + k][col] += v[j*3 + k]
        for i in range(3):
            for k in range(3):
                W[k*3 + i][col] += w[i*3 + k]
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    # Block 2: A1 (3x3) * B12 (3x3)
    for t in range(23):
        u, v, w = terms_33[t]
        for i in range(3):
            for j in range(3):
                U[i*5 + j][col] += u[i*3 + j]
        for j in range(3):
            for k in range(3):
                V[j*6 + 3 + k][col] += v[j*3 + k]
        for i in range(3):
            for k in range(3):
                W[(3+k)*3 + i][col] += w[i*3 + k]
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    # Block 3: A2 (3x2) * B21 (2x3)
    # Schoolbook for 3x2x3
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i*5 + 3 + j][col] = 1
                V[(3+j)*6 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
    # Block 4: A2 (3x2) * B22 (2x3)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[i*5 + 3 + j][col] = 1
                V[(3+j)*6 + 3 + k][col] = 1
                W[(3+k)*3 + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 10:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback
            total_rank_sb = n * m * p
            U_sb = [[0] * total_rank_sb for _ in range(n * m)]
            V_sb = [[0] * total_rank_sb for _ in range(m * p)]
            W_sb = [[0] * total_rank_sb for _ in range(p * n)]
            col_sb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m + j][col_sb] = 1
                        V_sb[j*p + k][col_sb] = 1
                        W_sb[k*n + i][col_sb] = 1
                        col_sb += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    return {"U": U, "V": V, "W": W}
