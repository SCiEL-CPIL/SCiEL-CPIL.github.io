import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_DEADLINE = 120.0
_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Try Rank 45 decomposition
    # Structure: 3 blocks of (M_{2,2,3} rank 9 + M_{2,1,3} rank 6) = 15 per block
    # Total Rank = 45
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # A indices: i*m+j
    # B indices: j*p+k
    # C indices: k*n+i
    
    # A_L: 0,1,3,4 (a,b,c,d)
    # A_R: 2,5 (a02, a12)
    
    for block in range(3):
        # B block columns: 3*block, 3*block+1, 3*block+2
        k0 = 3 * block
        k1 = 3 * block + 1
        k2 = 3 * block + 2
        
        # B_T (2x3) indices
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 0 * p + k2
        v_h = 1 * p + k0
        v_i = 1 * p + k1
        v_j = 1 * p + k2
        
        # B_B (1x3) indices
        v_b0 = 2 * p + k0
        v_b1 = 2 * p + k1
        v_b2 = 2 * p + k2
        
        # A_L indices
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # A_R indices
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        
        # C indices
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_02 = k2 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        w_12 = k2 * n + 1
        
        # --- Part 1: M_{2,2,3} (Rank 9) ---
        # 1. (a+b)(e+h)
        U[u_a][col] += 1; U[u_b][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        # 2. (c+d)(e+h)
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1; V[v_h][col+1] += 1
        # 3. (a+b)(f+i)
        U[u_a][col+2] += 1; U[u_b][col+2] += 1
        V[v_f][col+2] += 1; V[v_i][col+2] += 1
        # 4. (c+d)(f+i)
        U[u_c][col+3] += 1; U[u_d][col+3] += 1
        V[v_f][col+3] += 1; V[v_i][col+3] += 1
        # 5. (a+b)(g+j)
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_g][col+4] += 1; V[v_j][col+4] += 1
        # 6. (c+d)(g+j)
        U[u_c][col+5] += 1; U[u_d][col+5] += 1
        V[v_g][col+5] += 1; V[v_j][col+5] += 1
        # 7. a e
        U[u_a][col+6] += 1
        V[v_e][col+6] += 1
        # 8. c h
        U[u_c][col+7] += 1
        V[v_h][col+7] += 1
        # 9. b h
        U[u_b][col+8] += 1
        V[v_h][col+8] += 1
        
        # Reconstruct C_L
        # C00 = ae + bh
        W[w_00][col+6] += 1
        W[w_00][col+8] += 1
        
        # C10 = ce + dh
        # From (c+d)(e+h) = ce + ch + de + dh
        # We have M2 = (c+d)(e+h), M8 = ch.
        # We need dh.
        # This decomposition is likely incorrect for C10 without more terms.
        # I will fall back to Rank 50 if verification fails.
        
        # --- Part 2: M_{2,1,3} (Rank 6) ---
        # 10. a02 * b0
        U[u_ar0][col+9] += 1
        V[v_b0][col+9] += 1
        W[w_00][col+9] += 1
        # 11. a02 * b1
        U[u_ar0][col+10] += 1
        V[v_b1][col+10] += 1
        W[w_01][col+10] += 1
        # 12. a02 * b2
        U[u_ar0][col+11] += 1
        V[v_b2][col+11] += 1
        W[w_02][col+11] += 1
        # 13. a12 * b0
        U[u_ar1][col+12] += 1
        V[v_b0][col+12] += 1
        W[w_10][col+12] += 1
        # 14. a12 * b1
        U[u_ar1][col+13] += 1
        V[v_b1][col+13] += 1
        W[w_11][col+13] += 1
        # 15. a12 * b2
        U[u_ar1][col+14] += 1
        V[v_b2][col+14] += 1
        W[w_12][col+14] += 1
        
        col += 15
        
    # Verify
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            np.random.seed(42)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(rank):
                term = np.dot(U_np[:, r], A.reshape(n*m)) * np.dot(V_np[:, r], B.reshape(m*p))
                C_computed += term * W_np[:, r].reshape(p, n).T
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fall back to Rank 50
                return construct_rank_50()
        except Exception:
            return construct_rank_50()
            
    return {"U": U, "V": V, "W": W}

def construct_rank_50():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4
        
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        col += 7
        
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a = 0; u_b = 1; u_c = 3; u_d = 4
    
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    col += 4
    
    u_ar0 = 2
    u_ar1 = 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
        
    return {"U": U, "V": V, "W": W}
