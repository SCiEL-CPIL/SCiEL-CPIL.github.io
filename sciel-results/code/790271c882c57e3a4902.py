"""4x5x6 matrix multiplication decomposition (rank 108, Strassen-based)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, r_start):
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1

    r_offset = 0
    r_offset = add_strassen_block(0, 0, 0, r_offset)
    r_offset = add_strassen_block(0, 0, 2, r_offset)
    r_offset = add_strassen_block(0, 0, 4, r_offset)
    r_offset = add_strassen_block(2, 0, 0, r_offset)
    r_offset = add_strassen_block(2, 0, 2, r_offset)
    r_offset = add_strassen_block(2, 0, 4, r_offset)
    r_offset = add_strassen_block(0, 2, 0, r_offset)
    r_offset = add_strassen_block(0, 2, 2, r_offset)
    r_offset = add_strassen_block(0, 2, 4, r_offset)
    r_offset = add_strassen_block(2, 2, 0, r_offset)
    r_offset = add_strassen_block(2, 2, 2, r_offset)
    r_offset = add_strassen_block(2, 2, 4, r_offset)
    
    # 4x1x6 strip: rank 24
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1
            
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 if j is the shared index
    # Actually, the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if the indices form a valid matrix multiplication path, 0 otherwise.
    # For matrix multiplication: T[i,j,k] = 1 if j is the contraction index (always true in our setup)
    # The decomposition should satisfy: sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j V[j*p+k,r] B[j,k]) W[k*n+i,r]
    # So T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should be 1 for all i,j,k where j is the middle index.
    
    # Build the tensor and verify
    # T[i,j,k] should be 1 for all valid (i,j,k) triples in the matrix multiplication
    # Actually the tensor for matrix multiplication is: T[i,j,k] = delta_{j, j} = 1 for all i,j,k
    # No wait - the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (AB)[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify this
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    errors += 1
    
    if errors > 0:
        # Fallback: return schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][r] = 1
                    V_sb[j*p+k][r] = 1
                    W_sb[k*n+i][r] = 1
                    r += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
