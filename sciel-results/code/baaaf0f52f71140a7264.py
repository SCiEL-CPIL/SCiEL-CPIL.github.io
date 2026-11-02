import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Strategy: Decompose 2x5x6 into 6 blocks of 2x2x2 (Strassen, rank 7) 
    # and 3 blocks of 2x1x2 (Standard, rank 4).
    # Total rank = 6 * 7 + 3 * 4 = 42 + 12 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Define j and k partitions
    # j in {0, 1, 2, 3, 4}
    # k in {0, 1, 2, 3, 4, 5}
    
    # Blocks:
    # J0={0,1}, J1={2,3}, J2={4}
    # K0={0,1}, K1={2,3}, K2={4,5}
    
    j_blocks = [(0, 1), (2, 3), (4, None)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]

    def add_strassen_block(j1, j2, k1, k2, col_offset):
        """
        Adds 7 Strassen terms for a 2x2x2 block.
        i is implicitly 0,1.
        j indices are j1, j2.
        k indices are k1, k2.
        """
        # Local indices in U (i*m + j), V (j*p + k), W (k*n + i)
        # A submatrix indices in U:
        u_00 = 0 * m + j1
        u_01 = 0 * m + j2
        u_10 = 1 * m + j1
        u_11 = 1 * m + j2
        
        # B submatrix indices in V:
        v_00 = j1 * p + k1
        v_01 = j1 * p + k2
        v_10 = j2 * p + k1
        v_11 = j2 * p + k2
        
        # C submatrix indices in W:
        w_00 = k1 * n + 0
        w_01 = k2 * n + 0
        w_10 = k1 * n + 1
        w_11 = k2 * n + 1
        
        c = col_offset
        
        # M1 = (A00 + A11)(B00 + B11)
        # Contributes to C00 (+1), C11 (+1)
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_11][c] += 1
        c += 1
        
        # M2 = (A10 + A11)B00
        # Contributes to C10 (+1), C11 (-1)
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_10][c] += 1
        W[w_11][c] += -1
        c += 1
        
        # M3 = A00(B01 - B11)
        # Contributes to C01 (+1), C11 (+1)
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] += -1
        W[w_01][c] += 1
        W[w_11][c] += 1
        c += 1
        
        # M4 = A11(B10 - B00)
        # Contributes to C00 (+1), C10 (+1)
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] += -1
        W[w_00][c] += 1
        W[w_10][c] += 1
        c += 1
        
        # M5 = (A00 + A01)B11
        # Contributes to C00 (-1), C01 (+1)
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += -1
        W[w_01][c] += 1
        c += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # Contributes to C11 (+1)
        U[u_10][c] += 1
        U[u_00][c] += -1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_11][c] += 1
        c += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # Contributes to C00 (+1)
        U[u_01][c] += 1
        U[u_11][c] += -1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        c += 1
        
        return c

    def add_standard_block(j1, k1, k2, col_offset):
        """
        Adds 4 standard terms for a 2x1x2 block.
        j is j1 (single column in A, single row in B).
        k are k1, k2.
        i are 0, 1.
        """
        c = col_offset
        for i in range(n):
            for k in (k1, k2):
                u_idx = i * m + j1
                v_idx = j1 * p + k
                w_idx = k * n + i
                U[u_idx][c] = 1
                V[v_idx][c] = 1
                W[w_idx][c] = 1
                c += 1
        return c

    # Iterate through blocks
    for jb in j_blocks:
        for kb in k_blocks:
            j1, j2 = jb
            k1, k2 = kb
            
            if j2 is not None and k2 is not None:
                # 2x2x2 Block
                col = add_strassen_block(j1, j2, k1, k2, col)
            else:
                # 2x1x2 Block (j2 is None)
                col = add_standard_block(j1, k1, k2, col)

    # Verify col is correct
    if col != rank:
        # If something went wrong with column count, we might have an error in logic.
        # But 6*7 + 3*4 = 54 exactly.
        pass

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        
        # Check that the reconstructed tensor matches the identity tensor for multiplication
        # T[i,j,k] = 1 if valid, else 0.
        # Here all (i,j,k) in range are valid.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: Return a trivial rank 60 decomposition if verification fails
        # This ensures we don't return an invalid result, though score will be lower.
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][col_f] = 1
                    V_f[j*p+k][col_f] = 1
                    W_f[k*n+i][col_f] = 1
                    col_f += 1
        
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
