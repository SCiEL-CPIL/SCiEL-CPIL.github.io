"""Matrix multiplication 3,4,4 rank minimization."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    def verify(U, V, W):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True
    
    def strassen_2x2x2():
        """
        Return Strassen's rank-7 decomposition for 2x2 x 2x2 -> 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's 7 products for C = A*B where A,B,C are 2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        #
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # U entries: U[i*2+j, r] for i in {0,1}, j in {0,1}, r in 0..6
        # U[0, r] = a11 coeff, U[1, r] = a12 coeff, U[2, r] = a21 coeff, U[3, r] = a22 coeff
        U_s = np.zeros((4, 7), dtype=np.float64)
        # M1: a11=1, a22=1
        U_s[0, 0] = 1.0; U_s[3, 0] = 1.0
        # M2: a21=1, a22=1
        U_s[2, 1] = 1.0; U_s[3, 1] = 1.0
        # M3: a11=1
        U_s[0, 2] = 1.0
        # M4: a22=1
        U_s[3, 3] = 1.0
        # M5: a11=1, a12=1
        U_s[0, 4] = 1.0; U_s[1, 4] = 1.0
        # M6: a21=1, a11=-1
        U_s[2, 5] = 1.0; U_s[0, 5] = -1.0
        # M7: a12=1, a22=-1
        U_s[1, 6] = 1.0; U_s[3, 6] = -1.0
        
        # V entries: V[j*2+k, r] for j in {0,1}, k in {0,1}
        # V[0, r] = b11 coeff, V[1, r] = b12 coeff, V[2, r] = b21 coeff, V[3, r] = b22 coeff
        V_s = np.zeros((4, 7), dtype=np.float64)
        # M1: b11=1, b22=1
        V_s[0, 0] = 1.0; V_s[3, 0] = 1.0
        # M2: b11=1
        V_s[0, 1] = 1.0
        # M3: b12=1, b22=-1
        V_s[1, 2] = 1.0; V_s[3, 2] = -1.0
        # M4: b21=1, b11=-1
        V_s[2, 3] = 1.0; V_s[0, 3] = -1.0
        # M5: b22=1
        V_s[3, 4] = 1.0
        # M6: b11=1, b12=1
        V_s[0, 5] = 1.0; V_s[1, 5] = 1.0
        # M7: b21=1, b22=1
        V_s[2, 6] = 1.0; V_s[3, 6] = 1.0
        
        # W entries: W[k*2+i, r] for k in {0,1}, i in {0,1}
        # W[0, r] = c11 coeff (i=0,k=0), W[1, r] = c12 coeff (i=0,k=1)
        # W[2, r] = c21 coeff (i=1,k=0), W[3, r] = c22 coeff (i=1,k=1)
        W_s = np.zeros((4, 7), dtype=np.float64)
        # M1: c11+=1, c22+=1
        W_s[0, 0] = 1.0; W_s[3, 0] = 1.0
        # M2: c21+=1
        W_s[2, 1] = 1.0
        # M3: c12+=1
        W_s[1, 2] = 1.0
        # M4: c11+=1, c21+=1
        W_s[0, 3] = 1.0; W_s[2, 3] = 1.0
        # M5: c11-=1, c12+=1
        W_s[0, 4] = -1.0; W_s[1, 4] = 1.0
        # M6: c22+=1
        W_s[3, 5] = 1.0
        # M7: c11+=1
        W_s[0, 6] = 1.0
        
        return U_s, V_s, W_s
    
    def build_block_decomposition():
        """
        Build the rank-44 decomposition using Strassen for 2x2x2 blocks.
        
        For 3x4 x 4x4:
        - Split rows of A into two groups: rows 0-1 (2 rows) and row 2 (1 row)
        - Split columns of A (and rows of B) into two groups: cols 0-1 (2 cols) and cols 2-3 (2 cols)
        - Split columns of B (and output cols) into two groups: cols 0-1 (2 cols) and cols 2-3 (2 cols)
        
        This gives 4 blocks:
        - Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2 x 2x2 -> 2x2) : rank 7
        - Block (0,1): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2 x 2x2 -> 2x2) : rank 7
        - Block (1,0): A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4] (1x2 x 2x2 -> 1x2) : rank 4 (trivial)
        - Block (1,1): A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4] (1x2 x 2x2 -> 1x2) : rank 4 (trivial)
        
        Wait, let me reconsider. The output C is 3x4.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Let's split:
        - A rows: i in {0,1} and i=2
        - A cols / B rows: j in {0,1} and j in {2,3}
        - B cols / C cols: k in {0,1} and k in {2,3}
        
        Blocks:
        1. A[0:2, 0:2] * B[0:2, 0:2] contributes to C[0:2, 0:2] : 2x2 * 2x2 -> 2x2, rank 7
        2. A[0:2, 2:4] * B[2:4, 0:2] contributes to C[0:2, 0:2] : 2x2 * 2x2 -> 2x2, rank 7
        3. A[2:3, 0:2] * B[0:2, 0:2] contributes to C[2:3, 0:2] : 1x2 * 2x2 -> 1x2, rank 4
        4. A[2:3, 2:4] * B[2:4, 0:2] contributes to C[2:3, 0:2] : 1x2 * 2x2 -> 1x2, rank 4
        5. A[0:2, 0:2] * B[0:2, 2:4] contributes to C[0:2, 2:4] : 2x2 * 2x2 -> 2x2, rank 7
        6. A[0:2, 2:4] * B[2:4, 2:4] contributes to C[0:2, 2:4] : 2x2 * 2x2 -> 2x2, rank 7
        7. A[2:3, 0:2] * B[0:2, 2:4] contributes to C[2:3, 2:4] : 1x2 * 2x2 -> 1x2, rank 4
        8. A[2:3, 2:4] * B[2:4, 2:4] contributes to C[2:3, 2:4] : 1x2 * 2x2 -> 1x2, rank 4
        
        Total rank: 7+7+4+4+7+7+4+4 = 44
        """
        total_rank = 44
        U = np.zeros((n * m, total_rank), dtype=np.float64)  # (12, 44)
        V = np.zeros((m * p, total_rank), dtype=np.float64)  # (16, 44)
        W = np.zeros((p * n, total_rank), dtype=np.float64)  # (12, 44)
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Block helper: for a 2x2 x 2x2 -> 2x2 block
        # A sub-block at rows [i0:i0+2], cols [j0:j0+2]
        # B sub-block at rows [j0:j0+2], cols [k0:k0+2]
        # C sub-block at rows [i0:i0+2], cols [k0:k0+2]
        def add_strassen_block(i0, j0, k0, offset):
            """Add Strassen rank-7 block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
            for r in range(7):
                # U: A indices (i, j) -> flat index i*m + j
                for di in range(2):
                    for dj in range(2):
                        u_val = U_s[di * 2 + dj, r]
                        if u_val != 0:
                            U[(i0 + di) * m + (j0 + dj), offset + r] += u_val
                
                # V: B indices (j, k) -> flat index j*p + k
                for dj in range(2):
                    for dk in range(2):
                        v_val = V_s[dj * 2 + dk, r]
                        if v_val != 0:
                            V[(j0 + dj) * p + (k0 + dk), offset + r] += v_val
                
                # W: C indices (k, i) -> flat index k*n + i
                for dk in range(2):
                    for di in range(2):
                        w_val = W_s[dk * 2 + di, r]
                        if w_val != 0:
                            W[(k0 + dk) * n + (i0 + di), offset + r] += w_val
        
        # Block helper: for a 1x2 x 2x2 -> 1x2 block (trivial rank 4)
        # A sub-block at row i0, cols [j0:j0+2]
        # B sub-block at rows [j0:j0+2], cols [k0:k0+2]
        # C sub-block at row i0, cols [k0:k0+2]
        def add_trivial_1x2_block(i0, j0, k0, offset):
            """Add trivial rank-4 block for A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0, k0:k0+2]"""
            for r in range(4):
                # r=0: j=j0, k=k0
                # r=1: j=j0, k=k0+1
                # r=2: j=j0+1, k=k0
                # r=3: j=j0+1, k=k0+1
                j = j0 + (r % 2)
                k = k0 + (r // 2)
                U[i0 * m + j, offset + r] = 1.0
                V[j * p + k, offset + r] = 1.0
                W[k * n + i0, offset + r] = 1.0
        
        offset = 0
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        add_strassen_block(0, 0, 0, offset)
        offset += 7
        
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        add_strassen_block(0, 2, 0, offset)
        offset += 7
        
        # Block 3: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
        add_trivial_1x2_block(2, 0, 0, offset)
        offset += 4
        
        # Block 4: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
        add_trivial_1x2_block(2, 2, 0, offset)
        offset += 4
        
        # Block 5: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        add_strassen_block(0, 0, 2, offset)
        offset += 7
        
        # Block 6: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        add_strassen_block(0, 2, 2, offset)
        offset += 7
        
        # Block 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
        add_trivial_1x2_block(2, 0, 2, offset)
        offset += 4
        
        # Block 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
        add_trivial_1x2_block(2, 2, 2, offset)
        offset += 4
        
        assert offset == total_rank, f"Offset {offset} != total_rank {total_rank}"
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify the decomposition is correct."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-8)
    
    # Try block decomposition first
    U, V, W = build_block_decomposition()
    
    if verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
