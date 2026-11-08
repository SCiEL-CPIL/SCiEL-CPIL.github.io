"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify_np(U, V, W, rank):
        """Verify using numpy for speed and correctness."""
        Ua = np.array(U[:rank]) if rank < len(U[0]) else np.array(U)
        Va = np.array(V[:rank]) if rank < len(V[0]) else np.array(V)
        Wa = np.array(W[:rank]) if rank < len(W[0]) else np.array(W)
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        Ua = Ua.reshape(n, m, rank)
        Va = Va.reshape(m, p, rank)
        Wa = Wa.reshape(p, n, rank)
        # Compute tensor: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # For each (i,j,k), check if sum equals 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(Ua[i,j,:] * Va[j,k,:] * Wa[k,i,:])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook_decomposition():
        """Return schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2(U, V, W, col, i_off, j_off, k_off):
        """
        Add a rank-7 Strassen decomposition for a 2x2x2 block.
        Uses the standard Strassen tensor decomposition.
        """
        # Standard rank-7 decomposition of 2x2x2 matmul tensor
        # Each term: (U_dict, V_dict, W_dict) where keys are local (i,j), (j,k), (k,i)
        terms = [
            # Term 1: (A00+A11)(B00+B11) contributes to C00 and C11
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # Term 2: (A10+A11)B00 contributes to C10 and C11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1, (1,1): -1}),
            # Term 3: A00(B10-B11) contributes to C00 and C01
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            # Term 4: A11(B01-B10) contributes to C00 and C11
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # Term 5: (A00+A10)B11 contributes to C00 and C10
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
            # Term 6: (A10-A00)(B01+B11) contributes to C01 and C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # Term 7: (A01-A11)(B00+B10) contributes to C00 and C01
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for u_pat, v_pat, w_pat in terms:
            for (li, lj), val_u in u_pat.items():
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][col] = val_u
            for (lj, lk), val_v in v_pat.items():
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][col] = val_v
            for (lk, li), val_w in w_pat.items():
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][col] = val_w
            col += 1
        return col

    def schoolbook_block(si, sj, sk, U, V, W, col, i_off, j_off, k_off):
        """Add schoolbook for a block."""
        for li in range(si):
            for lj in range(sj):
                for lk in range(sk):
                    gi = i_off + li
                    gj = j_off + lj
                    gk = k_off + lk
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        return col

    def build_block_strassen():
        """
        Build decomposition: 
        Block 1 (3x3x3): Strassen(2x2x2) + schoolbook remainder = 7 + 19 = 26
        Block 2 (3x2x3): Strassen(2x2x2) + schoolbook remainder = 7 + 10 = 17
        Total: 43
        """
        total_rank = 43
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        # Block 1: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
        col = strassen_2x2x2(U, V, W, col, 0, 0, 0)
        col = schoolbook_block(2, 2, 1, U, V, W, col, 0, 0, 2)
        col = schoolbook_block(2, 1, 2, U, V, W, col, 0, 2, 0)
        col = schoolbook_block(2, 1, 1, U, V, W, col, 0, 2, 2)
        col = schoolbook_block(1, 2, 2, U, V, W, col, 2, 0, 0)
        col = schoolbook_block(1, 2, 1, U, V, W, col, 2, 0, 2)
        col = schoolbook_block(1, 1, 2, U, V, W, col, 2, 2, 0)
        col = schoolbook_block(1, 1, 1, U, V, W, col, 2, 2, 2)
        
        # Block 2: i in {0,1,2}, j in {3,4}, k in {3,4,5}
        col = strassen_2x2x2(U, V, W, col, 0, 3, 3)
        col = schoolbook_block(2, 2, 1, U, V, W, col, 0, 3, 5)
        col = schoolbook_block(1, 2, 2, U, V, W, col, 2, 3, 3)
        col = schoolbook_block(1, 2, 1, U, V, W, col, 2, 3, 5)
        
        assert col == total_rank, f"Rank mismatch: expected {total_rank}, got {col}"
        return {"U": U, "V": V, "W": W}

    # Try the block-Strassen decomposition
    try:
        result = build_block_strassen()
        if verify_np(result["U"], result["V"], result["W"], 43):
            return result
    except Exception:
        pass

    # Fallback: schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
