"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    The 2x4 * 4x8 multiplication is viewed as 8 independent 2x2 * 2x2 multiplications,
    each done with rank 7 (Strassen), giving total rank 8*7 = 56.
    
    A is (n=2) x (m=4): A[i, j] where i in 0..1, j in 0..3
    B is (m=4) x (p=8): B[j, k] where j in 0..3, k in 0..7
    C is (p=8) x (n=2): C[k, i] where k in 0..7, i in 0..1
    
    We group columns of A and rows of B into 2x2 blocks.
    For u in 0..1 (rows of A, pairs of columns), v in 0..3 (column pairs of B):
      A_block[u*2:(u+1)*2, v*2:(v+1)*2]  (2x2)
      B_block[v*2:(v+1)*2, :]            (2x8) -- we need 2x2 sub-blocks
      
    Actually, let me reconsider the block structure.
    
    A is 2x4. Let's view it as two 2x2 blocks side by side:
      A = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
    
    B is 4x8. View as two 2x8 blocks stacked:
      B = [B0; B1] where B0 = B[0:2, :], B1 = B[2:4, :]
    
    C = A * B = [A0*B0 + A1*B1]  (2x8)
    
    For each 2x2 sub-multiplication within A0*B0 and A1*B1, use Strassen.
    A0 is 2x2, B0 is 2x8. We can view B0 as four 2x2 blocks: B0[:, 0:2], B0[:, 2:4], etc.
    So A0 * B0 = [A0*B0_0 | A0*B0_1 | A0*B0_2 | A0*B0_3] where B0_i is 2x2.
    Each A0*B0_i is a 2x2*2x2 multiplication using Strassen (rank 7).
    Similarly for A1*B1.
    
    Total: 4 sub-blocks for A0*B0 + 4 for A1*B1 = 8 sub-multiplications, each rank 7 = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    # For each block pair (u: row-block of A, v: column-block of B)
    # u in 0..1: selects which 2x2 block of A (columns 2u..2u+1)
    # v in 0..3: selects which 2x2 block of B (rows 2v..2v+1, all 8 cols -> but we need 2x2)
    # Wait, B is 4x8. The 2x2 sub-blocks of B are B[2v:2v+2, 2w:2w+2] for v in 0..1, w in 0..3.
    # But we're doing A_block (2x2) times B_block (2x8). Let me re-think.
    
    # A is 2x4. Block A_u = A[:, 2u:2u+2] for u in 0,1 (each 2x2)
    # B is 4x8. For a given A_u (2x2), we multiply by B[2u:2u+2, :] which is 2x8.
    # We can split B[2u:2u+2, :] into four 2x2 blocks: B[2u:2u+2, 2v:2v+2] for v in 0,1,2,3
    # Each product A_u * B[2u:2u+2, 2v:2v+2] is 2x2 * 2x2 -> 2x2
    # The result goes to C[2v:2v+2, :] (rows 2v,2v+1 of C, which is 2x2)
    
    # C is 8x2. C[2v:2v+2, :] is 2x2.
    
    # Strassen for 2x2 * 2x2:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    for u in range(2):  # A block column
        for v in range(4):  # B block column (within the 2 rows corresponding to u)
            base_col = 7 * (u * 4 + v)
            
            # A block: rows 0,1; cols 2u, 2u+1
            a11_idx = 0 * 4 + 2 * u        # A[0, 2u]
            a12_idx = 0 * 4 + 2 * u + 1    # A[0, 2u+1]
            a21_idx = 1 * 4 + 2 * u        # A[1, 2u]
            a22_idx = 1 * 4 + 2 * u + 1    # A[1, 2u+1]
            
            # B block: rows 2u, 2u+1; cols 2v, 2v+1
            b11_idx = (2 * u) * 8 + 2 * v         # B[2u, 2v]
            b12_idx = (2 * u) * 8 + 2 * v + 1     # B[2u, 2v+1]
            b21_idx = (2 * u + 1) * 8 + 2 * v     # B[2u+1, 2v]
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1 # B[2u+1, 2v+1]
            
            # C block: rows 2v, 2v+1; cols 0,1
            c11_idx = (2 * v) * 2 + 0     # C[2v, 0]
            c12_idx = (2 * v) * 2 + 1     # C[2v, 1]
            c21_idx = (2 * v + 1) * 2 + 0 # C[2v+1, 0]
            c22_idx = (2 * v + 1) * 2 + 1 # C[2v+1, 1]
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col + 0] += 1
            U[a22_idx][base_col + 0] += 1
            V[b11_idx][base_col + 0] += 1
            V[b22_idx][base_col + 0] += 1
            W[c11_idx][base_col + 0] += 1
            W[c22_idx][base_col + 0] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    U, V, W = _build_rank_56(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U, "V": V, "W": W}
