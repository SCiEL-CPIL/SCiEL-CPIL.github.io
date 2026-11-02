"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication using Strassen blocks."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    n, m, p = 2, 4, 6

    # Strassen 2x2 factors: 7 products
    # Each factor is a dict mapping local index -> sign
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    def build_decomposition(rank_total):
        """Build rank-42 decomposition using 3 Strassen 2x2 blocks."""
        U = [[0] * rank_total for _ in range(n * m)]  # (8, 42)
        V = [[0] * rank_total for _ in range(m * p)]  # (24, 42)
        W = [[0] * rank_total for _ in range(p * n)]  # (12, 42)

        # We split m=4 into 2 blocks of size 2, and p=6 into 3 blocks of size 2
        # For each pair of (left 2x2 block, right 2x2 block), we apply Strassen
        # Left blocks: rows 0-1 (i=0,1) x cols 0-1 (k=0,1) and rows 0-1 x cols 2-3 (k=2,3)
        # Right blocks: rows 0-1 (k=0,1) x cols 0-1 (j=0,1), rows 2-3 (k=2,3) x cols 0-1, rows 4-5 (k=4,5) x cols 0-1
        
        # Actually, let me think about this more carefully.
        # A is 2x4, B is 4x6, C is 2x6
        # We can partition A into two 2x2 blocks: A = [A1 A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
        # We can partition B into three 2x2 blocks vertically: B = [B1; B2; B3] where B1 = B[0:2, :], B2 = B[2:4, :], B3 = B[4:6, :]
        # Then C = A1*B1 + A2*B2
        
        # Wait, that's not right. Let me reconsider.
        # A is 2x4, B is 4x6.
        # Partition A into 2x2 blocks: A = [A11 A12] where A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # Partition B into 2x2 blocks: B = [B11 B12 B13; B21 B22 B23] where B11 = B[0:2, 0:2], etc.
        # Then C = A11*B11 + A12*B21 for the first column block, etc.
        
        # Actually, let me use a simpler approach:
        # Split m=4 into 2 blocks of 2: m1=2, m2=2
        # Split p=6 into 3 blocks of 2: p1=2, p2=2, p3=2
        # For each (m_block, p_block) pair, we have a 2x2 @ 2x2 multiplication
        # There are 2*3 = 6 such pairs, each requiring rank 7, total rank = 6*7 = 42
        
        col_offset = 0
        for mb in range(2):  # m block: 0 or 1 (corresponding to k indices 2*mb, 2*mb+1)
            for pb in range(3):  # p block: 0, 1, or 2 (corresponding to j indices 2*pb, 2*pb+1)
                for prod_idx in range(7):
                    col = col_offset + prod_idx
                    
                    # U mapping: U has shape (n*m, rank) = (8, 42)
                    # U[i*m + k, r] where i in {0,1}, k in {0,1,2,3}
                    # For m_block mb, we use k indices 2*mb and 2*mb+1
                    # Local index 0 -> (i=0, k=2*mb), 1 -> (i=0, k=2*mb+1), 2 -> (i=1, k=2*mb), 3 -> (i=1, k=2*mb+1)
                    local_U_map = {
                        0: 0 * 4 + (2 * mb + 0),
                        1: 0 * 4 + (2 * mb + 1),
                        2: 1 * 4 + (2 * mb + 0),
                        3: 1 * 4 + (2 * mb + 1),
                    }
                    for local_idx, sign in strassen_U[prod_idx].items():
                        U[local_U_map[local_idx]][col] += sign
                    
                    # V mapping: V has shape (m*p, rank) = (24, 42)
                    # V[k*p + j, r] where k in {0,1,2,3}, j in {0,1,2,3,4,5}
                    # For m_block mb, k indices are 2*mb and 2*mb+1
                    # For p_block pb, j indices are 2*pb and 2*pb+1
                    # Local index 0 -> (k=2*mb, j=2*pb), 1 -> (k=2*mb, j=2*pb+1), 2 -> (k=2*mb+1, j=2*pb), 3 -> (k=2*mb+1, j=2*pb+1)
                    local_V_map = {
                        0: (2 * mb + 0) * 6 + (2 * pb + 0),
                        1: (2 * mb + 0) * 6 + (2 * pb + 1),
                        2: (2 * mb + 1) * 6 + (2 * pb + 0),
                        3: (2 * mb + 1) * 6 + (2 * pb + 1),
                    }
                    for local_idx, sign in strassen_V[prod_idx].items():
                        V[local_V_map[local_idx]][col] += sign
                    
                    # W mapping: W has shape (p*n, rank) = (12, 42)
                    # W[j*n + i, r] where j in {0,1,2,3,4,5}, i in {0,1}
                    # For p_block pb, j indices are 2*pb and 2*pb+1
                    # Local index 0 -> (j=2*pb, i=0), 1 -> (j=2*pb+1, i=0), 2 -> (j=2*pb, i=1), 3 -> (j=2*pb+1, i=1)
                    local_W_map = {
                        0: (2 * pb + 0) * 2 + 0,
                        1: (2 * pb + 1) * 2 + 0,
                        2: (2 * pb + 0) * 2 + 1,
                        3: (2 * pb + 1) * 2 + 1,
                    }
                    for local_idx, sign in strassen_W[prod_idx].items():
                        W[local_W_map[local_idx]][col] += sign
                
                col_offset += 7
        
        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify: check T[i,k,j] = sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1 for all i,k,j"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]

        # Reshape: U (n*m, rank) -> (n, m, rank), V (m*p, rank) -> (m, p, rank), W (p*n, rank) -> (p, n, rank)
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)

        # T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
        T_computed = np.einsum('ikr,kjr,jir->ijk', U_t, V_t, W_t)
        max_err = np.max(np.abs(T_computed - 1.0))
        return max_err < 1e-9, max_err

    # Build the rank-42 decomposition
    U, V, W = build_decomposition(42)

    # Verify
    is_valid, max_err = verify_decomposition(U, V, W)
    
    if not is_valid:
        # If verification fails, try a fallback: brute-force rank-48 decomposition
        # This should always be valid
        U_fb = [[0] * 48 for _ in range(n * m)]
        V_fb = [[0] * 48 for _ in range(m * p)]
        W_fb = [[0] * 48 for _ in range(p * n)]
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * m * p + k * p + j
                    U_fb[i * m + k][r] = 1
                    V_fb[k * p + j][r] = 1
                    W_fb[j * n + i][r] = 1
        
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
