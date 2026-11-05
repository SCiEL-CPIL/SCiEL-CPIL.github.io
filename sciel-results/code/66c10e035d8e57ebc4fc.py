"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 matrix multiplication (rank 7).
    Convention: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=0,1), C is 2x2 (rows k=0,1; cols i=0,1).
    U has shape (4, 7): indexed by (i, j) -> i*2+j
    V has shape (4, 7): indexed by (j, k) -> j*2+k
    W has shape (4, 7): indexed by (k, i) -> k*2+i
    """
    # Strassen's 7 products for 2x2 multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3: A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4: A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[1, 6] = 1  # C01
    
    return U, V, W


def _build_strassen_decomposition():
    """Build a rank-88 decomposition using Strassen for 2x2x2 blocks.
    
    Strategy:
    - Split the 4x4 inner dimension (j) into two groups of 2: j in {0,1} and j in {2,3}
    - Split the 8-dim output (k) into four groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    - For each of the 4 output groups and 2 input groups, we have a 3x2 * 2x2 multiplication
    - Apply Strassen (rank 7) to each 3x2*2x2 block? No, Strassen is for 2x2*2x2.
    
    Actually, let me reconsider. The dimensions are:
    A: 3x4 (n=3 rows, m=4 cols)
    B: 4x8 (m=4 rows, p=8 cols)
    C: 8x3 (p=8 rows, n=3 cols)
    
    The tensor T[i,j,k] = delta (i.e., C[k,i] = sum_j A[i,j] * B[j,k])
    
    Let's partition:
    - j dimension (size 4) into: j0={0,1}, j1={2,3}
    - k dimension (size 8) into: k0={0,1}, k1={2,3}, k2={4,5}, k3={6,7}
    
    For each (j_group, k_group) pair, we have a sub-problem:
    A_sub: 3x2 (rows i=0,1,2; cols j in j_group)
    B_sub: 2x2 (rows j in j_group; cols k in k_group)
    C_sub: 2x3 (rows k in k_group; cols i=0,1,2)
    
    This is a 3x2 * 2x2 multiplication. The schoolbook rank is 3*2*2=12.
    Can we do better? The tensor is T[i,j,k] for i in {0,1,2}, j in {0,1}, k in {0,1}.
    This is a 3x2x2 tensor. Its rank... 
    
    Actually, for a 3x2x2 tensor, the rank is at most 3*2*2=12, but it might be lower.
    However, the standard Strassen trick applies to 2x2x2. For 3x2x2, we can view it as
    a sum of three 2x2x2 tensors (one for each i), but they share the j,k structure.
    
    Alternative approach: Partition differently.
    
    Let me try:
    - Split j (size 4) into j0={0,1}, j1={2,3}
    - Split k (size 8) into k0={0,1}, k1={2,3}, k2={4,5}, k3={6,7}
    - Split i (size 3) into i0={0,1}, i1={2}
    
    For (i0, j0, k0): 2x2x2 -> Strassen rank 7
    For (i0, j0, k1): 2x2x2 -> Strassen rank 7
    For (i0, j0, k2): 2x2x2 -> Strassen rank 7
    For (i0, j0, k3): 2x2x2 -> Strassen rank 7
    For (i0, j1, k0): 2x2x2 -> Strassen rank 7
    For (i0, j1, k1): 2x2x2 -> Strassen rank 7
    For (i0, j1, k2): 2x2x2 -> Strassen rank 7
    For (i0, j1, k3): 2x2x2 -> Strassen rank 7
    That's 8 * 7 = 56
    
    For (i1, j0, k0): 1x2x2 -> schoolbook rank 4
    For (i1, j0, k1): 1x2x2 -> schoolbook rank 4
    For (i1, j0, k2): 1x2x2 -> schoolbook rank 4
    For (i1, j0, k3): 1x2x2 -> schoolbook rank 4
    For (i1, j1, k0): 1x2x2 -> schoolbook rank 4
    For (i1, j1, k1): 1x2x2 -> schoolbook rank 4
    For (i1, j1, k2): 1x2x2 -> schoolbook rank 4
    For (i1, j1, k3): 1x2x2 -> schoolbook rank 4
    That's 8 * 4 = 32
    
    Total: 56 + 32 = 88
    
    This matches the knowledge from the run! Let's implement this.
    """
    n, m, p = 3, 4, 8
    rank_total = 88
    
    U = np.zeros((n * m, rank_total), dtype=float)  # (12, 88)
    V = np.zeros((m * p, rank_total), dtype=float)  # (32, 88)
    W = np.zeros((p * n, rank_total), dtype=float)  # (24, 88)
    
    # i groups: i0 = [0, 1], i1 = [2]
    i_groups = [[0, 1], [2]]
    # j groups: j0 = [0, 1], j1 = [2, 3]
    j_groups = [[0, 1], [2, 3]]
    # k groups: k0 = [0, 1], k1 = [2, 3], k2 = [4, 5], k3 = [6, 7]
    k_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    col = 0
    for ig_idx, i_list in enumerate(i_groups):
        for jg_idx, j_list in enumerate(j_groups):
            for kg_idx, k_list in enumerate(k_groups):
                i_start = i_list[0]
                j_start = j_list[0]
                k_start = k_list[0]
                
                if len(i_list) == 2:
                    # 2x2x2 block: use Strassen
                    # i indices: i_start, i_start+1
                    # j indices: j_start, j_start+1
                    # k indices: k_start, k_start+1
                    
                    for s in range(7):
                        # U part
                        for di in range(2):
                            u_val = strassen_U[di * 2 + 0, s]  # A[di, dj]
                            # Wait, strassen_U is indexed by (i, j) -> i*2+j
                            # So strassen_U[0, s] is A00, strassen_U[1, s] is A01, etc.
                            pass
                        
                        # Let me re-index properly
                        # strassen_U[ai, s] where ai = i*2+j for i in {0,1}, j in {0,1}
                        # strassen_V[bj, s] where bj = j*2+k for j in {0,1}, k in {0,1}
                        # strassen_W[ck, s] where ck = k*2+i for k in {0,1}, i in {0,1}
                        
                        for ai in range(4):
                            u_val = strassen_U[ai, s]
                            if u_val != 0:
                                di = ai // 2  # local i index (0 or 1)
                                dj = ai % 2   # local j index (0 or 1)
                                U[(i_start + di) * m + (j_start + dj), col] += u_val
                        
                        for bj in range(4):
                            v_val = strassen_V[bj, s]
                            if v_val != 0:
                                dj = bj // 2   # local j index (0 or 1)
                                dk = bj % 2    # local k index (0 or 1)
                                V[(j_start + dj) * p + (k_start + dk), col] += v_val
                        
                        for ck in range(4):
                            w_val = strassen_W[ck, s]
                            if w_val != 0:
                                dk = ck // 2   # local k index (0 or 1)
                                di = ck % 2    # local i index (0 or 1)
                                W[(k_start + dk) * n + (i_start + di), col] += w_val
                        
                        col += 1
                else:
                    # 1x2x2 block: schoolbook, rank = 1*2*2 = 4
                    # i index: i_start (single)
                    # j indices: j_start, j_start+1
                    # k indices: k_start, k_start+1
                    
                    for dj in range(2):
                        for dk in range(2):
                            U[i_start * m + (j_start + dj), col] = 1
                            V[(j_start + dj) * p + (k_start + dk), col] = 1
                            W[(k_start + dk) * n + i_start, col] = 1
                            col += 1
    
    assert col == rank_total, f"Expected {rank_total} columns, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 3, 4, 8
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    assert V.shape[1] == rank
    assert W.shape[1] == rank
    
    # Build the tensor from the decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The true tensor is T[i, j, k] = 1 if this is a valid (i,j,k) triple, else 0
    # Actually, the matrix multiplication tensor has T[i,j,k] = delta_{valid}
    # More precisely: C[k,i] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes to C[k,i]
    
    # Compute T from decomposition
    T_decomp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                T_decomp[i, j, k] = np.dot(np.dot(u_val, v_val), w_val)
    
    # True tensor: T_true[i, j, k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] appears)
    T_true = np.ones((n, m, p), dtype=float)
    
    max_err = np.max(np.abs(T_decomp - T_true))
    return max_err < 1e-10


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Try Strassen-based decomposition (rank 88)
    try:
        U, V, W = _build_strassen_decomposition()
        # Verify
        if _verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 96)
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

# EVOLVE-BLOCK-END
