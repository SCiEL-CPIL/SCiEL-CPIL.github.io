import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    dim_A = n * m  # 6
    dim_B = m * p  # 15
    dim_C = p * n  # 10

    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i

    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def verify(U, V, W):
        for (i, j, k) in triples:
            val = sum(U[a_idx(i, j), r] * V[b_idx(j, k), r] * W[c_idx(k, i), r]
                      for r in range(U.shape[1]))
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def is_valid_entries(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    # Strassen 2x2x2 decomposition (rank 7)
    # Local indices: a=(0..1, 0..1), b=(0..1, 0..1), c=(0..1, 0..1)
    # 7 terms with ±1 patterns
    strassen_U = [
        [1, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, -1],
        [0, 1, -1, 1, 0, 1, 1],
        [-1, 1, 1, 0, 1, 0, 1],
    ]
    strassen_V = [
        [1, 1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, -1, 1],
        [1, 0, 1, 1, 0, 1, 0],
        [1, -1, 1, 0, 1, 1, 0],
    ]
    strassen_W = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, -1],
    ]

    def build_2x2x2_block(row_offset, col_offset, rank_offset, U, V, W):
        """Add a 2x2x2 Strassen block.
        row_offset: starting row in A (0 or 1 for the 2x3 matrix)
        col_offset: starting col in B (for the 2x2 block of cols)
        rank_offset: starting rank index
        """
        # A is 2x2: rows 0,1; cols row_offset, row_offset+1
        # B is 2x2: rows col_offset, col_offset+1; cols 0,1
        # C is 2x2: rows 0,1; cols 0,1
        for li in range(2):  # local A row
            for lj in range(2):  # local A col
                for r in range(7):
                    U[li, rank_offset + r] += strassen_U[li * 2 + lj] * 0  # placeholder
        # Let me redo this properly with explicit index mapping
        # Global A[i, j] -> U[a_idx(i,j), r]
        # Global B[j, k] -> V[b_idx(j,k), r]  
        # Global C[k, i] -> W[c_idx(k,i), r]
        #
        # For 2x2x2 block:
        # A local (i, j) where i in {0,1}, j in {j0, j0+1}
        # B local (j, k) where j in {j0, j0+1}, k in {k0, k0+1}
        # C local (k, i) where k in {k0, k0+1}, i in {0,1}
        pass

    # Let me build this more carefully with explicit construction
    rank = 17
    U = np.zeros((dim_A, rank))
    V = np.zeros((dim_B, rank))
    W = np.zeros((dim_C, rank))

    col = 0

    # Block 1: j=0 (first column), all k in 0..4
    # This is 2x1 x 1x5 = 2x5, but we decompose as 2x1 @ 1x5
    # For each k: C[k, i] = sum_j A[i,j] * B[j,k] with only j=0
    # C[k,i] = A[i,0] * B[0,k]
    # We can do this with rank 2: 
    #   term 1: U[i,0]=1, V[0,0]=1, W[k,0]=A[i,0]*B[0,k]... no that's not right
    # Actually: C[k,i] = A[i,0] * B[0,k]
    # Use rank 2:
    #   r=0: U[a_idx(i,0), 0] = 1, V[b_idx(0,k), 0] = 1, W[c_idx(k,i), 0] = 1
    #   But this gives sum over r of U*V*W = 1*1*1 = 1 for each (i,k)
    #   Wait, we need C[k,i] = A[i,0]*B[0,k], not 1.
    #   
    # Hmm, I need to rethink. The tensor is T[i,j,k] = 1 for all i,j,k.
    # The multiplication tensor: C[k,i] = sum_j A[i,j] * B[j,k]
    # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 (the tensor value)
    # 
    # So each term contributes U[i,j,r]*V[j,k,r]*W[k,i,r] to the (i,j,k) entry.
    # For the full sum to equal 1 for all (i,j,k).
    #
    # Block for j=0 only: We want terms that only activate when j=0.
    # If U[i,0,r] = 0 for j != 0, and V[j,k,r] = 0 for j != 0, then
    # only (i, j=0, k) entries get contributions.
    #
    # For j=0, k in 0..4, i in 0..1: need sum_r U[i,0,r]*V[0,k,r]*W[k,i,r] = 1
    # This is a 2x5x1 tensor (i x k x 1, but really just 2x5 matrix).
    # Rank of a 2x5 matrix is at most 2. So we can do it with rank 2.
    # U[i,0,r] * V[0,k,r] * W[k,i,r] summed over r should = 1.
    # Set W[k,i,0] = 1, W[k,i,1] = 1 for all k,i.
    # Then U[i,0,0]*V[0,k,0] + U[i,0,1]*V[0,k,1] = 1 for all i,k.
    # Let U[i,0,0] = 1, U[i,0,1] = 0 for all i.
    # Then V[0,k,0] = 1 for all k, V[0,k,1] = 0.
    # Check: 1*1 + 0*0 = 1. 

    # Block 1: j=0, rank 2
    U[a_idx(0, 0), col] = 1
    U[a_idx(1, 0), col] = 1
    V[b_idx(0, 0), col] = 1
    V[b_idx(0, 1), col] = 1
    V[b_idx(0, 2), col] = 1
    V[b_idx(0, 3), col] = 1
    V[b_idx(0, 4), col] = 1
    for k in range(5):
        W[c_idx(k, 0), col] = 1
        W[c_idx(k, 1), col] = 1
    # Second rank term for j=0: all zeros (or we can just use rank 1)
    # Actually rank 1 suffices for this block. Let me use rank 1.
    col += 1
    # Reset: actually let me use a cleaner approach

    # Let me restart with a cleaner block structure
    U = np.zeros((dim_A, rank))
    V = np.zeros((dim_B, rank))
    W = np.zeros((dim_C, rank))
    col = 0

    # Block A: j=0, all k. Rank 1.
    # U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1 -> product = 1 for (i,0,k). 
    for i in range(2):
        U[a_idx(i, 0), col] = 1
    for k in range(5):
        V[b_idx(0, k), col] = 1
        for i in range(2):
            W[c_idx(k, i), col] = 1
    col += 1  # col=1

    # Block B: j in {1,2}, k=0. This is 2x2 x 2x1 = 2x2.
    # A is 2x2 (i in 0,1; j in 1,2), B is 2x1 (j in 1,2; k=0), C is 2x2 (k=0; i in 0,1)
    # C[0,i] = A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # Tensor: for (i,j,k) with j in {1,2}, k=0: T=1
    # Need sum_r U[i,j,r]*V[j,0,r]*W[0,i,r] = 1 for i in {0,1}, j in {1,2}
    # This is like a 2x2 matrix multiplication where one dim is 1.
    # Rank of 2x2x1 tensor: the tensor is [1,1;1,1] (i x j). Rank = 1? 
    # No: T[i,j] = 1 for all i,j. This is a rank-1 matrix (all ones).
    # But we have the V[j,0,r] factor. Let me think again.
    # Actually the "subtensor" for j in {1,2}, k=0 is:
    # T[i, j, 0] = 1 for i in {0,1}, j in {1,2}
    # We need U[i,j,r]*V[j,0,r]*W[0,i,r] summed over r = 1
    # This is a 2x2x1 "tensor" which is just a 2x2 matrix of ones.
    # Minimum rank to decompose: we can do it with rank 2:
    #   r=0: U[i,1,0]=1, V[1,0,0]=1, W[0,i,0]=1 -> gives 1 for (i,1,0)
    #   r=1: U[i,2,1]=1, V[2,0,1]=1, W[0,i,1]=1 -> gives 1 for (i,2,0)
    # Total: for (i,1,0): 1+0=1, for (i,2,0): 0+1=1. 
    # Rank 2 for this block.
    for i in range(2):
        U[a_idx(i, 1), col] = 1
    for i in range(2):
        U[a_idx(i, 2), col + 1] = 1
    V[b_idx(1, 0), col] = 1
    V[b_idx(2, 0), col + 1] = 1
    for i in range(2):
        W[c_idx(0, i), col] = 1
        W[c_idx(0, i), col + 1] = 1
    col += 2  # col=3

    # Block C: j in {1,2}, k in {1,2}. This is 2x2 x 2x2 = 2x2. Strassen rank 7.
    # Map: local i' = i (0,1), local j' = j-1 (0,1), local k' = k-1 (0,1)
    # U[i, j, r] = strassen_U[i'*2 + j']  (where j = j'+1)
    # V[j, k, r] = strassen_V[j'*2 + k']  (where j = j'+1, k = k'+1)
    # W[k, i, r] = strassen_W[k'*2 + i']  (where k = k'+1)
    for li in range(2):
        for lj in range(2):
            for r in range(7):
                U[a_idx(li, lj + 1), col + r] = strassen_U[li * 2 + lj]
    for lj in range(2):
        for lk in range(2):
            for r in range(7):
                V[b_idx(lj + 1, lk + 1), col + r] = strassen_V[lj * 2 + lk]
    for lk in range(2):
        for li in range(2):
            for r in range(7):
                W[c_idx(lk + 1, li), col + r] = strassen_W[lk * 2 + li]
    col += 7  # col=10

    # Block D: j in {1,2}, k in {3,4}. This is 2x2 x 2x2 = 2x2. Strassen rank 7.
    # Map: local i' = i (0,1), local j' = j-1 (0,1), local k' = k-3 (0,1)
    for li in range(2):
        for lj in range(2):
            for r in range(7):
                U[a_idx(li, lj + 1), col + r] = strassen_U[li * 2 + lj]
    for lj in range(2):
        for lk in range(2):
            for r in range(7):
                V[b_idx(lj + 1, lk + 3), col + r] = strassen_V[lj * 2 + lk]
    for lk in range(2):
        for li in range(2):
            for r in range(7):
                W[c_idx(lk + 3, li), col + r] = strassen_W[lk * 2 + li]
    col += 7  # col=17

    # Verify
    assert col == rank, f"Column mismatch: {col} vs {rank}"

    # Final verification
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[a_idx(i, j), c] = 1
                    V[b_idx(j, k), c] = 1
                    W[c_idx(k, i), c] = 1
                    c += 1

    if not is_valid_entries(U) or not is_valid_entries(V) or not is_valid_entries(W):
        rank = n * m * p
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[a_idx(i, j), c] = 1
                    V[b_idx(j, k), c] = 1
                    W[c_idx(k, i), c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
