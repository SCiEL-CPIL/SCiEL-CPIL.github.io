import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    deadline = 120  # safety margin from 150s

    # Strassen 2x2x2 decomposition patterns
    # For each rank index r (0..6), store non-zero entries as (flat_index, value)
    # U: flat index = i*2 + j (i,j in {0,1})
    # V: flat index = j*2 + k (j,k in {0,1})
    # W: flat index = k*2 + i (k,i in {0,1})  [trace convention: W[k*n+i, r]]
    
    strassen_U = [
        [(0, 1), (3, 1)],   # r=0: i=0,j=0 and i=1,j=1
        [(2, 1), (3, 1)],   # r=1: i=1,j=0 and i=1,j=1
        [(0, 1)],           # r=2: i=0,j=0
        [(3, 1)],           # r=3: i=1,j=1
        [(0, 1), (1, 1)],   # r=4: i=0,j=0 and i=0,j=1
        [(2, 1), (0, -1)],  # r=5: i=1,j=0 and i=0,j=0
        [(1, 1), (3, -1)],  # r=6: i=0,j=1 and i=1,j=1
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # r=0: j=0,k=0 and j=1,k=1
        [(0, 1)],           # r=1: j=0,k=0
        [(1, 1), (3, -1)],  # r=2: j=0,k=1 and j=1,k=1
        [(2, 1), (0, -1)],  # r=3: j=1,k=0 and j=0,k=0
        [(3, 1)],           # r=4: j=1,k=1
        [(0, 1), (1, 1)],   # r=5: j=0,k=0 and j=0,k=1
        [(2, 1), (3, 1)],   # r=6: j=1,k=0 and j=1,k=1
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # r=0: k=0,i=0 and k=1,i=1
        [(1, 1), (3, -1)],  # r=1: k=0,i=1 and k=1,i=1
        [(2, 1), (3, 1)],   # r=2: k=1,i=0 and k=1,i=1
        [(0, 1), (1, 1)],   # r=3: k=0,i=0 and k=0,i=1
        [(0, -1), (2, 1)],  # r=4: k=0,i=0 and k=1,i=0
        [(3, 1)],           # r=5: k=1,i=1
        [(0, 1)],           # r=6: k=0,i=0
    ]

    def build_strassen_kronecker(rank):
        """Build the rank-56 Kronecker product of Strassen blocks."""
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # 8 blocks: for j_block in {0,1} and k_block in {0,1,2,3}
        # j_block selects which 2 of the 4 j indices
        # k_block selects which 2 of the 8 k indices
        block_idx = 0
        for j_block in range(2):
            for k_block in range(4):
                j_off = 2 * j_block
                k_off = 2 * k_block
                base = 7 * block_idx
                if base + 7 > rank:
                    break
                for r in range(7):
                    cr = base + r
                    for idx, val in strassen_U[r]:
                        local_i, local_j = divmod(idx, 2)
                        gi = local_i
                        gj = local_j + j_off
                        U[gi * m + gj, cr] += val
                    for idx, val in strassen_V[r]:
                        local_j, local_k = divmod(idx, 2)
                        gj = local_j + j_off
                        gk = local_k + k_off
                        V[gj * p + gk, cr] += val
                    for idx, val in strassen_W[r]:
                        local_k, local_i = divmod(idx, 2)
                        gk = local_k + k_off
                        gi = local_i
                        W[gk * n + gi, cr] += val
                block_idx += 1
        return U, V, W

    def verify(U, V, W):
        """Verify that sum_r U[i,j,r] V[j,k,r] W[k*n+i, r] = 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Start with the known rank-56 solution
    best_rank = 56
    best_U, best_V, best_W = build_strassen_kronecker(56)
    
    if not verify(best_U, best_V, best_W):
        # Fallback: trivial rank 64 (one per (i,j,k) triple)
        rank = n * m * p  # 64
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Attempt rank reduction via linear dependency analysis
    # Idea: if some rank terms are linearly dependent in a way that allows cancellation,
    # we might be able to reduce the rank. We check if the rank-56 solution has any
    # redundant terms by examining if removing a term and adjusting others works.
    
    # Try a structured rank-48 approach:
    # Instead of 8 independent Strassen blocks, use shared structure.
    # The key insight: for the 2x4x8 case, we can group the k-dimension.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split k into two groups of 4: k in {0,1,2,3} and k in {4,5,6,7}
    # C[i, 0:4] = A * B[:, 0:4]  (2x4 x 4x4)
    # C[i, 4:8] = A * B[:, 4:8]  (2x4 x 4x4)
    # Each 2x4 x 4x4 can be done with rank 24 (two Strassen 2x2x2 blocks in j)
    # Total: 2 * 24 = 48
    # But wait, that's the same as 8 blocks of 2x2x2 = 56. 
    # Actually 2x4 x 4x4 = two 2x2 x 2x4 blocks, each needs 2 Strassen blocks = 14
    # So 2x4 x 4x4 = 28, and 2x4 x 4x8 = 2*28 = 56. Same.
    
    # Let's try a different decomposition: use the fact that n=2 is small.
    # For each j, the contribution to C is: A[:,j] * B[j,:]
    # A[:,j] is a 2x1 vector, B[j,:] is a 1x8 vector.
    # Their outer product is 2x8, which has rank 1.
    # So C = sum_j A[:,j] B[j,:] is a sum of 4 rank-1 matrices.
    # This gives rank 4 for the matrix, but for the tensor decomposition we need
    # the bilinear form, so this doesn't directly help.
    
    # Try: can we do rank 49-55 by exploiting symmetries?
    # For now, return the rank-56 solution as it's verified.
    
    if time.time() - start_time > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt: check if we can remove one rank term and compensate
    # This is a heuristic - try each rank term and see if the remaining 55 can be adjusted
    # to still satisfy the equations. This is a nonlinear problem, so we use a simple check.
    
    # For each rank index r to try removing, set U[:,r]=0, V[:,r]=0, W[:,r]=0
    # and check if the remaining terms still sum to 1. This won't work directly
    # because removing a term breaks the sum. But we can check if the term is zero
    # (which it isn't in Strassen).
    
    # Alternative: use the structure to find a rank-55 decomposition.
    # This is non-trivial, so we stick with rank 56.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
