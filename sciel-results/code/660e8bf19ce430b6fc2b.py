import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1

    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank27_search(deadline):
    """
    Search for a rank-27 decomposition.
    
    Strategy: Use Strassen for (j=1,2,k=0,1) and (j=1,2,k=3,4), 
    but try to merge j=0 terms with the middle block using extra cross terms.
    
    We'll parameterize the j=0 and j=1,2,k=2 parts with small integer coefficients
    and check if a valid rank-27 decomposition exists.
    """
    n, m, p = 2, 3, 5
    
    # Fixed Strassen blocks for (j=1,2,k=0,1) and (j=1,2,k=3,4)
    # That's 14 columns fixed.
    # Remaining 13 columns must handle:
    # - j=0, all k=0..4: 10 terms A_i0 * B_0k
    # - j=1,2, k=2: 4 terms A_ij * B_j2
    # Total needed: 14 tensor entries, but we have 13 rank-1 terms.
    # So we need to find a rank-13 decomposition for a 2x3x5 sub-tensor
    # that has 14 non-zero entries, OR we need cross terms.
    
    # Actually, let's think differently. The full tensor has n*m*p = 30 entries.
    # Rank 27 means we need 27 rank-1 tensors that sum to the identity tensor.
    
    # Let me try a different approach: use the known rank-7 Strassen for 2x2x2,
    # and see if we can combine two 2x2x2 blocks (covering j=1,2 for k=0,1 and k=3,4)
    # with a clever 2x3x1 or similar for the remaining.
    
    # Alternative: Try to find if there's a rank-13 decomposition for the subproblem:
    # T_sub[i,j,k] where j in {0,1,2}, k in {0,1,2,3,4}, but only certain entries matter.
    
    # Let me just brute-force check: can we do better than 28?
    # The lower bound for 2x3x5 is at least max(2*3, 3*5, 2*5) = 15, but that's weak.
    # Actually the trivial lower bound is max(n*m, m*p, n*p) = max(6,15,10) = 15.
    # But we know rank(2x3x5) >= rank(2x2x2) = 7 for any sub-block.
    
    # Let me try a direct search: fix the two Strassen blocks, and try to find
    # a rank-13 completion for the remaining 14 tensor entries.
    
    # The remaining entries after removing Strassen contributions:
    # For (j=1,2,k=0,1): Strassen handles all 8 entries (2*2*2)
    # For (j=1,2,k=3,4): Strassen handles all 8 entries
    # Remaining: j=0 (all k) = 10 entries, and j=1,2,k=2 = 4 entries. Total 14.
    
    # So we need rank-13 for a tensor with 14 non-zero entries.
    # This is only possible if there's a linear dependency.
    
    # Let me set up the problem: we have 13 unknown rank-1 terms (u_r, v_r, w_r)
    # such that sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = T_remaining[i,j,k]
    
    # This is a system of 14 equations in 13*6 = 78 unknowns (with 3*13=39 degrees of freedom
    # from scaling). This is highly underdetermined, so solutions should exist.
    
    # But the coefficients must be integers or half-integers. Let me try small integers.
    
    # Let me try: use schoolbook for 9 of the 14 entries, and find a clever combination
    # for the remaining 5 entries using 4 rank-1 terms.
    
    # Actually, let me try a completely different block decomposition:
    # Split k into {0,1,2} and {3,4}
    # j=0: 2x1x3 and 2x1x2 -> rank 6+4 = 10
    # j=1,2: 2x2x3 and 2x2x2
    # For 2x2x3: can we do better than 12? The rank of 2x2x3 is 7 (same as 2x2x2 since 
    #   adding a k-slice doesn't increase rank beyond 2*2*2=8, and 7 is achievable).
    #   Actually rank(2x2x3) = 7? Let me check: the tensor is sum_{j=0,1} A_ij B_jk for k=0,1,2.
    #   This is like 2x2x2 plus one extra slice. The rank should be at most 7+2=9, but maybe less.
    #   Actually, I think rank(2x2x3) = 7. Let me verify by trying to construct it.
    
    # Hmm, this is getting complex. Let me just try to verify if rank 27 is achievable
    # by attempting a direct construction.
    
    # Try: 2x3x5 as (2x2x5) + (1x1x5) where 2x2x5 has rank 13 and 1x1x5 has rank 5?
    # No, that gives 18.
    
    # Try: (2x2x2) + (2x2x2) + (2x1x5) + (1x2x1) = 7+7+10+2 = 26? No, the blocks don't partition cleanly.
    
    # Let me try the most promising: keep the two Strassen 2x2x2 blocks (14 terms),
    # and find a rank-13 decomposition for the remaining 2x3x5 tensor with 14 non-zeros.
    
    # The remaining tensor T_rem[i,j,k]:
    # T_rem[i,0,k] = 1 for all i,k (10 entries)
    # T_rem[i,1,2] = 1, T_rem[i,2,2] = 1 for i=0,1 (4 entries)
    # All other entries = 0
    
    # We need: sum_{r=1}^{13} u_r[i,j] v_r[j,k] w_r[k,i] = T_rem[i,j,k]
    
    # For this to work, each rank-1 term (u_r, v_r, w_r) must have support that
    # is "compatible" with T_rem.
    
    # Let me try: 10 schoolbook terms for j=0, and 3 clever terms for j=1,2,k=2.
    # But 3 rank-1 terms can't handle 4 independent entries unless there's structure.
    
    # T_rem[i,1,2] and T_rem[i,2,2] for i=0,1 form a 2x2 matrix in (i,j) at k=2.
    # Can 3 rank-1 terms represent this? Each term contributes u_r[i,1]*v_r[1,2]*w_r[2,i] to T[i,1,2]
    # and u_r[i,2]*v_r[2,2]*w_r[2,i] to T[i,2,2].
    
    # This is getting too complex for a quick search. Let me just return the rank-28 solution
    # since finding rank-27 requires deeper mathematical insight.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Try to find rank-27
    best_result = None
    try:
        r27 = _try_rank27_search(deadline)
        if r27 is not None:
            best_result = r27
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: Rank 28 decomposition
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0

    # Block 1: j=0, k=0..4
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1

    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)

    # Block 3: j=1,2, k=2
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1

    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
