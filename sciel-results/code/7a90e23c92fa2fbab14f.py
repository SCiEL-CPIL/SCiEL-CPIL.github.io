import numpy as np
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to construct rank 21 decomposition
    # Based on 3 copies of rank-7 2x2x2 decomposition
    rank_target = 21
    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]
    
    # Define the 7 terms for one 2x2 block
    # U indices: 0,1,2,3 (i,j)
    # V indices: 0,1,2,3 (j,k)
    # W indices: 0,1,2,3 (k,i) -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Note: Task W index is k*n+i.
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    
    # Strassen-like terms (need to be verified)
    # I will use a known valid set if I can find one, else fallback.
    # Since I don't have a verified set, I will just return schoolbook.
    # But I can try to verify a candidate.
    
    # Candidate from literature (rank 7):
    # This is a known rank-7 decomposition.
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([0, 0, 1, -1], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [1, 0, 1, 0], [1, 0, -1, 0])
    ]
    
    # Map to U, V, W for the 3 blocks
    # Block l corresponds to columns 2l, 2l+1 of B and C.
    for l in range(3):
        for r_local in range(7):
            r_global = l * 7 + r_local
            u_vec, v_vec, w_vec = terms[r_local]
            
            # Fill U
            for idx in range(4):
                if u_vec[idx] != 0:
                    i = idx // 2
                    j = idx % 2
                    U[i * m + j][r_global] = u_vec[idx]
            
            # Fill V
            for idx in range(4):
                if v_vec[idx] != 0:
                    j = idx // 2
                    k = idx % 2
                    V[j * p + (2 * l + k)][r_global] = v_vec[idx]
            
            # Fill W
            for idx in range(4):
                if w_vec[idx] != 0:
                    k = idx // 2
                    i = idx % 2
                    W[(2 * l + k) * n + i][r_global] = w_vec[idx]
    
    # Verify decomposition
    # T[i,j, j',k', k'',i''] should be 1 if j=j', k=k'', i=i''
    # Sum over r U[i m + j, r] V[j p + k, r] W[k n + i, r]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank_target):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                # The tensor T has 1 at (i,j, j,k, k,i)
                # So for fixed i,j,k, the sum should be 1?
                # No, the sum is over r.
                # The term is U[i m + j, r] V[j p + k, r] W[k n + i, r].
                # This product is non-zero only if the indices match.
                # In the schoolbook, for a specific (i,j,k), only one r is non-zero.
                # In the decomposition, multiple r might contribute.
                # The sum must equal 1 if the indices are consistent?
                # The tensor T_{(i,j), (j',k'), (k'',i'')} = 1 if j=j', k=k'', i=i''.
                # Here we are summing for a specific (i,j,k).
                # The indices in V are j,k. In W are k,i.
                # So the term is non-zero if the r-th tensor has support on (i,j), (j,k), (k,i).
                # The sum of these products must equal 1.
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
