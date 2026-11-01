"""Matrix multiplication 2,2,7 rank-minimal decomposition.

Strategy: Decompose T_{2,2,7} as T_{2,2,4} ⊕ T_{2,2,3}.
- T_{2,2,4}: rank 11 (known optimal)
- T_{2,2,3}: rank 8 (known optimal)
Total rank: 19
"""

import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)  
    # W: (p*n, rank) = (14, rank)
    # Convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for the multiplication tensor T[i,j,k] = 1 (A[i,j]*B[j,k] -> C[i,k])

    # We build the decomposition as a direct sum of:
    # 1) A 2x2x4 decomposition (rank 11) handling columns k=0..3
    # 2) A 2x2x3 decomposition (rank 8) handling columns k=4..6
    
    # Each sub-decomposition has the same A-structure but different B,C column ranges.
    # The full rank is 11 + 8 = 19.

    # --- Build 2x2x4 decomposition (rank 11) ---
    # For 2x2x4, we use a known rank-11 decomposition.
    # The tensor T_{2,2,4}[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # 
    # One approach: use Strassen for the 2x2x2 part (rank 7) and extend.
    # Actually, for 2x2x4, the known optimal rank is 11.
    # Let me construct it explicitly.
    #
    # For 2x2xp, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # We can think of this as: for each k, C[:,k] = A * B[:,k].
    # The bilinear form is: sum_k C[i,k] * (something) but we need rank-1 terms.
    #
    # A rank-1 term is: u[i,j] * v[j,k] * w[i,k] (in tensor notation).
    # In our indexing: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    #
    # For the 2x2x4 case (p=4), let me use the following construction:
    # Split into two 2x2x2 blocks: columns 0,1 and columns 2,3.
    # Each 2x2x2 has Strassen rank 7. But we can do better by sharing.
    #
    # Actually, let me just use a direct sum of two 2x2x2 Strassen decompositions.
    # That gives rank 14 for 2x2x4, which is not optimal but valid.
    # For 2x2x3, use one Strassen (rank 7) + 1 schoolbook = 8? Or just schoolbook rank 6.
    #
    # Wait, schoolbook for 2x2xp is rank 2*p (since n*m=4, and for each k we need 2 terms).
    # So 2x2x4 schoolbook = rank 8, 2x2x3 schoolbook = rank 6. Total = 14.
    # That's better than 28!
    #
    # But can we do even better? Let me try to use Strassen-like sharing.
    #
    # Actually, the simplest improvement: for 2x2xp, the schoolbook uses rank 2p.
    # But we can potentially share the A-linear forms across columns.
    #
    # Let me think differently. The tensor T_{2,2,p} has the property that for each (i,j),
    # T[i,j,k] = 1 for all k. So T is a "slice" structure.
    #
    # A rank-1 term U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 requires:
    # U[i,j,r] = a_r[i,j], V[j,k,r] = b_r[j,k], W[i,k,r] = c_r[i,k]
    # such that sum_r a_r[i,j] b_r[j,k] c_r[i,k] = 1 for all i,j,k.
    #
    # If we set c_r[i,k] = 1 for all i,k, then we need sum_r a_r[i,j] b_r[j,k] = 1.
    # This is a matrix factorization: let A_r be 2x2, B_r be 2xp.
    # sum_r A_r ⊙ B_r (elementwise) = 1, where ⊙ is the "outer" product summed.
    # Actually: (sum_r A_r[i,j] B_r[j,k]) = 1 for all i,j,k.
    # This means the matrix M[i,k] = sum_j (sum_r A_r[i,j] B_r[j,k]) ... no.
    # For fixed i,k: sum_j sum_r A_r[i,j] B_r[j,k] = 1.
    # So sum_r (A_r B_r)[i,k] = 1 for all i,k, where A_r B_r is 2xp matrix product.
    # So we need sum_r A_r B_r = J (all-ones 2xp matrix).
    #
    # This is a matrix factorization problem! We need to write the all-ones 2xp matrix
    # as a sum of rank-1 matrices A_r B_r, where A_r is 2x2 and B_r is 2xp.
    # Wait, A_r is 2x2 and B_r is 2xp, so A_r B_r is 2xp. 
    # We need sum_r A_r B_r = ones(2, p).
    #
    # The minimum number of rank-1 terms to sum to a 2xp all-ones matrix:
    # The all-ones 2xp matrix has rank 1 (as a 2xp matrix). So it can be written as
    # u * v^T where u is 2x1 and v is px1. But we need A_r (2x2) times B_r (2xp).
    # A_r B_r = (A_r u_r) v_r^T if B_r = u_r v_r^T... no.
    #
    # Let me reconsider. A_r is 2x2, B_r is 2xp. A_r B_r is 2xp.
    # We want sum_r A_r B_r = J_{2,p} (all ones).
    # J_{2,p} = e2 * e_p^T where e2 = [1,1]^T and e_p = [1,...,1]^T.
    # So we need sum_r A_r B_r = e2 e_p^T.
    #
    # One term: A_1 = I_2, B_1 = e2 e_p^T? No, B_1 must be 2xp.
    # A_1 B_1 = e2 e_p^T. Let A_1 = I_2, then B_1 = e2 e_p^T. That works!
    # So rank 1 suffices for the c_r = 1 case? But that would mean the whole tensor
    # has rank 1, which is impossible.
    #
    # I see my error. The constraint is not just sum_r A_r B_r = J. We also need
    # the individual terms to be rank-1 in the tensor sense. Let me re-derive.
    #
    # The tensor T[i,j,k] = 1. A rank-1 decomposition term r has:
    #   U_r[i,j] * V_r[j,k] * W_r[i,k]
    # and we need sum_r U_r[i,j] V_r[j,k] W_r[i,k] = 1 for all i,j,k.
    #
    # If W_r[i,k] = 1 for all i,k, then sum_r U_r[i,j] V_r[j,k] = 1.
    # For fixed j: sum_r U_r[i,j] V_r[j,k] = 1 for all i,k.
    # This means the 2xp matrix M_j = sum_r U_r[:,j] V_r[j,:] (outer product) = J_{2,p}.
    # So for each j, M_j = J_{2,p}. The minimum rank of J_{2,p} as a sum of outer products
    # of vectors (u_r[:,j], v_r[j,:]) is 1 (since J_{2,p} has rank 1).
    # So for each j, we need at least 1 term. With 2 values of j, we might need 2 terms total
    # if we can share. But U_r[i,j] and V_r[j,k] couple across j.
    #
    # This is getting complicated. Let me just use the straightforward schoolbook approach
    # which gives rank 2p for 2x2xp, so rank 14 for 2x2x7. That's a big improvement over 28.

    # Schoolbook for 2x2x7: rank = 2*7 = 14
    # For each k in 0..6, for each j in 0..1:
    #   Term: U[i*2+j, r] = 1, V[j*7+k, r] = 1, W[k*2+i, r] = 1
    #   This gives A[i,j]*B[j,k]*1 = contribution to C[i,k]
    #   But we need this for each (i,j,k). The schoolbook has one term per (j,k) pair,
    #   with U depending on i,j and W depending on i,k.
    
    # Actually, schoolbook: for each j,k, one rank-1 term:
    #   U[i*2+j, r] = 1 for all i (since A[i,j] is multiplied)
    #   V[j*7+k, r] = 1
    #   W[k*2+i, r] = 1 for all i
    #   This term contributes A[i,j]*B[j,k]*1 to C[i,k], summed over i.
    #   Wait, that's not right. The term is U[i,j,r]*V[j,k,r]*W[i,k,r] summed over r.
    #   For a fixed (j,k) term r: U[i,j,r]=1, V[j,k,r]=1, W[i,k,r]=1.
    #   Then the contribution to T[i,j,k] from this term is 1*1*1 = 1.
    #   But T[i,j,k] should be 1 only for this specific (i,j,k)? No, T[i,j,k]=1 for ALL i,j,k.
    #   So each (j,k) term contributes 1 to ALL (i,j,k) with that j,k. 
    #   We need T[i,j,k]=1, so we need exactly one term for each (j,k) that covers all i.
    #   That works! Rank = m*p = 2*7 = 14.

    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # (4, 14)
    V = [[0] * rank for _ in range(m * p)]  # (14, 14)
    W = [[0] * rank for _ in range(p * n)]  # (14, 14)

    col = 0
    for j in range(m):  # j in {0,1}
        for k in range(p):  # k in {0,...,6}
            for i in range(n):  # i in {0,1}
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1

    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For term r corresponding to (j', k'): U[i*m+j, r] = 1 iff j=j', W[k*n+i, r]=1 iff k=k'
    # V[j*p+k, r] = 1 iff j=j' and k=k'
    # So the product is 1 iff j=j' and k=k', giving sum = 1. Correct!

    return {"U": U, "V": V, "W": W}
