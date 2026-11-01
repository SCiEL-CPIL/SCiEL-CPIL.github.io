"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p = 52
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find a better construction: rank 3p+1 = 40
    # For 2x2 * 2xp, use the following construction:
    # For each k, we need:
    #   C[0,k] = a*B[0,k] + b*B[1,k]
    #   C[1,k] = c*B[0,k] + d*B[1,k]
    #
    # Use 3 columns per k plus 1 shared:
    # Column 3k:   (a+b) * (B[0,k]+B[1,k]) -> contributes to C[0,k] and C[1,k]
    # Column 3k+1: (a-b) * (B[0,k]-B[1,k]) -> helps separate
    # Column 3k+2: some correction term
    #
    # Actually, let's use a verified construction:
    # For each k:
    #   col 3k:   a * B[0,k]  -> C[0,k]
    #   col 3k+1: b * B[1,k]  -> C[0,k]
    #   col 3k+2: (c+d) * B[0,k] ... no, that's not rank-1 in the right sense.
    #
    # Let me try a different approach: use the fact that for 2x2, 
    # we can write the 4 products using 3 multiplications per "block" 
    # if we allow additions in U and V.
    #
    # Strassen-like for 2x2 * 2x1:
    # C[0] = a*b + c*d (where b=B[0], d=B[1])
    # C[1] = e*b + f*d (where e,c are A[1,0], A[1,1])
    #
    # Using: m1 = (a+f)*(b+d), m2 = (c+e)*b, m3 = a*(b-d), m4 = f*(d-c)
    # This gives rank 4 for 2x2*2x1, same as schoolbook.
    #
    # For 2x2 * 2xp, the minimal rank is believed to be 4p for small p,
    # but let's try a construction with rank 3p+1.
    #
    # Construction attempt for rank 3p+1:
    # Use columns 0..p-1 for: a * B[0,k] -> C[0,k]
    # Use columns p..2p-1 for: b * B[1,k] -> C[0,k]
    # Use columns 2p..3p-1 for: c * B[0,k] -> C[1,k]
    # Use column 3p for: d * B[1,k] -> C[1,k] ... but that's only 1 column for p terms.
    #
    # That doesn't work. Let me just verify schoolbook and return it.
    
    # Verify schoolbook correctness
    def verify(U, V, W, n, m, p):
        """Verify that the tensor decomposition is correct."""
        # The tensor T[i,j,k,r] should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta for correct decomposition
        # Actually, the multiplication tensor has T[i,j,k] = 1 if the output position (i,k) 
        # depends on input A[i,j] and B[j,k], else 0.
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (it's the identity in a sense)
        # No wait, the tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] = 1 
        # where i in [0,n), j in [0,m), k in [0,p), and the bilinear form is:
        # (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor)
        # And we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But W is indexed by (k*n+i) which is the output position.
        # Actually the convention is: W[k*n+i, r] means output C[i,k] gets contribution.
        # So the condition is: for each i,j,k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for invalid combinations (shouldn't happen with proper indexing), it should be 0.
        
        # Let's check: for each i in range(n), j in range(m), k in range(p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # And for other output positions, it should equal 0.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
                    # Check that it doesn't contribute to wrong output positions
                    for k2 in range(p):
                        if k2 != k:
                            s2 = 0
                            for r in range(len(U[0])):
                                s2 += U[i*m+j][r] * V[j*p+k2][r] * W[k2*n+i][r]
                            if s2 != 0:
                                return False
        return True
    
    if verify(U, V, W, n, m, p):
        pass  # Schoolbook is valid
    
    # Now try to construct a better decomposition
    # For 2x2 * 2xp, let's try rank = 3p + 1 = 40
    # 
    # Key insight: We can group the computation.
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    #
    # Let's define for each k:
    #   x_k = B[0,k], y_k = B[1,k]
    #   C[0,k] = a*x_k + b*y_k
    #   C[1,k] = c*x_k + d*y_k
    #
    # We can write:
    #   C[0,k] = (a+b)*x_k + b*(y_k - x_k)  ... not helpful
    #   C[0,k] = a*x_k + b*y_k
    #   C[1,k] = c*x_k + d*y_k
    #
    # Using shared terms:
    #   s_k = (a+b) * x_k  -> contributes to C[0,k] with coeff b... no
    #
    # Let me try: 
    #   m1_k = (a+b) * (x_k + y_k)
    #   m2_k = (a-b) * (x_k - y_k)
    #   m3_k = d * y_k
    #
    #   C[0,k] = a*x_k + b*y_k = ?
    #   m1_k + m2_k = (a+b)(x_k+y_k) + (a-b)(x_k-y_k) 
    #                = (a+b)x_k + (a+b)y_k + (a-b)x_k - (a-b)y_k
    #                = 2a*x_k + 2b*y_k
    #   So C[0,k] = (m1_k + m2_k) / 2
    #
    #   C[1,k] = c*x_k + d*y_k
    #   We need c*x_k. We have m2_k = (a-b)(x_k-y_k) = (a-b)x_k - (a-b)y_k
    #   This doesn't directly give c*x_k.
    #
    # This approach is getting complicated. Let me try a different grouping.
    #
    # Alternative: Use the fact that we can factor out common terms.
    # For columns 0..p-1: compute (a+b) * x_k, contributes to C[0,k] (coeff b) and C[1,k] (coeff c)... 
    # No, each rank-1 term is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    #
    # Let me just try a specific construction for rank 3p+1:
    #
    # Columns 0 to p-1: 
    #   U[0, r] = 1 (a), V[0*p+k, r] = 1 (B[0,k]), W[k*n+0, r] = 1 (C[0,k])
    #   This gives: a * B[0,k] -> C[0,k]
    #
    # Columns p to 2p-1:
    #   U[1, r] = 1 (b), V[1*p+k, r] = 1 (B[1,k]), W[k*n+0, r] = 1 (C[0,k])
    #   This gives: b * B[1,k] -> C[0,k]
    #
    # Columns 2p to 3p-1:
    #   U[2, r] = 1 (c), V[0*p+k, r] = 1 (B[0,k]), W[k*n+1, r] = 1 (C[1,k])
    #   This gives: c * B[0,k] -> C[1,k]
    #
    # Column 3p:
    #   U[3, r] = 1 (d), V[1*p+k, r] = 1 (B[1,k]) for all k, W[k*n+1, r] = 1 for all k
    #   But this is NOT rank-1 because V and W depend on k.
    #
    # So we can't combine the last term. We need 4p columns total.
    #
    # UNLESS we use a different decomposition. Let me think...
    #
    # What if we use:
    #   For each k: 3 columns
    #   Plus 1 extra column
    #
    # Column 3k:   (a+d) * B[0,k] -> contributes to C[0,k] (coeff a) and C[1,k] (coeff d)... 
    #   But a single rank-1 term U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] only has ONE i and ONE j.
    #   So U[0*m+0, r]=1 means it uses A[0,0]=a, and V[0*p+k, r]=1 means B[0,k], 
    #   and W[k*n+0, r]=1 means output C[0,k]. It can't simultaneously contribute to C[1,k].
    #
    # So each rank-1 term is: one A entry * one B entry * one C entry.
    # We need: for each (i,j,k) triple, the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for "wrong" output positions, it should be 0.
    #
    # The constraint is: U[i*m+j,r] * V[j*p+k,r] * W[k'*n+i',r] = 0 for (i',k') != (i,k)
    # when the term is supposed to contribute to C[i,k].
    #
    # Actually no, the full condition is:
    # For all i,j,k,k',i': 
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k'*n+i',r] = delta_{k,k'} * delta_{i,i'}
    #
    # This is a system of equations. Let me just go with schoolbook rank 52.
    
    return best_result
# EVOLVE-BLOCK-END
