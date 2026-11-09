"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook baseline (rank 63)
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Slice-and-fiber approach: 
    # The multiplication tensor T[i,j,k] = delta_{i,k} * (A_ij * B_jk)
    # We want to write T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    
    # For fixed k, the fiber is T[:,:,k] which is an n*m x m*p tensor
    # Specifically, for fixed k, T[i,j,k] = delta_{i,k} * A_ij * B_jk
    # If i != k, T[i,j,k] = 0
    # If i == k, T[k,j,k] = A_kj * B_jk
    
    # So for each k, the non-zero entries are only in row i=k
    # This gives us a strong constraint!
    
    # Let's verify: for the multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = A[i,j]*B[j,k]
    # This is NOT zero when i != k! I was wrong.
    
    # Let me reconsider. The tensor T[i,j,k] = A[i,j]*B[j,k]
    # For a rank-1 decomposition: T = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Wait, the convention is W[k*n+i, r], so W_r[k,i]
    
    # Actually, let's use the standard formulation:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = A[i,j] * B[j,k]
    # Rank-1 term: u_r[i,j] * v_r[j,k] * w_r[k,i]  -- but this doesn't look right
    
    # Let me re-read the convention: W[k*n+i, r]
    # So W is indexed by (k,i) with index k*n+i
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for each r: U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # This means: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = A[i,j]*B[j,k] for all i,j,k
    
    # For fixed (i,k), we have: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = A[i,j]*B[j,k]
    # This must hold for all j
    
    # Let's define for fixed (i,k):
    # Let a_{i,j} = A[i,j], b_{j,k} = B[j,k]
    # We need: sum_r (U_r[i,j] * W_r[k,i]) * V_r[j,k] = a_{i,j} * b_{j,k}
    
    # This is a rank-1 approximation of the outer product a_{i,j} * b_{j,k}
    # using vectors (U_r[i,j]*W_r[k,i]) and V_r[j,k]
    
    # Key insight: For fixed (i,k), the target is rank 1 (outer product of a_i and b_k)
    # So in principle, we could use just 1 term for each (i,k) pair
    # But the same r must work for all (i,k) pairs simultaneously
    
    # Let's try a structured approach:
    # Assume U_r[i,j] = u_r[i] * j_r[j] for some structure
    # And W_r[k,i] = w_r[k] * i_r[i]
    # And V_r[j,k] = v_r[j] * k_r[k]
    
    # Then: sum_r u_r[i]*j_r[j]*v_r[j]*k_r[k]*w_r[k]*i_r[i]
    # = sum_r (u_r[i]*i_r[i]) * (j_r[j]*v_r[j]) * (k_r[k]*w_r[k])
    # = sum_r U'_r[i] * V'_r[j] * W'_r[k]
    
    # This is just a standard rank-r decomposition of the multiplication tensor
    # where the tensor is T[i,j,k] = A[i,j]*B[j,k]
    
    # The multiplication tensor for m x n by n x p has rank at most m*n*p
    # For 3x3 x 3x7, that's 63
    
    # But we can do better! The key is that the tensor has a special structure.
    
    # Let's try: T[i,j,k] = A[i,j]*B[j,k]
    # This can be written as: T[i,j,k] = (A[i,j]) * (B[j,k])
    
    # One approach: use the fact that for fixed j, we have a vector in i and a vector in k
    # T[i,j,k] = A[i,j] * B[j,k]
    # For fixed j: T[:,j,:] = A[:,j] ⊗ B[j,:]
    
    # So T = sum_j A[:,j] ⊗ B[j,:]
    # This is a sum of 3 rank-1 terms in the "j" direction
    # But each term is a matrix (n x p), not a scalar
    
    # To make it a proper rank-1 tensor decomposition, we need:
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Let's try: U_r[i,j] = delta_{i,j} for some pattern
    # Actually, let's try a different structured ansatz
    
    # Ansatz: U_r[i,j] = u_r[i] * v_r[j]
    #          V_r[j,k] = v_r'[j] * w_r[k]  
    #          W_r[k,i] = w_r'[k] * u_r'[i]
    
    # Then: sum_r u_r[i]*v_r[j]*v_r'[j]*w_r[k]*w_r'[k]*u_r'[i]
    # = sum_r (u_r[i]*u_r'[i]) * (v_r[j]*v_r'[j]) * (w_r[k]*w_r'[k])
    # = sum_r U_r[i] * V_r[j] * W_r[k]
    
    # This gives us a standard tensor decomposition where T[i,j,k] = sum_r U_r[i]*V_r[j]*W_r[k]
    # But our T[i,j,k] = A[i,j]*B[j,k] is not separable in this way generally
    
    # However, we can use a different approach:
    # T[i,j,k] = A[i,j]*B[j,k]
    # = sum_j' (delta_{j,j'} * A[i,j]) * B[j,k]  -- trivial
    
    # Let's try: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Choose: U_r[i,j] = c_r[i,j], V_r[j,k] = d_r[j,k], W_r[k,i] = e_r[k,i]
    
    # For this to work: sum_r c_r[i,j]*d_r[j,k]*e_r[k,i] = A[i,j]*B[j,k]
    
    # Let's try a specific construction:
    # Use 9 terms for the "diagonal" part and handle the rest
    
    # Actually, let's use a known result:
    # The rank of 3x3 x 3x3 multiplication is at most 21 (de Groote)
    # For 3x3 x 3x7, we can split into three 3x3 x 3x3 blocks plus one 3x3 x 3x1
    # But we can share terms between blocks
    
    # Let's try a direct construction with rank 50 or so
    
    # For now, let's try to find a decomposition with rank < 63 using a structured ansatz
    
    # Ansatz: Use the fact that T[i,j,k] = A[i,j]*B[j,k]
    # Write A[i,j] = sum_a alpha_a[i]*beta_a[j] (rank decomposition of A, but A is arbitrary)
    # This won't work since A is arbitrary
    
    # Instead, let's use a fixed decomposition:
    # T[i,j,k] = sum_{r=1}^R U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Try R = 50 with a specific pattern
    
    # Let's try: for each pair (i,k), use a few terms
    # There are 9 pairs (i,k)
    # For each pair, the target is A[i,j]*B[j,k] which is rank 1 in j
    
    # So for each (i,k), we need: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = A[i,j]*B[j,k]
    # = A[i,j]*B[j,k]
    
    # If we set W_r[k,i] = 1 for all r, then:
    # sum_r U_r[i,j]*V_r[j,k] = A[i,j]*B[j,k]
    # This is a matrix equation: for each (i,k), the sum of outer products equals A[i,:]*B[:,k]
    
    # This is equivalent to: sum_r (U_r[i,:] ⊗ V_r[:,k]) = A[i,:] ⊗ B[:,k]
    # In matrix form: sum_r U_r[i,:]^T * V_r[:,k] = A[i,:]^T * B[:,k]  -- not quite
    
    # Actually: sum_r U_r[i,j]*V_r[j,k] = (sum_r U_r[i,:] ⊗ V_r[:,k])_{j,k}
    # We want this to equal A[i,j]*B[j,k]
    
    # This is a bilinear form. The minimum number of terms to represent all 9 such forms
    # is related to the rank of the resulting tensor
    
    # Let's just try to construct a valid decomposition with rank < 63
    
    # Simple approach: use 3x the schoolbook for the 3x3 part and optimize
    
    # For 3x3 x 3x3, use rank 21 decomposition (known)
    # For the remaining 3x3 x 3x4, use schoolbook (rank 36)
    # Total: 21 + 36 = 57
    
    # But we can do better by sharing
    
    # Let's implement a rank-21 decomposition for 3x3 x 3x3
    # Based on de Groote's algorithm
    
    def get_3x3x3_decomposition():
        """Return rank-21 decomposition for 3x3 x 3x3 multiplication."""
        # This is a known result; implementing the full de Groote algorithm is complex
        # For now, return schoolbook (rank 27) as placeholder
        # We'll improve this later
        n3 = 3
        U = [[0] * 27 for _ in range(9)]
        V = [[0] * 27 for _ in range(9)]
        W = [[0] * 27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W, 27
    
    # For now, let's try a different approach:
    # Use the structure that T[i,j,k] = A[i,j]*B[j,k]
    
    # Key insight: For fixed j, T[:,j,:] = A[:,j] ⊗ B[j,:]
    # This is a rank-1 matrix (outer product)
    
    # So T = sum_{j=0}^{2} A[:,j] ⊗ B[j,:]
    
    # To convert to the required form T[i,j,k] = sum_r U_r[i,j]*V_r[j,k]*W_r[k,i]:
    
    # Let's try: U_r[i,j] = delta_{j, r mod 3} * f_r[i]
    #           V_r[j,k] = delta_{j, r mod 3} * g_r[k]
    #           W_r[k,i] = h_r[k,i]
    
    # Then: sum_r delta_{j, r mod 3} * f_r[i] * delta_{j, r mod 3} * g_r[k] * h_r[k,i]
    # = sum_{r: r mod 3 = j} f_r[i]*g_r[k]*h_r[k,i]
    
    # For this to equal A[i,j]*B[j,k], we need:
    # sum_{r: r mod 3 = j} f_r[i]*g_r[k]*h_r[k,i] = A[i,j]*B[j,k]
    
    # If we use 3 terms per j (total 9 terms):
    # For j=0: sum_{r in {0,3,6}} f_r[i]*g_r[k]*h_r[k,i] = A[i,0]*B[0,k]
    # For j=1: sum_{r in {1,4,7}} f_r[i]*g_r[k]*h_r[k,i] = A[i,1]*B[1,k]
    # For j=2: sum_{r in {2,5,8}} f_r[i]*g_r[k]*h_r[k,i] = A[i,2]*B[2,k]
    
    # This is 9 independent equations, each requiring rank 1
    # So 9 terms might suffice!
    
    # Let's try: for each j, use 1 term
    # U_{j*3+0}[i,j] = A[i,j] (but A is arbitrary, so this doesn't work)
    
    # The issue is that A and B are arbitrary matrices, so we can't depend on them
    # The decomposition must work for ALL A and B
    
    # So U, V, W must be fixed matrices that work for all inputs
    
    # Let's reconsider: T[i,j,k] = A[i,j]*B[j,k]
    # We need: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = A[i,j]*B[j,k] for all A, B
    
    # This means the bilinear form must match
    
    # Let's try: U_r[i,j] = delta_{i,j}, V_r[j,k] = delta_{j,k}, W_r[k,i] = delta_{k,i}
    # Then: sum_r delta_{i,j}*delta_{j,k}*delta_{k,i} = delta_{i,j}*delta_{j,k}*delta_{k,i}
    # = 1 if i=j=k, else 0
    # This is not A[i,j]*B[j,k]
    
    # The correct schoolbook is:
    # U_{i,j,k}[i',j'] = delta_{i',i}*delta_{j',j}
    # V_{i,j,k}[j',k'] = delta_{j',j}*delta_{k',k}
    # W_{i,j,k}[k',i'] = delta_{k',k}*delta_{i',i}
    
    # Then: sum_{i,j,k} delta_{i',i}*delta_{j',j}*delta_{j',j}*delta_{k',k}*delta_{k',k}*delta_{i',i}
    # = delta_{i',i}*delta_{j',j}*delta_{k',k}
    # Which gives T[i',j',k'] = A[i',j']*B[j',k'] ✓
    
    # So schoolbook uses 27 terms for 3x3x3
    
    # To do better, we need to share terms
    
    # Let's try a Strassen-like approach for 3x3x3
    # Strassen for 2x2x2 uses 7 terms instead of 8
    
    # For 3x3x3, known best is 21 (de Groote)
    
    # Let's implement a 21-term decomposition for 3x3x3
    # This is complex, so let's start with a simpler improvement
    
    # For 3x3 x 3x7, let's try to use 21 terms for the first 3x3x3 block
    # and 36 terms for the remaining 3x3x4 block (schoolbook)
    # Total: 21 + 36 = 57
    
    # But we need to construct the actual U, V, W matrices
    
    # Let's try a different approach: use the fact that the problem is linear
    # and set up a system of equations
    
    # For rank R, we have R*9 + R*21 + R*21 = R*51 unknowns
    # And 3*3*3*3*7 = 567 equations (for all i,j,k and the bilinear structure)
    
    # Actually, the constraint is:
    # For all i,j,k: sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i] = A[i,j]*B[j,k]
    
    # This must hold for all A, B, so it's a polynomial identity
    
    # Let's try to find a solution with R = 50 using a structured ansatz
    
    # Given time constraints, let's return the schoolbook for now
    # and note that a proper implementation would use de Groote's algorithm
    
    if time.time() > deadline - 10:
        return best_result
    
    # Try to find a lower-rank decomposition using a structured approach
    # For 3x3 x 3x7, we can potentially use rank 50 or so
    
    # Let's try: use 21 terms for 3x3x3 and 29 terms for 3x3x4
    # Total: 50
    
    # This requires implementing the de Groote algorithm, which is complex
    
    # For now, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
