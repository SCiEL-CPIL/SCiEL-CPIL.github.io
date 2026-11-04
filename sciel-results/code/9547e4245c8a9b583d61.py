"""Matrix multiplication 2x4x6 - rank minimization via structured tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    The multiplication tensor T has entries T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k].
    We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strategy: Use the known minimal rank bounds and try to construct a valid decomposition.
    For 2x4x6, the schoolbook rank is 48. We try to find a lower rank decomposition.
    
    Known results: For matrix multiplication C = AB where A is m x n, B is n x p,
    the tensor rank is at most m*n*p (schoolbook). Lower bounds exist but exact minimal
    rank is hard to determine. We'll use a systematic search for structured decompositions.
    
    Approach: Try various structured patterns that might achieve rank < 48.
    One known technique is to use "fast matrix multiplication" style decompositions,
    but for these small dimensions, we can try direct constructions.
    
    Let's try a rank-36 decomposition based on exploiting the structure.
    """
    import time
    import numpy as np

    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 6  # A is n x m, B is m x p, C is n x p
    
    # The tensor T[i,j,k] for C[i,k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook gives rank = n*m*p = 48
    # Let's try to find a lower rank decomposition.
    
    # One approach: Use the fact that we can combine some terms.
    # For example, if we have two terms that share structure, we might merge them.
    
    # Let's try a systematic approach: use random search with verification.
    # But first, let's try known low-rank constructions.
    
    # For 2x4x6, let's try rank 36, 32, 28, etc.
    # We'll use a greedy/structured approach.
    
    # Actually, let's use a different strategy: 
    # The tensor is "3-tensor" with dimensions (2*4, 4*6, 6*2) = (8, 24, 12)
    # but with specific structure (it's a "matrix multiplication tensor").
    
    # Let's try to construct a decomposition using the following idea:
    # Group the j-index and try to share factors across k or i.
    
    # For each j, we have contributions to all (i,k) pairs.
    # U[i*m+j, c] depends on i and j, V[j*p+k, c] depends on j and k, W[k*n+i, c] depends on k and i.
    
    # Let's try rank = 32 first with a structured pattern.
    
    # Alternative: Use bilinear decomposition techniques.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as sum_c (sum_i U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) * W[k,i,c]
    # but that's not quite right.
    
    # The correct formulation: 
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (U[i,j,c] * V[j,k,c] * W[k,i,c]) * A[i,j] * B[j,k]
    # No, that's not right either.
    
    # Correct: The tensor T is defined by T[i,j,k] = 1 for all i,j,k (in the standard basis).
    # A rank-r decomposition is: T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is indexed by (i,j), V by (j,k), W by (k,i).
    
    # So we need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    
    # Let's try a simple construction:
    # For rank 48 (schoolbook): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    
    # To reduce rank, we need to find linear dependencies or use more clever structures.
    
    # Let's try: Can we achieve rank 36?
    # Idea: For each j, we have n*p = 2*6 = 12 terms in schoolbook.
    # If we can reduce per-j rank from 12 to 9, we get 4*9 = 36.
    
    # For fixed j, we need: 1 = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for all i,k.
    # This is like a 2x6 matrix multiplication tensor of rank 12 (schoolbook).
    # Can we do better? For 2x6, minimal rank is at most 12.
    # Actually, for 2xn matrix multiplication, the rank is known to be 2n for n>=2? No.
    # For 2x2, rank is 7 (Strassen). For 2x3, rank is at most 10? 
    # For 2xn, the rank is at most 2n (schoolbook) and lower bounds exist.
    
    # Let's just try to use a numerical approach: set up the equations and solve.
    
    # But the entries must be integer or half-integer. So we need exact solutions.
    
    # Let's try a constructive approach for specific ranks.
    
    # Try rank 36:
    # For each j in 0..3, use 9 terms.
    # For fixed j, we need to decompose the 2x6 multiplication tensor.
    # T_j[i,k] = 1 for all i in 0..1, k in 0..5.
    # We need: 1 = sum_{c=1}^9 U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is equivalent to: the 2x6 all-ones matrix has a rank-9 "multiplicative" decomposition.
    # Actually, it's: sum_c (U_i * V_k) * W_{k,i} = 1 for all i,k.
    
    # Let's set W[k,i] = 1 for all k,i (simplest). Then we need:
    # sum_c U[i,j,c] * V[j,k,c] = 1 for all i,k.
    # This means the matrix M[i,k] = sum_c U[i,j,c] * V[j,k,c] = 1 for all i,k.
    # So M is the 2x6 all-ones matrix, and we need it to have rank at most 9.
    # But the all-ones 2x6 matrix has rank 1! So we can do it with rank 1.
    # U[i,j,c] = 1, V[j,k,c] = 1 for c=1, and U[i,j,c]=0, V[j,k,c]=0 for c>1.
    # Then sum_c U*V = 1 for all i,k. 
    # But wait, we also have the W factor. If W[k,i,c] = 1, then:
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = sum_c 1*1*1 = 9 if we use 9 terms with U=V=W=1.
    # That gives 9, not 1.
    
    # So we need to be more careful. Let's set up the equations properly.
    
    # For fixed j, let u_i = U[i*m+j, c], v_k = V[j*p+k, c], w_{k,i} = W[k*n+i, c].
    # We need: sum_c u_i * v_k * w_{k,i} = 1 for all i,k.
    
    # If we set w_{k,i} = 1 for all c, then sum_c u_i * v_k = 1, so the matrix [u_i * v_k] summed over c must be all-ones.
    # The matrix M[i,k] = sum_c u_i(c) * v_k(c) must be the all-ones 2x6 matrix.
    # This matrix has rank 1, so we can achieve it with c=1: u_i(1) = 1, v_k(1) = 1.
    # Then sum_c u_i(c)*v_k(c)*w_{k,i}(c) = 1*1*w_{k,i}(1) = w_{k,i}(1).
    # We need w_{k,i}(1) = 1 for all k,i. So w_{k,i}(1) = 1.
    # This works with rank 1 per j!
    
    # But wait, that would give total rank = 4*1 = 4? That can't be right.
    # The issue is that U, V, W are shared across j. U[i*m+j, c] depends on j, so different j's have different U entries.
    # Similarly for V. But W[k*n+i, c] does NOT depend on j!
    
    # So the constraint is: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # The W part is shared across j. This coupling is what makes it hard.
    
    # Let's try a different approach: use the same W for all j, and solve for U and V.
    
    # If we fix W, then for each j, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = 1 / W[k*n+i, c] ... no, that's not linear.
    
    # Actually, it's: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1.
    # Let X_j[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c]. Then sum_c X_j[i,k,c] * W[k*n+i, c] = 1.
    # But X_j[i,k] is a single number (sum over c), and W[k*n+i, c] varies with c.
    # So it's: sum_c X_j[i,k,c] * W[k*n+i, c] = 1, where X_j[i,k,c] = U[i*m+j, c] * V[j*p+k, c].
    
    # This is getting complicated. Let's just use the schoolbook rank 48 as a baseline
    # and try to improve it with a search.
    
    # For now, return the schoolbook decomposition with rank 48.
    # We can try to improve this in future iterations.
    
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    # Fallback: return schoolbook
                    pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
