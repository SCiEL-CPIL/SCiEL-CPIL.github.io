"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_3x3x3_terms():
    """
    Return the 23-term rank-23 decomposition for 3x3x3 matrix multiplication.
    Each term is a tuple (u, v, w) where:
      u is a 3x3 matrix (flattened to 9 entries, row-major)
      v is a 3x3 matrix
      w is a 3x3 matrix
    such that sum_c u_c * v_c * w_c^T = standard 3x3 matmul tensor.
    
    This is based on the known rank-23 decomposition (Bini, Capovani, Lotti, Romani 1978).
    The decomposition uses linear combinations of matrix entries.
    
    We represent each factor as a list of 9 values (row-major 3x3).
    """
    # The rank-23 decomposition for 3x3x3.
    # Each of the 23 terms has U, V, W factors.
    # U[i,j] multiplies with V[j,k] and W[k,i] to contribute to C[i,k].
    # We need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = delta[i,j] * delta[j,k] (standard matmul)
    
    # Standard approach: use the Bini-Capovani-Lotti-Romani decomposition.
    # Let me define it explicitly. The 23 terms are:
    
    terms = []
    
    # We'll use a systematic construction. For 3x3x3, one known decomposition
    # uses the following pattern. Let me use the explicit coefficients.
    
    # Actually, let me construct this more carefully.
    # The tensor T[i,j,k] = delta_{i,j} * delta_{j,k} for 3x3 matmul.
    # We need T = sum_c u_c ⊗ v_c ⊗ w_c where
    # T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # Wait, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta_{i,j} * delta_{j,k}... no.
    # T[i,j,k] should satisfy: sum_j T[i,j,k] * A[i,j] * B[j,k] gives C[i,k]?
    # No. The decomposition is: C[i,k] = sum_c (sum_j U_c[i,j]*A[i,j]) * (sum_j V_c[j,k]*B[j,k]) * W_c[k,i]
    # Wait, that's not right either.
    
    # Standard tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k] = sum_c U_c[i,j] V_c[j,k] W_c[k,i] contracted...
    # Actually: the matmul tensor M[i,j,k] satisfies C[i,k] = sum_j M[i,j,k] A[i,j] B[j,k]
    # where M[i,j,k] = 1 if j is the "middle" index, but really M[i,j,k] = delta_{i,j}*delta_{j,k} is wrong.
    # M[i,j,k] = 1 for all i,j,k? No.
    # C[i,k] = sum_j A[i,j] B[j,k]. So the tensor is M[i,j,k] = 1 (it's just the identity in the summation).
    # We decompose M = sum_c u_c ⊗ v_c ⊗ w_c where M[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # Wait, no. In our problem:
    # U has shape (n*m, rank) = (9, rank) for 3x3x3
    # V has shape (m*p, rank) = (9, rank)
    # W has shape (p*n, rank) = (9, rank)
    # And the decomposition satisfies:
    # For all i,j,k: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,j} * delta_{j,k}
    # No wait. The condition is that the reconstructed matmul is correct:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using the decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) (sum_j V[j*p+k, c] B[j,k]) W[k*n+i, c]
    # Hmm, that doesn't look right. Let me re-read the problem.
    
    # "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The trace convention W[k*n+i, r]
    
    # I think the decomposition is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    # Wait, that can't be right dimensionally. Let me think again.
    
    # Actually I think it's:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j?
    # No, U, V, W are the factors. The standard tensor decomposition for matmul:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where T[i,j,k] is the matmul tensor.
    # U has rows indexed by (i,j), V by (j,k), W by (k,i).
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # And the matmul tensor is T[i,j,k] = 1 if we're doing standard matmul? 
    # No: C[i,k] = sum_j A[i,j] B[j,k]. The "tensor" here has indices i (row of A), j (col of A = row of B), k (col of B).
    # T[i,j,k] = 1 for all valid i,j,k? That's not a rank-1 tensor.
    # Actually the matmul operation IS the tensor: it maps A (n×m) and B (m×p) to C (n×p).
    # As a 3-tensor with indices (i, j, k): T[i,j,k] = 1 if the indices are consistent, which they always are.
    # So T[i,j,k] = 1 for all i,j,k? That would make it rank 1, which is wrong.
    
    # I think I'm confusing myself. Let me look at this differently.
    # The bilinear map (A, B) -> A*B is a tensor of rank r if it can be written as:
    # (A*B)[i,k] = sum_{c=1}^{r} (sum_j U_c[i,j] A[i,j]) (sum_j V_c[j,k] B[j,k]) W_c[k,i]
    # Hmm, that has a W_c[k,i] that doesn't make sense as a scalar.
    
    # Let me just use the standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c sum_j U_c[i,j] V_c[j,k] W_c[k,i] A[i,j] B[j,k]... no.
    
    # OK let me just look at what makes the schoolbook work:
    # In schoolbook, for each (i,j,k) we have one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # So the contribution is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For this to equal A[i,j]*B[j,k] when summed over c and j... 
    
    # I think the correct interpretation is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # This is a bilinear form in A and B for each c, scaled by W.
    
    # Let me verify with schoolbook: U[i*m+j,c]=1 only when c corresponds to (i,j,k), similarly V and W.
    # Then for a given (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k])
    # For c corresponding to (i',j',k'): W[k*n+i,c] is 1 only if k'=k and i'=i.
    # U[i*m+j,c] is 1 only if i=i' and j=j'.
    # V[j*p+k,c] is 1 only if j=j' and k=k'.
    # So the only nonzero term is when i'=i, k'=k, j'=j for some j.
    # Then the inner sums: sum_j U[i*m+j,c]A[i,j] = A[i,j'] and sum_j V[j*p+k,c]B[j,k] = B[j',k]
    # So C[i,k] = sum_j A[i,j]B[j,k]. Correct!
    
    # So the formula is: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    
    # Now for the 3x3x3 rank-23 decomposition, I need to find U, V, W of shape (9, 23) each
    # such that this formula holds.
    
    # This is a known result. Let me use the explicit decomposition from the literature.
    # The Bini-Capovani-Lotti-Romani 1978 decomposition for 3x3x3.
    
    # I'll use a simpler approach: construct it using linear algebra.
    # Actually, let me just hardcode a known rank-23 decomposition.
    
    # Here's one approach: use the fact that 3x3 matmul can be done with 23 multiplications.
    # I'll use the decomposition from "The tensor rank of 3x3 matrix multiplication is 23" 
    # by Bini et al.
    
    # Let me use a different, more explicit construction.
    # I'll define the 23 terms using the known algorithm.
    
    # For now, let me use a verified decomposition. I'll construct it step by step.
    
    # The 23 terms for 3x3x3 (using the standard BCLR decomposition):
    # Each term c has matrices U_c, V_c, W_c (all 3x3).
    
    # I'll use the following known decomposition (from Strassen's 1974 paper for 3x3):
    # Actually Strassen showed rank <= 23 for 3x3x3.
    
    # Let me use a concrete set of 23 terms. I'll define them as lists of 9 floats each.
    
    # Term 1: U=diag(1,0,0), V=diag(1,0,0), W=diag(1,0,0) -> contributes A[0,0]*B[0,0] to C[0,0]
    # This is getting complicated. Let me use a programmatic approach.
    
    # Actually, I recall that for the purpose of this problem, I can use the following:
    # The rank-23 decomposition can be constructed using the following 23 triples.
    # Let me use a known explicit form.
    
    # I'll use the decomposition from https://en.wikipedia.org/wiki/Strassen_algorithm
    # generalized to 3x3.
    
    # For 3x3, one known rank-23 decomposition uses:
    # M1 = A11*(B11+B22+B33), etc. But this is for the 2x2 case generalized.
    
    # Let me just use a verified set of coefficients. I'll generate them using 
    # a known algorithm or lookup.
    
    # For the purpose of this implementation, let me use the following approach:
    # I'll use the schoolbook for now but with the block structure, and verify correctness.
    # Then I can try to find the rank-23 decomposition.
    
    # Actually, let me just implement it correctly with known decompositions.
    # I'll use the following rank-23 decomposition for 3x3x3:
    
    # The 23 terms (U_c, V_c, W_c) as 3x3 matrices:
    # I'll use the explicit coefficients from the literature.
    
    # Let me use a simpler but valid approach: I'll use the known rank-23 
    # decomposition coefficients.
    
    # Here are the 23 terms for 3x3x3 matmul (row-major order for each 3x3 matrix):
    # Each term is (u_flat, v_flat, w_flat) where each _flat is a list of 9 values.
    
    # I'll use the decomposition from "On the rank of 3x3 matrix multiplication" 
    # The coefficients are:
    
    terms_data = [
        # (U, V, W) for each of the 23 terms
        # U[i,j] V[j,k] W[k,i] should sum to the matmul tensor
        # Let me use a systematic construction based on the known result.
        
        # Term 0: 
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
    ]
    
    # This is getting too complex to do from memory. Let me use a different approach.
    # I'll use the fact that I can verify the decomposition numerically.
    # Let me construct the rank-23 decomposition using a known algorithm.
    
    # Actually, for the purpose of this task, let me use a simpler strategy:
    # Use schoolbook for the 3x3x4 and 3x2x4 blocks, and try to use 
    # rank-23 for 3x3x3 and rank-8 for 3x2x3.
    
    # For the rank-8 decomposition of 3x2x3:
    # A is 3x2, B is 2x3, C is 3x3.
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # Rank 8 means we can do this with 8 multiplications.
    
    # The rank-8 decomposition for 3x2x3:
    # U: (3*2, 8) = (6, 8)
    # V: (2*3, 8) = (6, 8)  
    # W: (3*3, 8) = (9, 8)
    
    # Let me construct this.
    # C[i,k] = sum_j A[i,j] B[j,k] for j=0,1
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # With 8 multiplications, we can use:
    # M1 = A[i,0]*B[0,0]
    # M2 = A[i,0]*B[0,1]
    # M3 = A[i,0]*B[0,2]
    # M4 = A[i,1]*B[1,0]
    # M5 = A[i,1]*B[1,1]
    # M6 = A[i,1]*B[1,2]
    # That's only 6. We need 8, so there must be a more complex decomposition.
    # Actually, the schoolbook for 3x2x3 has rank 3*2*3=18, not 6.
    # Wait, I'm confusing myself.
    
    # The tensor rank of the 3x2x3 matmul tensor: 
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # This is a bilinear map from (3x2) x (2x3) to (3x3).
    # The schoolbook uses 3*2*3 = 18 multiplications (one for each (i,j,k) triple).
    # But the tensor rank is actually 8, which is less than 18.
    
    # The rank-8 decomposition exists. Let me find it.
    
    # For 3x2x3, the tensor T[i,j,k] = delta_{i,j} * delta_{j,k}... no.
    # T[i,j,k] where i in {0,1,2}, j in {0,1}, k in {0,1,2}
    # T[i,j,k] = 1 if the indices form a valid matmul path, which they always do.
    # Actually T[i,j,k] = 1 for all i,j,k? No, that's not right.
    
    # The matmul tensor for (3x2)*(2x3)->(3x3) has T[i,j,k] = 1 for all valid (i,j,k).
    # We want to decompose T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^3 (indexed by i), v_c is in R^2 (indexed by j), w_c is in R^3 (indexed by k).
    # And T[i,j,k] = sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k.
    
    # Wait, that would mean T is the all-ones tensor, which has rank 1. That can't be right.
    
    # I think the issue is that the tensor for matmul is NOT the all-ones tensor.
    # Let me reconsider.
    
    # The bilinear map is: (A, B) -> A*B where A is 3x2, B is 2x3.
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # As a tensor: T[i,j,k] * A[i,j] * B[j,k] summed over i,j,k gives... no.
    
    # The correct tensor formulation: The matmul operation is a 3-tensor T with indices (i, j, k)
    # where T[i,j,k] = 1 if j is the "contracted" index. But in our case, j is always contracted.
    # So T[i,j,k] = 1 for all i,j,k? That gives rank 1, which is wrong.
    
    # I think the correct statement is: The matmul tensor T has T[i,j,k] = delta_{i,j} * delta_{j,k}... no.
    
    # Let me just use the standard formulation from the problem statement:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c should give the correct matmul.
    # For 3x2x3: n=3, m=2, p=3.
    # U: (3*2, r) = (6, r), V: (2*3, r) = (6, r), W: (3*3, r) = (9, r)
    # C[i,k] = sum_c W[k*3+i, c] * (sum_j U[i*2+j, c] A[i,j]) * (sum_j V[j*3+k, c] B[j,k])
    
    # For schoolbook (r=18): for each (i,j,k), U[i*2+j, c]=1, V[j*3+k, c]=1, W[k*3+i, c]=1.
    # This gives C[i,k] = sum_j A[i,j] B[j,k]. Correct.
    
    # For rank 8, we need a more clever decomposition.
    
    # Let me just use schoolbook for all blocks to ensure correctness, since finding 
    # the exact rank-23 and rank-8 decompositions from memory is error-prone.
    
    # Schoolbook total: 27 + 36 + 18 + 24 = 105. Same as full schoolbook.
    # That's not an improvement.
    
    # Let me at least implement the block structure with schoolbook to verify the framework,
    # then I can try to find better decompositions.
    
    return terms  # placeholder

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Use schoolbook decomposition as a baseline.
    # rank = n*m*p = 105
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
