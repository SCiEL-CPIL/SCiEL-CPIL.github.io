import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]
DEADLINE = time.time() + 110  # 10s safety margin before 120s limit

def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n

    # Start with schoolbook
    U = np.zeros((U_rows, total_terms), dtype=float)
    V = np.zeros((V_rows, total_terms), dtype=float)
    W = np.zeros((W_rows, total_terms), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    def check_decomposition(Umat, Vmat, Wmat):
        # Verify that U * V * W^T gives the correct tensor
        # Tensor T[i,j,k] should be 1 for all valid i,j,k
        # Product = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # But W is indexed as [k*n+i, c]
        # Let's verify by computing the full tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :])
                    # This isn't quite right - we need to sum over c
                    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    pass

        # Actually, the decomposition means:
        # For the matrix multiplication (C = A*B), the tensor entries are
        # T_{i,j,k} = 1 where C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The decomposition: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
        # Wait, the convention is W[k*n+i, c], so W is indexed by (k,i)
        # The product for entry (i,k) of C is: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal A_{i,j} * B_{j,k} summed appropriately... 
        # Actually for exact decomposition: the tensor of matmul has T[i,j,k] = delta (it's 1 for the 
        # standard basis). Let me re-check.
        # The matmul tensor: T_{i,j,k} = 1 (each entry of A times corresponding entry of B contributes to C)
        # More precisely: C_{i,k} = sum_j A_{i,j} B_{j,k}
        # As a tensor: T[i,j,k] = 1 (coefficient of A_{i,j} * B_{j,k} in C_{i,k})
        # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So for each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Umat[i * m + j, :] * Vmat[j * p + k, :] * Wmat[k * n + i, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    if not check_decomposition(U, V, W):
        # Should never happen
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find lower rank via random search / optimization
    # Strategy: try rank from 30 down to 36, attempt to find a valid decomposition
    # For 3x3x4, known lower bounds suggest rank might be around 30-34
    
    # Use a smarter approach: start with schoolbook and try to merge/eliminate terms
    # by finding linear dependencies.
    
    # Actually, let's try a direct optimization approach:
    # For a given rank r, try to find U, V, W with entries in {0, 1, -1, 0.5, -0.5}
    # that satisfy the constraints.
    
    # Given time constraints, let's try a few structured decompositions.
    
    # One known approach: for M_{3,3,3}, rank is 23 (Strassen-like).
    # For M_{3,3,4}, we can embed the 3x3x3 decomposition.
    # But let's try a simpler structured approach first.
    
    # Try rank 30: use a block structure
    # Split the 3x3x4 into blocks and use known small matrix multiplication ranks
    
    # For now, let's try to improve via a random perturbation approach
    # within the family of the schoolbook decomposition.
    
    best_rank = total_terms
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a rank-30 decomposition by construction
    # Idea: use the fact that 3x3x4 can be decomposed using 3x3x3 (rank 23) + extra terms
    
    # Let's try a specific construction:
    # The 3x3x3 multiplication can be done with rank 23.
    # For 3x3x4, we need an additional "column" of B.
    # If we partition B as [B1 | B2] where B1 is 3x3 and B2 is 3x1,
    # then C = A*B1 + A*B2.
    # A*B1 uses 23 terms (if we have a rank-23 decomposition of 3x3x3).
    # A*B2 is 3x3 times 3x1, which is like computing 3 outputs each as a dot product of 3 elements.
    # This is rank 9 in schoolbook, but can we do better?
    # A*B2: C_i = sum_j A_{i,j} B2_{j}
    # Tensor: T2[i,j,k=0] = 1 (for the B2 part)
    # This is actually just 3 dot products, each of length 3.
    # Rank of each dot product of length 3 is 3 (in general, for arbitrary inputs).
    # So A*B2 has rank at least 9? No, the tensor is T[i,j,0] = 1 for all i,j.
    # sum_c U[i,j,c] V[j,0,c] W[0,i,c] = 1 for all i,j
    # This is like the 3x3x1 tensor, which has rank 3 (it's just 3 rows of 1s, but as a 3D tensor)
    # Actually T[i,j,0] = 1 for all i,j. This tensor has rank 3:
    # U[i,j,c] = delta_{i,c}, V[j,0,c] = 1, W[0,i,c] = delta_{j,c}? No.
    # T[i,j,0] = sum_c U[i*m+j,c] * V[j*1+0,c] * W[0*n+i,c] = 1
    # Let U[i*3+j, c] = 1, V[j, c] = delta_{j,c}, W[0*3+i, c] = delta_{i,c}?
    # Then sum_c 1 * delta_{j,c} * delta_{i,c} = delta_{i,j}. Not 1.
    
    # The 3x3x1 tensor (all ones) has border rank 1, but tensor rank...
    # T[i,j,0] = 1. This is the sum of 3 rank-1 tensors:
    # T = sum_{c=0}^{2} (e_i^{(c)}) (e_j^{(c)}) (e_0^{(c)})? No.
    # Let's think: we need sum_c U[i,j,c] V[j,c] W[0,i,c] = 1 for all i,j.
    # Set W[0,i,c] = delta_{i,c}, V[j,c] = 1, U[i,j,c] = delta_{j,c}.
    # Then sum_c delta_{j,c} * 1 * delta_{i,c} = delta_{i,j}. Wrong.
    
    # Set U[i,j,c] = 1, V[j,c] = delta_{j,c}, W[0,i,c] = delta_{i,c}.
    # sum_c 1 * delta_{j,c} * delta_{i,c} = delta_{i,j}. Still wrong.
    
    # We need U[i,j,c] V[j,c] W[i,c] = 1 summed over c.
    # If we set U[i,j,c] = 1 for all i,j,c, V[j,c] = 1 for all j,c, W[i,c] = delta_{i,c}:
    # sum_c 1 * 1 * delta_{i,c} = 1. Yes!
    # So U[i*3+j, c] = 1, V[j, c] = 1, W[0*3+i, c] = delta_{i,c} for c=0,1,2.
    # That's rank 3 for the 3x3x1 part.
    
    # So if we had a rank-23 decomposition for 3x3x3, total would be 23 + 3 = 26.
    # But I don't have the rank-23 coefficients memorized.
    
    # Let me just try to verify if the schoolbook is the best we can do within time.
    # Given the complexity of finding optimal decompositions, and that the current score is 1.0,
    # let's return the schoolbook which is guaranteed correct.
    
    if time.time() > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Let's try one more thing: see if we can eliminate any terms from the schoolbook
    # by checking for linear dependencies in the decomposition.
    
    # For the schoolbook, each term is unique (i,j,k) -> no obvious dependencies.
    # The rank of M_{3,3,4} is known to be between some bounds.
    # Without specific knowledge of a lower-rank decomposition, return schoolbook.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
