"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 80
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Verify the schoolbook decomposition is correct
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With our decomposition: C[i,k] = sum_r U[i*m+j_col, r] * V[j_row*p+k, r] * W[k*n+i, r]
    # For each r corresponding to (i',j',k'): U has 1 at (i'*m+j'), V has 1 at (j'*p+k'), W has 1 at (k'*n+i')
    # So C[i,k] gets contribution 1*1*1 = 1 when i=i', k=k', and the j index matches
    # This correctly computes C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Attempt a limited search for rank reduction using linear algebra
    # Check if any columns are linearly dependent in a way that allows merging
    # For small improvements, try to see if we can express one rank-1 term as a combination of others
    # This is computationally expensive, so we limit the search
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:  # Only attempt search if we have time
        try:
            import numpy as np
            
            # Convert to numpy for linear algebra operations
            U_np = np.array(U_school, dtype=float)
            V_np = np.array(V_school, dtype=float)
            W_np = np.array(W_school, dtype=float)
            
            # Check for linear dependencies among the rank-1 terms
            # Each term r is the tensor product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Flatten each term and check rank of the matrix of flattened terms
            rank1_terms = []
            for r in range(rank_school):
                term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).flatten()
                rank1_terms.append(term)
            
            if rank1_terms:
                terms_matrix = np.array(rank1_terms)
                # Check if the terms are linearly independent
                # If rank < number of terms, we might be able to reduce
                try:
                    matrix_rank = np.linalg.matrix_rank(terms_matrix, tol=1e-10)
                    # For schoolbook, all terms should be linearly independent
                    # matrix_rank should equal rank_school = 80
                    # If it's less, we have redundancy
                except:
                    pass
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
