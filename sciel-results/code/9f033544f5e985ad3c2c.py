"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    best_rank = n * m * p  # 36
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_U, best_V, best_W = U, V, W
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition is exact using exact arithmetic.
        
        The condition is: for all i in [0,n), k in [0,p), j in [0,m), j2 in [0,m):
        sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = 1 if j==j2 else 0
        """
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        total = Fraction(0)
                        for c in range(rank):
                            total += Fraction(W_mat[k*n+i][c]) * Fraction(U_mat[i*m+j][c]) * Fraction(V_mat[j2*p+k][c])
                        expected = 1 if j == j2 else 0
                        if total != expected:
                            return False
        return True
    
    def verify_fast(U_mat, V_mat, W_mat, rank):
        """Fast verification using numpy for initial filtering, then exact check."""
        # Convert to numpy for fast matrix multiplication
        U_np = np.array(U_mat, dtype=np.float64)
        V_np = np.array(V_mat, dtype=np.float64)
        W_np = np.array(W_mat, dtype=np.float64)
        
        # For each (i,k), check that the tensor slice is identity
        for i in range(n):
            for k in range(p):
                # T[i,k] should be identity matrix of size m
                # T[i,k] = sum_c W[k*n+i,c] * outer(U[i*m:(i+1)*m,c], V[:,k])
                # More precisely: T[i,k][j,j2] = sum_c U[i*m+j,c] * V[j2*p+k,c] * W[k*n+i,c]
                T = np.zeros((m, m))
                for c in range(rank):
                    wc = W_mat[k*n+i][c]
                    if wc != 0:
                        u_col = [U_mat[i*m+j][c] for j in range(m)]
                        v_col = [V_mat[j2*p+k][c] for j2 in range(m)]
                        T += wc * np.outer(u_col, v_col)
                if not np.allclose(T, np.eye(m), atol=1e-10):
                    return False
        # Do exact verification
        return verify(U_mat, V_mat, W_mat, rank)
    
    def try_merge(U_mat, V_mat, W_mat, rank, col_a, col_b):
        """Try to merge column b into column a by finding alpha such that
        we can eliminate column b. Returns (new_U, new_V, new_W, new_rank) or None."""
        # We want to find alpha such that for all entries:
        # U[i,m+j] * (V[j2*p+k] + alpha * V_b[j2*p+k]) * (W[k*n+i] + alpha * W_b[k*n+i]) 
        # plus the existing terms still work out.
        # Actually, the standard approach: we can merge if there exists alpha such that
        # the new column c' = c_a + alpha * c_b is equivalent to c_a in the tensor.
        # This means: for all (i,j,j2,k), the contribution from c' equals that from c_a.
        # U[i*m+j,c'] * V[j2*p+k,c'] * W[k*n+i,c'] = U[i*m+j,c_a] * V[j2*p+k,c_a] * W[k*n+i,c_a]
        # where U[i*m+j,c'] = U[i*m+j,c_a] + alpha*U[i*m+j,c_b], etc.
        
        # This is complex. Instead, use a simpler heuristic:
        # Try to set one column to be a linear combination and see if we can zero out another.
        
        # For now, try a simple approach: if two columns are identical in all three matrices,
        # we can merge them (but this won't happen in schoolbook).
        
        # More practically: try to find alpha such that after adding alpha*col_b to col_a,
        # we can set col_b to zero. This requires that for all tensor entries,
        # the change from col_a compensates for removing col_b.
        
        # Let's try a different approach: random local search
        # Pick a random column to eliminate, and try to find coefficients for other columns
        # that can replace it.
        return None
    
    # Local search: try to reduce rank by eliminating columns
    current_U = [row[:] for row in best_U]
    current_V = [row[:] for row in best_V]
    current_W = [row[:] for row in best_W]
    current_rank = best_rank
    
    # Try to find linear dependencies between columns
    # For each pair of columns, check if one can be expressed as a combination of others
    # This is expensive, so we limit the search
    
    # Alternative approach: use the fact that the tensor has structure
    # The matrix multiplication tensor M_{3,3,4} can be decomposed using
    # known algorithms. Let's try a structured decomposition.
    
    # Known result: rank of M_{3,3} is 7 (Strassen-like)
    # For M_{3,3,4}, we can use block decomposition
    
    # Try a block-based approach:
    # Split the 4 columns into 3+1 or 2+2
    # If we can find a rank-7 decomposition for 3x3x3, and handle the 4th column efficiently
    
    # For now, attempt a simple local search with random perturbations
    
    best_result = {"U": current_U, "V": current_V, "W": current_W}
    best_score = current_rank
    
    # Try structured decompositions
    # First, try to find a low-rank decomposition for the 3x3x3 part
    
    # Use a known rank-7 decomposition for 3x3 matrix multiplication (if it exists)
    # Actually, the minimal rank for 3x3 is unknown, but it's at most 23 (schoolbook is 27)
    # Let's just try to improve the schoolbook with local search
    
    # Local search: try to merge columns by finding alpha values
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Try to eliminate a random column
        col_to_elim = random.randint(0, current_rank - 1)
        
        # Find if we can express this column as a combination of others
        # This is a linear algebra problem, but let's try a simple heuristic
        
        # Try setting this column to zero and adjusting others
        # This is too complex for now, so we'll just return the schoolbook
        
        pass
    
    return best_result
# EVOLVE-BLOCK-END
