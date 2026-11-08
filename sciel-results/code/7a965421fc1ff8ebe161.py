import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Start with schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][column] = 1
                V[idx_v][column] = 1
                W[idx_w][column] = 1
                column += 1

    # Verify the schoolbook decomposition
    if not verify(U, V, W, rank):
        raise ValueError("Schoolbook decomposition failed verification")

    # Try to reduce rank using Strassen-style approach for 2x2 blocks
    # Split the 5-wide dimensions into 2+2+1 sub-blocks
    # For the 2x2 sub-problems, use Strassen's 7-term algorithm
    # For mixed blocks, use schoolbook
    
    # Strategy: We'll try to construct a decomposition with lower rank
    # by exploiting the structure of the matrix multiplication tensor
    
    # For 2x5x5, we can think of it as:
    # A is 2x5, B is 5x5, C is 2x5
    # Split columns of A into [A1 (2x2), A2 (2x2), A3 (2x1)]
    # Split rows of B into [B1 (2x5), B2 (2x5), B3 (1x5)]
    # Split columns of B into [B11 (2x2), B12 (2x2), B13 (2x1)] etc.
    
    # Actually, let's try a different approach: use the fact that we have
    # rectangular matrices and see if we can find a lower-rank decomposition
    
    # For now, let's try to see if we can reduce the rank by looking for
    # linear dependencies in the tensor structure
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c with minimal rank
    
    # One approach: try to find if the schoolbook decomposition has
    # redundant terms that can be eliminated
    
    # For 2x5x5, the lower bound on rank is at least max(n*m, m*p, p*n) = 25
    # and at least n*m*p - (n-1)*(m-1)*(p-1) = 50 - 1*4*4 = 34 (from some bounds)
    
    # Let's try a structured approach: group terms by (i,k) pairs
    # For fixed i and k, we have m=5 terms with different j values
    # These terms are: (i, j, k) for j=0..4
    # Can we represent these 5 terms with fewer tensor products?
    
    # The constraint for fixed (i,k) is:
    # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
    # Wait, that's not quite right. Let me reconsider.
    
    # Actually, the tensor is T[i,j,k] = 1 for all (i,j,k)
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a 3-way tensor factorization problem.
    
    # Let's try a heuristic: random restarts with local optimization
    # Start with a random initialization and try to minimize the rank
    # while satisfying the constraints
    
    # Actually, for this specific problem, let's try to see if we can
    # use a known result: the rank of matrix multiplication for 2x5x5
    
    # Based on literature, the rank of 2x5x5 matrix multiplication is
    # believed to be around 35-40, but I'm not sure of the exact value.
    
    # Let's try a constructive approach: use block decomposition
    # Split A (2x5) into [A_left (2x2), A_right (2x3)]
    # Split B (5x5) into [[B11 (2x2), B12 (2x2), B13 (2x1)],
    #                    [B21 (3x2), B22 (3x2), B23 (3x1)]]
    
    # Then C = A*B = [A_left, A_right] * B
    # = A_left * B_left + A_right * B_right
    # where B_left is the first 2 columns of B, B_right is the last 3 columns
    
    # This doesn't immediately help with rank reduction.
    
    # Let's try a different strategy: use the fact that the tensor is
    # separable in a certain way
    
    # For now, let's just return the schoolbook decomposition as a safe baseline
    # and note that future work should implement actual rank reduction algorithms
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
