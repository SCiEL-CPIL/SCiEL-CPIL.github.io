"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Uses the Kronecker product structure to decompose into sub-problems.
    
    The matrix multiplication tensor for C = A * B where A is (n x m) and B is (m x p)
    has entries T[i, j, k] = A[i, j] * B[j, k].
    
    We use the fact that we can decompose the middle dimension (m=5) into sums.
    Specifically, 5 = 2 + 3, so we can split the 5-dimensional index into a 2-part and a 3-part.
    
    For a 2x2 matrix multiplication, the minimal rank is 7 (Strassen).
    For a 2x3 matrix multiplication, we need to find the minimal rank.
    For a 2x3x6 problem, we can think of it as 2 * (3x6) but that's not quite right.
    
    Actually, let's think more carefully. The tensor is T[i, j, k] where i in [0, n), j in [0, m), k in [0, p).
    We want to write T = sum_r U[:, r] V[:, r] W[:, r]^T (in appropriate tensor form).
    
    The standard schoolbook decomposition has rank n*m*p.
    
    Key insight: The matrix multiplication tensor for (n, m, p) can be decomposed using
    the structure of the indices. If we can find a decomposition for smaller dimensions,
    we can combine them.
    
    For (2, 5, 6):
    - We can split m = 5 into m1 + m2 where m1 = 2, m2 = 3.
    - Then the tensor becomes a direct sum of (2, 2, 6) and (2, 3, 6) tensors.
    - But (2, 2, 6) and (2, 3, 6) still need decomposition.
    
    Actually, let's use a different approach. The minimal rank of matrix multiplication
    tensor M_{n,m,p} is known for some cases:
    - M_{2,2,2} has rank 7 (Strassen)
    - M_{2,2,p} can be decomposed using Strassen-like methods
    
    Let me think about this differently. We can use the fact that:
    M_{n, m, p} can be built from smaller multiplications.
    
    For (2, 5, 6), let's try to use a Kronecker product approach:
    5 = 2 + 3, 6 = 2 * 3
    
    Actually, the cleanest approach: use the known minimal ranks for small rectangular multiplications
    and combine them.
    
    Let me try a direct construction. The tensor T has shape (n*m, m*p, p*n) in the flattened form
    where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank)  
    - W has shape (p*n, rank)
    
    And the constraint is that for all i, j, k:
    T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right. Let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by [k*n + i, r] where k is the column of B and i is the row of A.
    
    The tensor entry for (i, j, k) where i is row of A, j is column of A / row of B, k is column of B
    should be: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this corresponds to C[i,k] = sum_j A[i,j]B[j,k]
    
    Actually, the matrix multiplication tensor is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has entries T[i, j, k] = 1 for the "coefficient" of A[i,j] * B[j,k] in C[i,k].
    
    In the flattened form:
    - U row index: i*m + j (for A[i,j])
    - V row index: j*p + k (for B[j,k])
    - W row index: k*n + i (for C[i,k], following the trace convention)
    
    And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i, j, k.
    
    Now, for the minimal rank:
    - The trivial rank is n*m*p = 2*5*6 = 60.
    - We can do better using Strassen-like methods.
    
    Let me try to construct a decomposition using the following idea:
    Split the middle index j into two parts: j = j1 + j2 where j1 in [0, 2) and j2 in [0, 3).
    Then sum_j A[i,j] B[j,k] = sum_{j1, j2} A[i, j1 + 3*j2] B[j1 + 3*j2, k]
    
    Hmm, this doesn't quite work because the indices don't separate cleanly.
    
    Let me try a different approach. I'll use a randomized or algebraic method to find a low-rank decomposition.
    
    Actually, let me just try to find the minimal rank using known results or a simple search.
    
    For small matrices, the minimal rank of matrix multiplication is:
    - (2,2,2): 7
    - (2,2,3): ? 
    - (2,3,3): ?
    
    Let me try to use a constructive approach. I'll start with the schoolbook decomposition and try to reduce the rank.
    
    Given the time constraint, let me try a practical approach: use the known Strassen decomposition for the 2x2 case
    and extend it.
    
    Actually, let me just implement a simple algebraic approach. I'll set up the equations and try to solve them.
    
    For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    For all i, j, k: sum_{r'=0}^{r-1} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    This is a system of nonlinear equations. Let me try to solve it for small r.
    
    Given the complexity, let me try a heuristic approach: start with a reasonable rank and use optimization.
    """
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank using a simple approach
    # We'll try ranks from a lower bound up to the schoolbook rank
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But we know Strassen gives 7 for 2x2x2, so we can do better than 30
    
    # Let's try to find a decomposition with rank around 30-40
    # Use a simple iterative approach
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    def check_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i * m + j] * V_mat[j * p + k] * W_mat[k * n + i])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Try different ranks
    for test_rank in range(30, 61):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            U_test = np.zeros((n * m, test_rank), dtype=float)
            V_test = np.zeros((m * p, test_rank), dtype=float)
            W_test = np.zeros((p * n, test_rank), dtype=float)
            
            # Initialize with random values
            U_test = np.random.randn(n * m, test_rank)
            V_test = np.random.randn(m * p, test_rank)
            W_test = np.random.randn(p * n, test_rank)
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for iteration in range(2000):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U_test)
                grad_V = np.zeros_like(V_test)
                grad_W = np.zeros_like(W_test)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] * (current_val - 1)
                            current_val = np.sum(U_test[i * m + j] * V_test[j * p + k] * W_test[k * n + i])
                            error = current_val - 1.0
                            
                            grad_U[i * m + j] += error * V_test[j * p + k] * W_test[k * n + i]
                            grad_V[j * p + k] += error * U_test[i * m + j] * W_test[k * n + i]
                            grad_W[k * n + i] += error * U_test[i * m + j] * V_test[j * p + k]
                
                U_test -= lr * grad_U
                V_test -= lr * grad_V
                W_test -= lr * grad_W
                
                if iteration % 100 == 0:
                    max_error = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_test[i * m + j] * V_test[j * p + k] * W_test[k * n + i])
                                max_error = max(max_error, abs(val - 1.0))
                    if max_error < 1e-6:
                        break
            
            if check_decomposition(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = test_rank
                break
        
        if best_rank == test_rank:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
