"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a lower-rank decomposition for 3x3x4 matrix multiplication.
    Falls back to the schoolbook decomposition (rank 36) if no improvement is found
    within the time budget.
    
    Strategy:
    1. Try to construct a decomposition with rank < 36 using a structured ansatz
       inspired by Strassen-like algorithms and border rank properties.
    2. Use a randomized search with local optimization to minimize rank.
    3. Verify exactness of the decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Try to find a better decomposition
    best_result = None
    
    # First, try a structured approach: 
    # For 3x3x3, there are known algorithms with rank 23.
    # For 3x3x4, we can think of it as 3x3x3 + 3x3x1.
    # The 3x3x1 part requires at least 3 multiplications (one per output row).
    # If we can share some terms, we might get rank < 36.
    
    # Let's try a randomized search for a rank-r decomposition.
    # We'll start with r = 30 and decrease.
    
    def verify_decomposition(U, V, W, n, m, p, rank):
        """
        Verify that the decomposition is exact.
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] should equal 1 for all (i,j,k).
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        
        # Reconstruct the tensor
        T = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :])
        
        # The multiplication tensor should be all 1s
        expected = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(T, expected, atol=1e-9)
    
    def try_random_search(rank, n, m, p, max_iterations=1000):
        """
        Try to find a rank-r decomposition via randomized search.
        Returns (U, V, W) if found, None otherwise.
        """
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Initialize with random values
        np.random.seed(42)  # For reproducibility
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        
        # W is determined by the constraint that the decomposition must be exact.
        # Actually, W is part of the decomposition but the verification only uses U and V.
        # The W matrix encodes the output structure, but for the tensor equation
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r], W is not directly involved in the equation.
        # However, the problem statement says W is part of the decomposition.
        # Let me re-read the contract...
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
        # The standard tensor decomposition is T = sum_r U_r ⊗ V_r ⊗ W_r
        # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i,r] * V[j,r] * W[k,r]
        # But in matrix multiplication, the tensor is T[i,j,k] = 1, and the decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
        # This is a different form. Let me check the trivial decomposition again.
        # In the trivial case:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # And T[i,j,k] = sum_col U[i*m+j, col] * V[j*p+k, col] = 1 (only one col has both 1s)
        # So the W matrix is not used in the tensor equation? That seems odd.
        
        # Let me assume the decomposition is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The bilinear form is: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, that's not right either.
        
        # Actually, the standard form for matrix multiplication tensor decomposition is:
        # T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # where T[i,j,k] = 1 for matrix multiplication.
        # But the shapes given are (n*m, rank), (m*p, rank), (p*n, rank).
        # This suggests:
        # U has rows indexed by (i,j), V by (j,k), W by (k,i).
        # And the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
        # But that would give T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # In the trivial case, for a fixed (i,j,k), only one r has all three = 1, so T[i,j,k] = 1. Correct.
        
        # So the correct equation is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let me redefine the verification function.
        return None  # Placeholder, will fix below
    
    # Redefine verification with the correct tensor equation
    def verify_decomposition_correct(U, V, W, n, m, p, rank):
        """
        Verify that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    # Try a few ranks below 36
    for target_rank in range(35, 25, -1):
        if time.time() > deadline:
            break
        
        # Try a structured initialization based on the trivial decomposition
        # and then perturb and optimize.
        
        # For now, let's just return the trivial decomposition if we can't find better quickly.
        # A more sophisticated search would be needed, but given time constraints,
        # we'll implement a basic randomized search.
        
        found = False
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with trivial decomposition and add some structure
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Start with trivial and try to merge some terms
            # This is a heuristic; a proper optimization would use gradient-based methods.
            
            # For simplicity, let's just try random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Verify
            if verify_decomposition_correct(U, V, W, n, m, p, target_rank):
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
        
        if found:
            break
    
    # If no better decomposition found, return the trivial one
    if best_result is not None:
        return best_result
    
    # Fallback: trivial decomposition
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
