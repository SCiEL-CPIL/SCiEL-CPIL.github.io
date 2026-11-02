"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x4 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use a combination of:
    1. Start with schoolbook decomposition (rank 64)
    2. Use algebraic methods to find lower rank decompositions
    3. For 4x4 matrix multiplication, the known lower bound is 49 (Landsberg) and upper bound is 49 (Strassen-like)
       Actually, for 4x4 matrices, the best known rank is 49 (by Strassen's algorithm adapted).
       
    Let's implement a systematic search using random linear combinations and rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=4
    
    # Schoolbook rank is n*m*p = 64
    # We want to find the minimum rank
    
    # The tensor for matrix multiplication has indices (i,j,k) for input A[i,j], B[j,k] -> C[i,k]
    # In the convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For 4x4x4 matrix multiplication, we can use known algorithms.
    # Strassen's algorithm gives rank 7 for 2x2, and for 4x4 we can use recursive Strassen or other methods.
    
    # Let's try to construct a rank-49 decomposition using known techniques.
    # Actually, let's first verify what ranks are achievable.
    
    # For now, let's use a computational approach:
    # 1. Represent the multiplication tensor
    # 2. Try to find a low-rank decomposition
    
    # The multiplication tensor for n x m * m x p has dimension n*m*p = 64
    # Each rank-1 term corresponds to a triple of vectors (u, v, w)
    
    # Let's use the fact that for 4x4 matrices, we can use the following approach:
    # Split into 2x2 blocks and use Strassen recursively, or use direct algorithms.
    
    # Known result: The rank of 4x4 matrix multiplication is at most 49.
    # Let's try to find such a decomposition.
    
    # Approach: Use random linear algebra to find a decomposition
    # We'll use the following method:
    # 1. Start with a random basis
    # 2. Try to express the tensor as a sum of rank-1 tensors
    
    # For efficiency, let's use a different approach:
    # We know that the matrix multiplication tensor can be decomposed using
    # Strassen's algorithm. For 4x4, we can use:
    # - Split each 4x4 matrix into four 2x2 blocks
    # - Apply Strassen to each pair of blocks
    # - This gives a rank of 7 * (number of Strassen calls)
    
    # Actually, let's use a more systematic approach.
    
    # The standard Strassen algorithm for 2x2 uses 7 multiplications.
    # For 4x4, we can view it as 4 blocks of 2x2, and we need to compute:
    # C = A * B where A and B are 4x4
    # Split into 2x2 blocks: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Then C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21, C22 = A21*B12 + A22*B22
    
    # Using Strassen for each 2x2 multiplication:
    # We need 7 multiplications per 2x2 block multiplication
    # But we have 4 block multiplications, so 4*7 = 28? No, that's not right.
    
    # Actually, the standard way to apply Strassen to 4x4 is:
    # Use 7 Strassen multiplications on 2x2 blocks, but we need to combine them.
    
    # Let me think differently. The rank of 4x4 matrix multiplication is known to be 49.
    # Let's construct it using the following known decomposition.
    
    # For simplicity and correctness, let's use a verified rank-49 decomposition.
    
    # Actually, let's use a computational search with time limit.
    
    # Method: Use the "border rank" or "tensor rank" computation via numerical methods.
    
    # Given time constraints, let's return a known good decomposition.
    
    # For 4x4 matrix multiplication, one known rank is 49.
    # Let's construct it using the following approach:
    
    # We'll use the fact that the multiplication tensor can be written as:
    # T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # where u_r, v_r, w_r are vectors in R^{16}, R^{16}, R^{16} respectively.
    
    # For rank 49, we can use the following construction based on Strassen's algorithm
    # extended to 4x4 matrices.
    
    # Let's implement a specific rank-49 decomposition.
    
    # The following is a known rank-49 decomposition for 4x4 matrix multiplication.
    # It's based on the work of Strassen and subsequent improvements.
    
    # For now, let's use a simpler approach: return the schoolbook decomposition
    # and then try to improve it within the time limit.
    
    # First, create the schoolbook decomposition as a fallback
    rank_schoolbook = n * m * p  # 64
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1
    
    # Now try to find a lower rank decomposition
    # We'll use a random search approach with verification
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        rank = U.shape[1]
        
        # For each (i,j,k), check that T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # The tensor T for matrix multiplication is T[i,j,k] = 1 if this is a valid multiplication
        # Actually, the multiplication tensor is: T[i,j,k] = 1 for all (i,j,k) in the multiplication pattern
        
        # More precisely, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
        # where the multiplication A[i,j] * B[j,k] contributes to C[i,k]
        
        # In our convention, the tensor is:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # And we want T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Wait, that's not right. The multiplication tensor is not all ones.
        # The multiplication tensor T satisfies:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for the "multiplication" operation
        # But in the tensor decomposition framework, we have:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # And T[i,j,k] should equal 1 for all (i,j,k) that participate in the multiplication
        
        # Actually, for matrix multiplication, the tensor T is defined as:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Because each product A[i,j]*B[j,k] contributes to C[i,k]
        
        # Let me reconsider. The multiplication tensor for n×m * m×p → n×p is:
        # T ∈ R^{n×m × m×p × n×p}
        # T[(i,j),(j',k),(i',k')] = 1 if j=j' and i=i' and k=k', else 0
        # Wait, that's not quite right either.
        
        # Let me be more careful. The bilinear map is:
        # (A, B) → C where C[i,k] = sum_j A[i,j] * B[j,k]
        
        # The tensor T associated with this bilinear map is:
        # T[i,j,k] = 1 for all i,j,k (in the appropriate ranges)
        # because the coefficient of A[i,j] * B[j,k] in C[i,k] is 1.
        
        # So in the decomposition:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using random search
    # This is computationally expensive, so we'll limit the search
    
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook
    
    # Try ranks from 49 to 63
    for target_rank in range(49, 64):
        if time.time() > deadline - 30:  # Leave 30 seconds for verification
            break
        
        # Try random decompositions
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Generate random vectors
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Normalize to make verification more stable
            for c in range(target_rank):
                U_rand[:, c] /= np.linalg.norm(U_rand[:, c]) + 1e-10
                V_rand[:, c] /= np.linalg.norm(V_rand[:, c]) + 1e-10
                W_rand[:, c] /= np.linalg.norm(W_rand[:, c]) + 1e-10
            
            # Convert to lists for verification
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_U = U_list
                best_V = V_list
                best_W = W_list
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Return the best decomposition found
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
# EVOLVE-BLOCK-END
