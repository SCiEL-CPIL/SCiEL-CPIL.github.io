"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin for evaluation

    def _verify_decomposition(U, V, W, n, m, p, num_trials=50):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using the decomposition
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        # Left factor: sum_j U[i*m+j, r] * A[i,j]
                        left = 0.0
                        for j in range(m):
                            left += U[i*m+j, r] * A[i,j]
                        # Right factor: sum_j V[j*p+k, r] * B[j,k]
                        right = 0.0
                        for j in range(m):
                            right += V[j*p+k, r] * B[j,k]
                        val += left * right * W[k*n+i, r]
                    C_computed[i,k] = val
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def _round_to_half_int(arr):
        """Round array entries to the nearest half-integer."""
        return np.round(arr * 2) / 2

    def _build_schoolbook(n, m, p):
        """Build the schoolbook decomposition with rank n*m*p."""
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return U.tolist(), V.tolist(), W.tolist()

    # First, try to find a better decomposition using structured search
    # For 3x3 * 3x8, we can try various approaches
    
    # Approach 1: Use the fact that we can share computation across columns
    # For each column k of B, we compute A * B[:,k]
    # If we can find a low-rank decomposition for 3x3 * 3x1, we can reuse it
    
    # For 3x3 * 3x1 (vector-matrix product), the tensor is:
    # T[i,j] = 1 for all i,j (a 2nd-order tensor, i.e., a matrix of ones)
    # This has rank 1! So U[i*m+j, r] = 1, V[j, r] = 1, W[i, r] = 1
    
    # But wait, for matrix multiplication, the structure is different.
    # Let me think about this more carefully.
    
    # For 3x3 * 3x8, the tensor T[i,j,k] where:
    # i in {0,1,2} (row of A and C)
    # j in {0,1,2} (column of A and row of B)
    # k in {0,...,7} (column of B and C)
    
    # The tensor entry T[i,j,k] = 1 if and only if the (i,j) entry of A
    # and the (j,k) entry of B contribute to the (i,k) entry of C.
    # This is always true, so T[i,j,k] = 1 for all valid i,j,k.
    
    # This is the all-ones tensor of shape (3, 3, 8).
    # The rank of the all-ones tensor is 1!
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    
    # Let's verify this.
    if time.time() < deadline - 5:
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        if _verify_decomposition(U, V, W, n, m, p, num_trials=20):
            return {"U": U, "V": V, "W": W}

    # If rank 1 doesn't work (which it shouldn't for matrix multiplication),
    # let's try a structured approach.
    
    # For 3x3 matrix multiplication, the best known rank is 19 (Landsberg).
    # For 3x3 * 3x8, we can potentially do better than 72.
    
    # Let's try a block approach:
    # Split B into two 3x4 blocks: B = [B1 | B2]
    # Then C = A * B = [A*B1 | A*B2]
    # If we can find a rank-r decomposition for 3x3 * 3x4,
    # we can use it twice for the two blocks.
    
    # But this doesn't save us anything directly.
    
    # Let's try a different approach: use a low-rank factorization of the
    # matrix multiplication tensor.
    
    # For 3x3 * 3x3, one known decomposition uses rank 19.
    # We can extend this to 3x3 * 3x8 by replicating for each of the 8 columns,
    # but that gives rank 19*8 = 152, which is worse than schoolbook.
    
    # Instead, let's try to find a decomposition that shares computation
    # across the 8 columns.
    
    # Idea: For each column k of B, we compute A * B[:,k].
    # If we precompute some linear combinations of the rows of A,
    # we can reuse them for all columns.
    
    # Specifically, for each row i of A, define:
    #   S_i = sum_j A[i,j] * e_j  (a 3-vector)
    # Then C[i,k] = sum_j A[i,j] * B[j,k] = S_i · B[:,k]
    
    # This is just the dot product, which doesn't help directly.
    
    # Let's try a more sophisticated approach: use a tensor decomposition
    # that exploits the structure of the all-ones tensor.
    
    # The all-ones tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{r=1}^R u_r[i] * v_r[j] * w_r[k]
    
    # For R=1: u[i]=1, v[j]=1, w[k]=1
    # This gives rank 1, but as we saw, it doesn't work for matrix multiplication.
    
    # The issue is that the U, V, W matrices in the decomposition
    # must satisfy specific constraints that come from the matrix multiplication
    # structure.
    
    # Let me re-examine the problem statement.
    # The interface is: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And the decomposition should satisfy:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # This is indeed the all-ones tensor, and it has rank 1.
    # But the verification function checks against actual matrix multiplication,
    # which is a different operation.
    
    # I think the key insight is that the "rank" in the tensor decomposition
    # sense is not the same as the "arithmetic complexity" of matrix multiplication.
    # The tensor rank gives the minimum number of scalar multiplications needed
    # if we allow arbitrary linear combinations of the inputs.
    
    # For the all-ones tensor, rank 1 means we can compute:
    # C[i,k] = (sum_j A[i,j]) * (sum_j B[j,k]) * 1
    # But this is NOT the same as sum_j A[i,j] * B[j,k]!
    
    # So the rank-1 decomposition of the all-ones tensor does NOT correspond
    # to a valid matrix multiplication algorithm.
    
    # The correct formulation is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j' V[j'*p+k,r] * B[j',k]) * W[k*n+i,r]
    
    # Wait, the indices don't match up. Let me look at this more carefully.
    
    # In a bilinear algorithm for matrix multiplication:
    # C = sum_r alpha_r * (U_r * A) * (V_r * B) * W_r
    # where U_r, V_r are matrices and W_r is a matrix.
    
    # In the tensor decomposition formulation:
    # C[i,k] = sum_r (sum_j u_r[i,j] * A[i,j]) * (sum_k' v_r[j,k'] * B[j,k']) * w_r[i,k]
    
    # But this doesn't make sense because the second sum is over k', not k.
    
    # I think the correct formulation is:
    # For each r, we compute:
    #   L_r = sum_j U[i*m+j, r] * A[i,j]  (for each i)
    #   R_r = sum_k V[j*p+k, r] * B[j,k]  (for each j)
    # Then C[i,k] = sum_r L_r[i] * R_r[k] * W[k*n+i, r]
    
    # No, that's still not right.
    
    # Let me just implement a proper search and see what works.
    
    # For now, let's return the schoolbook decomposition, but try to find
    # a better one using a more efficient method.
    
    # Use a gradient-based approach to minimize the error.
    
    def _search_decomposition(n, m, p, max_rank, max_iters=1000, seed=42):
        """Search for a low-rank decomposition using gradient descent."""
        rng = np.random.RandomState(seed)
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for rank in range(2, min(max_rank, 72) + 1):
            if time.time() > deadline - 10:
                break
            
            # Initialize randomly
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Try a few random initializations
            for init in range(2):
                if time.time() > deadline - 10:
                    break
                
                U = rng.randn(n * m, rank) * 0.5
                V = rng.randn(m * p, rank) * 0.5
                W = rng.randn(p * n, rank) * 0.5
                
                # Gradient descent
                lr = 0.01
                for it in range(max_iters):
                    if time.time() > deadline - 15:
                        break
                    
                    # Compute error and gradients
                    A = rng.randn(n, m)
                    B = rng.randn(m, p)
                    C_expected = A @ B
                    C_computed = np.zeros((n, p))
                    
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    err = 0.0
                    for i in range(n):
                        for k in range(p):
                            val = 0.0
                            left_vals = np.zeros(rank)
                            right_vals = np.zeros(rank)
                            for r in range(rank):
                                left = 0.0
                                for j in range(m):
                                    left += U[i*m+j, r] * A[i,j]
                                right = 0.0
                                for j in range(m):
                                    right += V[j*p+k, r] * B[j,k]
                                left_vals[r] = left
                                right_vals[r] = right
                                val += left * right * W[k*n+i, r]
                            C_computed[i,k] = val
                            err += (val - C_expected[i,k])**2
                            
                            # Gradients
                            for r in range(rank):
                                dU[i*m + range(m), r] += (val - C_expected[i,k]) * right_vals[r] * W[k*n+i, r] * A[i, range(m)]
                                dV[range(m)*p + k, r] += (val - C_expected[i,k]) * left_vals[r] * W[k*n+i, r] * B[range(m), k]
                                dW[k*n+i, r] += (val - C_expected[i,k]) * left_vals[r] * right_vals[r]
                    
                    # Update
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    if err < 1e-12:
                        if err < best_error:
                            best_error = err
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        return best_U, best_V, best_W, best_error
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-10:
                    break
            
            if best_error < 1e-10:
                break
        
        return best_U, best_V, best_W, best_error
    
    # Try the search
    if time.time() < deadline - 20:
        U, V, W, err = _search_decomposition(n, m, p, max_rank=50)
        if U is not None and err < 1e-10:
            U_rounded = _round_to_half_int(U)
            V_rounded = _round_to_half_int(V)
            W_rounded = _round_to_half_int(W)
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, num_trials=20):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank):
    """Try to determine if a rank-r decomposition exists."""
    return False

def _solve_decomposition(n, m, p, rank):
    """Solve for the decomposition matrices."""
    return None, None, None
# EVOLVE-BLOCK-END
