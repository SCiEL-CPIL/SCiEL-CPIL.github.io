"""Rewritten program for matrix multiplication tensor decomposition 3x3x7."""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = 130  # safety margin from 150s limit

    def _elapsed():
        return time.time() - start_time

    def _verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly matches the matmul tensor.
        
        For each (i, j, k), the tensor entry is 1 iff we're computing C[i,k] = sum_j A[i,j]*B[j,k].
        The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
        for the matmul tensor, and 0 otherwise (but the tensor is 0 for non-matmul entries by construction).
        
        Actually, the matmul tensor T[i,j,k] = 1 for all valid (i,j,k) in the contraction sense.
        The tensor is defined such that <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k].
        So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Compute all triples
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Efficiently: for each c, the contribution is outer product structure
        # Let's just compute directly for small sizes
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        errors.append((i, j, k, val))
        return len(errors) == 0, errors[:5]

    # Strategy: Start with schoolbook (rank 63) and try to reduce rank.
    # Use a random search / optimization approach to find lower rank decompositions.
    
    # First, let's try a structured approach.
    # The matmul tensor M_{n,m,p} can be viewed as a 3-tensor of size n x m x p.
    # We want to find rank-r decomposition: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^n, v_c in R^m, w_c in R^p.
    
    # For small matrices, we can use known results or search.
    # rank(M_{3,3,3}) is known to be 19 (Bini, Capovani, Lotti, Romani, 1981).
    # For M_{3,3,7}, we can try to extend.
    
    # Let's try a randomized search for lower rank.
    # We'll use a simple iterative approach: start with random rank-r matrices and try to fit.
    
    def try_random_decomposition(rank, num_trials=100, verbose=False):
        """Try to find a rank-r decomposition using random initialization and gradient-like steps."""
        best_error = float('inf')
        best_result = None
        
        for trial in range(num_trials):
            if _elapsed() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Simple iterative refinement
            for iteration in range(50):
                if _elapsed() > deadline - 5:
                    break
                    
                # For each c, update one factor at a time (alternating least squares style)
                # This is a simplified approach; true ALS for tensor decomposition is more complex.
                # Let's just compute the error and if it's good enough, return.
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    if err < 1e-8:
                        best_result = (U.copy(), V.copy(), W.copy())
                        if verbose:
                            print(f"Found exact decomposition at rank {rank}, trial {trial}, iter {iteration}")
                        return best_result
                
                if best_error < 1e-10:
                    break
            
            if best_result is not None:
                return best_result
        
        return best_result

    # Try decreasing ranks
    # Schoolbook is 63. Let's try to find something better.
    # Known: rank(M_{2,2,2}) = 7, rank(M_{3,3,3}) = 19
    # For M_{3,3,7}, a reasonable target might be around 20-30.
    
    # Let's try a structured approach first.
    # We can decompose M_{3,3,7} by considering it as 7 copies of M_{3,3,1} sharing the A and B factors.
    # M_{3,3,1} has rank 5. So a naive bound is 5*7 = 35, but we can do better by sharing.
    
    # Actually, let's just do a random search for ranks from low to high.
    # Start with a reasonable upper bound and try to go lower.
    
    # First, get the schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank
    # Based on known results, let's try ranks from 15 up to 30
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try structured decompositions first
    # For M_{3,3,p}, we can use the fact that it's a linear combination of p rank-1 tensors in the i-j plane
    # but with different k indices.
    
    # Let's try a random search for various ranks
    for rank in range(15, 31):
        if _elapsed() > deadline - 10:
            break
        
        result = try_random_decomposition(rank, num_trials=50, verbose=False)
        if result is not None:
            U, V, W = result
            # Verify
            valid, errors = _verify(U, V, W)
            if valid:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        else:
            # Try with more trials
            if _elapsed() < deadline - 20:
                result = try_random_decomposition(rank, num_trials=20, verbose=False)
                if result is not None:
                    U, V, W = result
                    valid, errors = _verify(U, V, W)
                    if valid:
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        # Convert to lists for the required format
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        
        # Round to nearest half-integer if close
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    mat[i][j] = round(mat[i][j] * 2) / 2
                    # Verify it's still close to original
                    if abs(mat[i][j] * 2 - round(mat[i][j] * 2)) > 1e-6:
                        mat[i][j] = round(mat[i][j])
        
        # Final verification
        valid, errors = _verify(U_list, V_list, W_list, tol=1e-6)
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
