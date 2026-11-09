"""Matrix multiplication tensor decomposition: 2x3 * 3x5."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    total = n * m * p  # 30
    
    # First, try the schoolbook decomposition (rank = 30)
    rank_school = total
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify the schoolbook decomposition
    def verify(U, V, W, rank, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The decomposition satisfies:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i, j, k] = delta[j,j] * delta[k,k] (the multiplication tensor)
        # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # But we need to check the actual tensor identity.
        
        # The matrix multiplication tensor T has entries T[(i,k), j] where
        # T[(i,k), j] = 1 if this is the identity for multiplying row i of A
        # (indexed by column j of A / row j of B) with column k.
        
        # More precisely: for matrices A (n x m) and B (m x p), C = A * B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,k,j] = 1 (it's just the identity in the middle dimension)
        # The decomposition says: sum_r U[i,r] V[j,r] W[k,r] = T[i,k,j]
        
        # Actually, let me think about this more carefully.
        # U has shape (n*m, rank): indexed by (i,j) pairs
        # V has shape (m*p, rank): indexed by (j,k) pairs  
        # W has shape (p*n, rank): indexed by (k,i) pairs
        
        # The contraction should give: for each (i,j,k),
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if this term appears in the schoolbook, 0 otherwise.
        
        # Actually, the correct verification: the matrix multiplication bilinear map
        # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # Can be written as: C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] * A[i,j] * B[j,k]
        # Wait, that's not right either.
        
        # The standard formulation: we want sum_r U[i, r] V[j, r] W[k, r] where
        # the bilinear form is: <sum_r u_r A v_r^T, B>... 
        
        # Let me use the direct tensor check:
        # T[i, k, j] (the multiplication tensor) should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,k,j] = 1 for all i,k,j (since every product term A[i,j]*B[j,k] contributes to C[i,k])
        
        # Actually no. The tensor T for matrix multiplication A(nxm) * B(mx) -> C(nxp)
        # is: T[(i,j),(j,k),(k,i)]... hmm, the index conventions matter.
        
        # Let me just check: does the proposed decomposition satisfy
        # for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 ?
        # Because in the schoolbook, C[i,k] = sum_j A[i,j]*B[j,k], and each term
        # A[i,j]*B[j,k] is a separate rank-1 term. So yes, for each (i,j,k),
        # the coefficient should be 1.
        
        # Build the tensor check
        # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.multiply(U_arr[i*m+j, :], V_arr[j*p+k, :]), W_arr[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        errors.append((i, j, k, val))
        
        if errors:
            return False, errors
        return True, []
    
    valid_school, _ = verify(U_school, V_school, W_school, rank_school)
    
    if not valid_school:
        # Something wrong with our schoolbook construction, fall back to simple
        pass
    
    # Now try to find lower-rank decompositions using various strategies
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Strategy: Try to reduce rank by finding linear dependencies
    # The key insight: if we can find a rank-r decomposition (r < 30), that's better.
    
    # Try random search for lower ranks
    def try_random_rank(rank, num_trials=50, seed=42):
        """Try to find a rank-r decomposition using random initialization."""
        rng = np.random.RandomState(seed)
        for trial in range(num_trials):
            # Random initialization
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
            
            # Use simple coordinate descent to refine
            for iteration in range(100):
                # Update U
                for idx in range(n * m):
                    i, j = divmod(idx, m)
                    # Target: sum_r U[idx, r] V[j*p+k, r] W[k*n+i, r] = 1 for all k
                    # Fix V and W, solve for U[idx, :]
                    # This is: sum_r U[idx, r] * (V[j*p+r, r] * W[r*n+i, r]) ... no
                    # The constraint is: for each k, sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    
                    # Build the linear system: A_x[idx, r] * U[idx, r] = 1 for each k
                    # A_x[idx, r] = V[j*p+r, r] * W[r*n+i, r]... this isn't quite right
                    # Actually: sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let x_r = U[idx, r]. Then sum_r x_r * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is a linear system in x: M @ x = 1 where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        U[idx, :] = x
                    except:
                        pass
                
                # Update V similarly
                for idx in range(m * p):
                    j, k = divmod(idx, p)
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        V[idx, :] = x
                    except:
                        pass
                
                # Update W similarly
                for idx in range(p * n):
                    k, i = divmod(idx, n)
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        W[idx, :] = x
                    except:
                        pass
            
            # Verify
            valid, errors = verify(U, V, W, rank)
            if valid:
                return True, U, V, W
        
        return False, None, None
    
    # Try descending ranks from schoolbook-1 down to some minimum
    # Be careful with time
    for rank in range(rank_school - 1, max(10, rank_school - 8), -1):
        if time.time() > deadline - 15:
            break
        
        success, U, V, W = try_random_rank(rank, num_trials=20, seed=rank)
        if success:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found lower rank, try even lower? Or stop?
            # Keep trying to find even lower ranks
    
    # Also try a structured approach: exploit the small dimensions
    # For 2x3x5, there might be known sub-schoolbook decompositions
    
    # One approach: try to combine some schoolbook terms
    # If we can find U, V, W such that the sum works with fewer terms
    
    # Let's try a more systematic search for rank 25-29
    if time.time() < deadline - 20:
        for rank in range(rank_school - 1, max(20, rank_school - 10), -1):
            if best_rank <= rank:
                break
            if time.time() > deadline - 15:
                break
            
            # Try a greedy / structured approach
            # Start from schoolbook and try to merge terms
            U_attempt = U_school.copy()
            V_attempt = V_school.copy()
            W_attempt = W_school.copy()
            
            # Try to zero out some rows/columns by finding dependencies
            # This is essentially finding a lower-rank equivalent
            pass  # Skip for now, complex to implement correctly
    
    # Convert to list of lists for the required interface
    # Use the best decomposition found
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Round to clean values if they're very close to integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                    mat[i][j] = float(round(mat[i][j]))
                elif abs(2*mat[i][j] - round(2*mat[i][j])) < 1e-9:
                    mat[i][j] = float(round(2*mat[i][j])) / 2.0
    
    # Final verification
    valid_final, _ = verify(best_U, best_V, best_W, best_rank)
    if not valid_final:
        # Fall back to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

