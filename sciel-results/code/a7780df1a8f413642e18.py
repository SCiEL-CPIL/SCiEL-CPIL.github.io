"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check dimensions
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The correct tensor for matrix multiplication should be T[i,j,k] = delta_{i,k} (for 4x4 * 4x8)
    # Wait, let me reconsider. For C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # But the W index is k*n+i per the convention.
    # So W[k*n+i, c] corresponds to the (k,i) entry.
    
    # Let's verify: for each (i,j,k), the sum should equal 1 if this is a valid decomposition
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # No wait - the tensor T represents the bilinear form. For matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 (it's the identity in the middle index sense)
    
    # Let's just check that the reconstruction works:
    # For random matrices A (n x m) and B (m x p), check A*B = sum_c U_c * V_c * W_c
    
    # Actually, let's just verify the tensor directly:
    # The multiplication tensor T[i,j,k] should equal 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k], and the tensor coefficient is 1
    
    # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i, j, k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < tol

def _reduce_rank(U, V, W, n, m, p, max_iter=1000, tol=1e-10):
    """Try to reduce the rank by finding linear dependencies."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Try to find linear dependencies among the rank-1 terms
    # We can try to combine columns that are linearly dependent
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try iterative column elimination
    for attempt in range(max_iter):
        # Find the column with smallest norm or most dependent column
        # Compute the rank of the combined tensor representation
        
        # For each column c, check if it can be expressed as a linear combination of others
        # This is expensive, so let's try a simpler approach:
        
        # Find the column that minimizes some dependency measure
        # Use the fact that if we have a linear dependency among U columns,
        # we might be able to eliminate one term
        
        # Try to find a column c such that U[:,c] is in the span of other U columns
        # AND V[:,c] is in the span of other V columns  
        # AND W[:,c] is in the span of other W columns
        
        eliminated = False
        for c in range(rank - 1):
            # Check if column c is linearly dependent on the others
            U_others = np.delete(U, c, axis=1)
            V_others = np.delete(V, c, axis=1)
            W_others = np.delete(W, c, axis=1)
            
            # Check if U[:,c] is in column space of U_others
            rank_full = np.linalg.matrix_rank(np.hstack([U_others, U[:,c:c+1]]), tol=1e-8)
            rank_others = np.linalg.matrix_rank(U_others, tol=1e-8)
            
            if rank_full > rank_others:
                continue
            
            # Similarly for V and W
            rank_full_v = np.linalg.matrix_rank(np.hstack([V_others, V[:,c:c+1]]), tol=1e-8)
            rank_others_v = np.linalg.matrix_rank(V_others, tol=1e-8)
            if rank_full_v > rank_others_v:
                continue
            
            rank_full_w = np.linalg.matrix_rank(np.hstack([W_others, W[:,c:c+1]]), tol=1e-8)
            rank_others_w = np.linalg.matrix_rank(W_others, tol=1e-8)
            if rank_full_w > rank_others_w:
                continue
            
            # All three are linearly dependent - try to eliminate
            # Find coefficients
            try:
                coeff_u, _, _, _ = np.linalg.lstsq(U_others, U[:,c], rcond=1e-8)
                coeff_v, _, _, _ = np.linalg.lstsq(V_others, V[:,c], rcond=1e-8)
                coeff_w, _, _, _ = np.linalg.lstsq(W_others, W[:,c], rcond=1e-8)
                
                # The coefficients should be consistent for a valid elimination
                # This is complex, so let's just try removing the column and verifying
                new_U = U_others
                new_V = V_others
                new_W = W_others
                
                if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-8):
                    U, V, W = new_U, new_V, new_W
                    rank -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = rank
                    eliminated = True
                    break
            except:
                continue
        
        if not eliminated:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    if time.time() < deadline:
        U, V, W, new_rank = _reduce_rank(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    # Round to clean values if possible
    for i in range(n * m):
        for j in range(best_rank):
            if abs(U_list[i][j] - round(U_list[i][j])) < 1e-8:
                U_list[i][j] = round(U_list[i][j])
    for i in range(m * p):
        for j in range(best_rank):
            if abs(V_list[i][j] - round(V_list[i][j])) < 1e-8:
                V_list[i][j] = round(V_list[i][j])
    for i in range(p * n):
        for j in range(best_rank):
            if abs(W_list[i][j] - round(W_list[i][j])) < 1e-8:
                W_list[i][j] = round(W_list[i][j])
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
