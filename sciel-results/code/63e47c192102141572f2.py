import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Target tensor values are 1 for all (i, j, k)
    # We want U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c to be 1.
    
    def verify(U, V, W, tol=1e-8):
        # Check if the decomposition is valid
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank(r, time_budget):
        """Try to find a decomposition of rank r."""
        # Initialize with small random values
        np.random.seed(42) # Fixed seed for reproducibility within the function
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Precompute indices for speed
        # For U update: for each (i,j), we need V[j*p+k,:] and W[k*n+i,:] for all k
        # For V update: for each (j,k), we need U[i*m+j,:] and W[k*n+i,:] for all i
        # For W update: for each (k,i), we need U[i*m+j,:] and V[j*p+k,:] for all j
        
        for it in range(200):
            if time.time() > time_budget:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    # Construct the linear system for U[i*m+j, :]
                    # We want U_row @ (V_col * W_col) = 1 for all k
                    # Let M be a matrix of size (p, r) where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Then sum_c U_row[c] * M[k, c] = 1 for all k
                    # This is M.T @ U_row = 1
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M.T @ U_row = ones(p)
                    # Using least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    # Construct linear system for V[j*p+k, :]
                    # We want V_row @ (U_col * W_col) = 1 for all i
                    # Let M be a matrix of size (n, r) where M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    # Then sum_c V_row[c] * M[i, c] = 1 for all i
                    # This is M.T @ V_row = 1
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    # Construct linear system for W[k*n+i, :]
                    # We want W_row @ (U_col * V_col) = 1 for all j
                    # Let M be a matrix of size (m, r) where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then sum_c W_row[c] * M[j, c] = 1 for all j
                    # This is M.T @ W_row = 1
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
                        
            # Check error occasionally
            if it % 10 == 0:
                # Calculate max error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-10:
                    return U, V, W, max_err
                    
        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return U, V, W, max_err

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from low to high
    # Known lower bounds suggest rank is likely < 50 for 4x5x5
    # Let's try a range
    for r in range(20, 101):
        if time.time() > deadline - 10:
            break
            
        U, V, W, err = try_rank(r, time.time() + 5) # Give 5s per rank attempt
        
        if err < 1e-8:
            # Found a valid decomposition
            best_rank = r
            best_U, best_V, best_W = U, V, W
            # We can stop early if we find a very low rank, but let's see if we can go lower
            # Actually, since we iterate from low to high, the first one found is the minimum in this sequence
            # But ALS might fail for the true minimum rank due to local minima.
            # So we might want to continue a bit to be sure, or just take the first success.
            # Given the time constraint, taking the first success is a good heuristic.
            break
            
    # If no low rank found, fall back to schoolbook
    if best_U is None:
        best_rank = n * m * p
        best_U = np.zeros((n * m, best_rank))
        best_V = np.zeros((m * p, best_rank))
        best_W = np.zeros((p * n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1
                    best_V[j * p + k, col] = 1
                    best_W[k * n + i, col] = 1
                    col += 1
    else:
        # Snap to half-integers if close
        # The problem asks for integer/half-integer entries.
        # If the error is very small, the values should be close to half-integers.
        # Let's check if snapping improves or maintains validity.
        U_snapped = np.round(best_U * 2) / 2
        V_snapped = np.round(best_V * 2) / 2
        W_snapped = np.round(best_W * 2) / 2
        
        # Verify snapped version
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_snapped[i*m+j, :] * V_snapped[j*p+k, :] * W_snapped[k*n+i, :])
                    if abs(val - 1.0) > 1e-5:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            best_U, best_V, best_W = U_snapped, V_snapped, W_snapped

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
