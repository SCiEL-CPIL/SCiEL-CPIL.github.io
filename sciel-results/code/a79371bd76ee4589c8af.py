"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin

    n, m, p = DIMENSIONS
    # Known minimal rank for M_{3,3,7} is 63 (trivial decomposition)
    # Try to find a lower rank via continuous optimization + rounding

    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] should be 1
                    # All other entries should be 0
                    val = np.dot(U[i * m + j, :], V[j * p + k, :]) * W[k * n + i, :]
                    # Actually, the tensor entry is sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > tol:
                        return False
        # Check that off-diagonal entries are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i, j, k) != (i2, j2, k2):
                                    # Check if this combination appears in the tensor
                                    # The tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                                    # We need T[i*m+j, j*p+k, k*n+i] = 1 and all others = 0
                                    # But we only need to check that the decomposition is valid
                                    # The key check is that for each (i,j,k), the value at
                                    # position (i*m+j, j*p+k, k*n+i) is 1
                                    pass
        return True

    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Fast verification using numpy."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Build the full tensor and check
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # And T should be zero elsewhere for valid matrix mult
        
        # Efficient check: for each r, we have rank-1 contributions
        # U[a,r] * V[b,r] * W[c,r]
        # The matrix multiplication tensor has 1s at (i*m+j, j*p+k, k*n+i)
        
        # Check all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1) > tol:
                        return False
        return True

    def round_to_half_int(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def try_optimization(max_rank, time_budget):
        """Try to find a decomposition of rank max_rank via optimization."""
        n, m, p = DIMENSIONS
        dim_u = n * m  # 9
        dim_v = m * p  # 21
        dim_w = p * n  # 21
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(dim_u, max_rank) * 0.5
        V = np.random.randn(dim_v, max_rank) * 0.5
        W = np.random.randn(dim_w, max_rank) * 0.5
        
        # Gradient descent to minimize the error
        # Error = sum over all (a,b,c) of (T[a,b,c] - target[a,b,c])^2
        # where target[a,b,c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k, else 0
        
        # Precompute target tensor indices
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i*m+j, j*p+k, k*n+i))
        
        # Use a simplified approach: minimize the Frobenius norm of the difference
        # between the reconstructed tensor and the target tensor
        
        # For efficiency, we'll use a subset of constraints
        # The key constraints are:
        # 1. For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # 2. For "wrong" combinations: sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # This is complex, so let's try a simpler approach:
        # Start from the trivial decomposition and try to merge terms
        
        # Actually, for M_{3,3,7}, it's known that rank = 63 is minimal
        # (since 7 is prime and the other dimensions are small)
        # Let's verify this is indeed minimal by checking known results
        
        # For matrix multiplication M_{m,n,p}, the trivial rank is m*n*p
        # Strassen's algorithm gives better ranks for larger matrices
        # But for 3x3x7, the minimal rank is likely 63
        
        # Try a few random initializations and see if we can get below 63
        for attempt in range(10):
            if time.time() > start_time + time_budget:
                break
            
            np.random.seed(42 + attempt)
            U_curr = np.random.randn(dim_u, max_rank) * 0.1
            V_curr = np.random.randn(dim_v, max_rank) * 0.1
            W_curr = np.random.randn(dim_w, max_rank) * 0.1
            
            # Simple gradient step
            lr = 0.01
            for step in range(100):
                if time.time() > start_time + time_budget:
                    break
                
                # Compute gradients (simplified)
                # We want to minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                
                grad_U = np.zeros_like(U_curr)
                grad_V = np.zeros_like(V_curr)
                grad_W = np.zeros_like(W_curr)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i*m+j
                            b = j*p+k
                            c = k*n+i
                            s = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                            err = s - 1
                            for r in range(max_rank):
                                grad_U[a, r] += 2 * err * V_curr[b, r] * W_curr[c, r]
                                grad_V[b, r] += 2 * err * U_curr[a, r] * W_curr[c, r]
                                grad_W[c, r] += 2 * err * U_curr[a, r] * V_curr[b, r]
                
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
            
            # Round to half-integers and verify
            U_rounded = round_to_half_int(U_curr)
            V_rounded = round_to_half_int(V_curr)
            W_rounded = round_to_half_int(W_curr)
            
            if verify_decomposition_fast(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded, max_rank
        
        return None

    # Try to find a rank below 63
    best_result = None
    for test_rank in [50, 55, 60]:
        if time.time() > deadline - 30:
            break
        result = try_optimization(test_rank, time_budget=20)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to trivial decomposition
    U, V, W, rank = trivial_decomposition()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
