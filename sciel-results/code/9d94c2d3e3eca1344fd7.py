"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    - Start with schoolbook decomposition (rank 63).
    - Use local search to try to reduce rank by merging terms.
    - Verify exactness at each step.
    """
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin
    
    n, m, p = 3, 3, 7
    
    def verify(U, V, W, n, m, p, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Check all combinations to be absolutely sure
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
                                if abs(actual - expected) > tol:
                                    return False
        return True
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2
    
    def try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=1000):
        """
        Try to reduce rank from current to target_rank using local search.
        Returns (success, U, V, W) if successful.
        """
        current_rank = U.shape[1]
        if target_rank >= current_rank:
            return False, U, V, W
        
        # Initialize with first target_rank columns
        U_curr = U[:, :target_rank].copy()
        V_curr = V[:, :target_rank].copy()
        W_curr = W[:, :target_rank].copy()
        
        # Add random perturbation
        np.random.seed(42)
        U_curr = U_curr + np.random.randn(n*m, target_rank) * 0.1
        V_curr = V_curr + np.random.randn(m*p, target_rank) * 0.1
        W_curr = W_curr + np.random.randn(p*n, target_rank) * 0.1
        
        best_error = np.inf
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        for iteration in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for i2 in range(n):
                                for k2 in range(p):
                                    expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                    actual = 0.0
                                    for r in range(target_rank):
                                        actual += U_curr[i*m+j, r] * V_curr[j2*p+k, r] * W_curr[k2*n+i2, r]
                                    error += (actual - expected) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            
            if best_error < 1e-10:
                # Check if we can round to half-integers
                U_rounded = round_to_half(best_U)
                V_rounded = round_to_half(best_V)
                W_rounded = round_to_half(best_W)
                
                if verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-8):
                    return True, U_rounded, V_rounded, W_rounded
                else:
                    # Try without rounding
                    if verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
                        return True, best_U, best_V, best_W
                    else:
                        continue
            
            # Gradient-based update (simplified)
            # For each term, adjust to reduce error
            lr = 0.01
            for r in range(target_rank):
                # Compute gradient for U[:, r]
                grad_U = np.zeros(n*m)
                grad_V = np.zeros(m*p)
                grad_W = np.zeros(p*n)
                
                for i in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k in range(p):
                                for i2 in range(n):
                                    for k2 in range(p):
                                        expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                        actual = 0.0
                                        for rr in range(target_rank):
                                            actual += U_curr[i*m+j, rr] * V_curr[j2*p+k, rr] * W_curr[k2*n+i2, rr]
                                        diff = actual - expected
                                        if rr == r:
                                            grad_U[i*m+j] += 2 * diff * V_curr[j2*p+k, r] * W_curr[k2*n+i2, r]
                                            grad_V[j2*p+k] += 2 * diff * U_curr[i*m+j, r] * W_curr[k2*n+i2, r]
                                            grad_W[k2*n+i2] += 2 * diff * U_curr[i*m+j, r] * V_curr[j2*p+k, r]
                
                U_curr[:, r] -= lr * grad_U
                V_curr[:, r] -= lr * grad_V
                W_curr[:, r] -= lr * grad_W
        
        # Final check
        if best_error < 1e-8:
            U_rounded = round_to_half(best_U)
            V_rounded = round_to_half(best_V)
            W_rounded = round_to_half(best_W)
            
            if verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                return True, U_rounded, V_rounded, W_rounded
            elif verify(best_U, best_V, best_W, n, m, p, tol=1e-6):
                return True, best_U, best_V, best_W
        
        return False, best_U, best_V, best_W
    
    # Start with schoolbook decomposition
    rank = n * m * p  # 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify initial decomposition
    if not verify(U, V, W, n, m, p):
        raise ValueError("Initial schoolbook decomposition failed verification")
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    # Try to reduce rank
    for target_rank in range(rank - 1, 10, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        success, U_new, V_new, W_new = try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=500)
        
        if success:
            best_rank = target_rank
            best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            # Continue trying to reduce further
            U, V, W = U_new, V_new, W_new
        else:
            # If we can't reduce from this rank, try a few more
            if target_rank < rank - 5:
                break
    
    return best_result
# EVOLVE-BLOCK-END
