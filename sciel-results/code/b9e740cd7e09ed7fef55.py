"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition():
    """Return the rank-120 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-6):
    """
    Verify that the tensor decomposition is exact.
    The tensor T[i,j,k] = 1 for all i,j,k.
    The decomposition gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check this for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr, tol=1e-4):
    """Round array entries to nearest half-integer, check if close."""
    rounded = np.round(arr * 2) / 2
    if np.allclose(arr, rounded, atol=tol):
        return rounded
    return None

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication 4x5 by 5x6.
    Returns dict with U, V, W as lists of lists (or arrays) with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try to get a good baseline with schoolbook
    U_sb, V_sb, W_sb = _schoolbook_decomposition()
    
    # Verify schoolbook
    assert _verify_decomposition(U_sb, V_sb, W_sb), "Schoolbook verification failed"
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = U_sb.shape[1]
    
    # Now try to find lower rank decompositions using structured search
    # Key insight: We can try to find rank r < n*m*p decompositions
    
    # Strategy: Use the fact that for matrix multiplication, there are known
    # lower bounds and constructions. For 4x5*5x6, the rank is at most 120.
    
    # Try structured random search for lower rank
    # We'll use a tensor decomposition approach
    
    def _try_rank_decomposition(rank, max_iters=2000, seed=None):
        """Try to find a rank-r decomposition using iterative methods."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent divergence
        norms_U = np.linalg.norm(U, axis=0, keepdims=True)
        norms_U[norms_U < 1e-10] = 1
        U = U / norms_U
        norms_V = np.linalg.norm(V, axis=0, keepdims=True)
        norms_V[norms_V < 1e-10] = 1
        V = V / norms_V
        norms_W = np.linalg.norm(W, axis=0, keepdims=True)
        norms_W[norms_W < 1e-10] = 1
        W = W / norms_W
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Update U: for each i,j, we want U[i*m+j, c] such that
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            
            # Actually, let's use a simpler approach: gradient descent on the loss
            # Loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            # Compute gradients
            # For U[i*m+j, c]: dL/dc = -2 * sum_k (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
            
            # Let's compute T first
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            
            err = np.sum((T - 1.0) ** 2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual = T[i, j, k] - 1.0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # grad_U[u_idx, c] += -2 * residual * V[v_idx, c] * W[w_idx, c]
                        grad_U[u_idx, :] += -2 * residual * V[v_idx, :] * W[w_idx, :]
                        grad_V[v_idx, :] += -2 * residual * U[u_idx, :] * W[w_idx, :]
                        grad_W[w_idx, :] += -2 * residual * U[u_idx, :] * V[v_idx, :]
            
            # Gradient descent with adaptive step size
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Re-normalize
            norms_U = np.linalg.norm(U, axis=0, keepdims=True)
            norms_U[norms_U < 1e-10] = 1
            U = U / norms_U
            norms_V = np.linalg.norm(V, axis=0, keepdims=True)
            norms_V[norms_V < 1e-10] = 1
            V = V / norms_V
            norms_W = np.linalg.norm(W, axis=0, keepdims=True)
            norms_W[norms_W < 1e-10] = 1
            W = W / norms_W
        
        # Check if we found a valid decomposition
        if best_err < 1e-6:
            # Try to round to half-integers
            U_rounded = _round_to_halfint(best_U)
            V_rounded = _round_to_halfint(best_V)
            W_rounded = _round_to_halfint(best_W)
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _verify_decomposition(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded, rank
            else:
                # Use the unrounded version if it's valid
                if _verify_decomposition(best_U, best_V, best_W):
                    return best_U, best_V, best_W, rank
        return None
    
    # Try various ranks below 120
    for rank in range(100, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_decomposition(rank, max_iters=1000, seed=rank)
        if result is not None:
            U_new, V_new, W_new, r = result
            if r < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = r
                print(f"Found rank {r} decomposition")
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
