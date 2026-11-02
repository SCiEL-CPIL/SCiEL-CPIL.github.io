import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time as _time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matmul is T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, that's not right. The tensor is T[i,j,k] which should equal 1 when
    # we're computing C[i,k] += A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute T[i,j,k] from the decomposition
                val = 0.0
                for r in range(R):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 0.0

def _try_rank_search(n, m, p, target_rank, time_limit=100.0):
    """Search for a rank-`target_rank` decomposition using structured random search."""
    start_time = _time.time()
    deadline = start_time + time_limit
    
    # The tensor we want to decompose: T[i,j,k] = 1 for all i,j,k
    # We seek U (n*m x R), V (m*p x R), W (p*n x R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Use a parameterized family. 
    # For matrix multiplication, there's a known structure.
    # Let's try a more systematic approach: 
    # Initialize with schoolbook and try to merge columns.
    
    # Actually, let's use a different approach: 
    # The bilinear complexity of 2x3x3 is known to be 14 (Strassen-like).
    # Let me try to construct it using known algorithms or random search.
    
    # Random search with verification
    best_result = None
    
    for attempt in range(100000):
        if _time.time() > deadline - 10:
            break
        
        # Random initialization with structure
        R = target_rank
        # Use random vectors but try to satisfy constraints
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Normalize to help convergence
        for r in range(R):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-6 and norm_v > 1e-6 and norm_w > 1e-6:
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # Check if this satisfies the tensor equation
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_result = (U, V, W)
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = _time.time()
    deadline = start_time + 120.0
    
    # Schoolbook rank is 18
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank solutions
    for target_rank in [14, 15, 16, 17]:
        if _time.time() > deadline - 20:
            break
        
        result = _try_rank_search(n, m, p, target_rank, time_limit=deadline - _time.time() - 15)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
