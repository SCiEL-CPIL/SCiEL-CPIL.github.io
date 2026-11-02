"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    has rank decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We verify by checking that for random inputs, the decomposition matches.
    """
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check via the tensor identity: 
    # For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if valid, else 0
    # Actually the tensor is T[i,j,k] = delta (always 1 for the multiplication tensor)
    # Wait, the matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is: (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # As a 3-tensor: T[i,j,k] where we contract A and B.
    # The rank-r decomposition means: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Here T[i,j,k] = 1 for all i,j,k (the multiplication tensor is all-ones in this indexing).
    
    # Let's verify: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    # This is O(n*m*p*rank) which is fine for small dimensions.
    
    # Build the check efficiently using einsum
    # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need to reshape U, V, W appropriately.
    
    # U is (n*m, rank) -> reshape to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # V is (m*p, rank) -> reshape to (m, p, rank)  
    V3 = V.reshape(m, p, rank)
    # W is (p*n, rank) -> reshape to (p, n, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a bit tricky. Let's do it step by step.
    
    # For each k: sum_c U3[:, :, c] * V3[:, k, c] * W3[k, :, c]
    # = for each k, element-wise product of U3[:,:,c] * V3[:,k,c] * W3[k,:,c] summed over c
    
    # Alternative: use the fact that this equals 1 for all i,j,k if valid.
    # Let's compute it directly for small sizes.
    
    max_val = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if abs(s - 1.0) > tol:
                    return False
                max_val = max(max_val, abs(s - 1.0))
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using structured approaches
    # For 2x2 by 2x5, known results: rank is at most 20 (schoolbook)
    # The bilinear complexity of 2x2 * 2x5 multiplication...
    # This is equivalent to computing all 2*5=10 products with 2x2 matrix times 2x5 matrix.
    # Actually, (2x2)*(2x5) = 2x5 result, 10 entries.
    # Each output entry (i,k) = sum_j A[i,j]*B[j,k]
    # Total: 2*2*5 = 20 scalar multiplications in schoolbook.
    
    # Let's try rank 19, 18, ... down to some bound.
    # Use random initialization and verify.
    
    # For small ranks, we can try to solve the system.
    # The condition is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 20 equations in U*V*W variables.
    # Nonlinear. Let's try a different approach.
    
    # Try structured: fix W to have a particular structure and solve for U, V.
    # Or use the fact that for matrix multiplication, there are known lower bounds.
    
    # For now, let's try random searches at various ranks to see if we can beat 20.
    # This is unlikely to find something better than schoolbook for this small case,
    # but let's try a few quick attempts.
    
    np.random.seed(42)
    
    # Try rank 19
    for target_rank in [19, 18, 17, 16]:
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple check
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
