import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# Rank-minimal exact tensor decomposition for 3x4 * 4x6
# Schoolbook rank: 3*4*6 = 72
# We search for a lower rank using a randomised / structured approach.

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    # Matmul tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For all i in [0,n), j in [0,m), k in [0,p): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # For other combinations (not all three indices valid in the product) the sum should be 0.
    # Actually, the matmul tensor is defined as: for matrices A (n x m), B (m x p), C = A @ B (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for the "core" tensor entries and 0 otherwise.
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 (it's the identity in the j-index pairing)
    # More precisely: the matmul tensor has T[i,j,k] = delta_{i,j} * delta_{j,k} ... no.
    # The standard: C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's a 3-tensor where each entry is 1)
    # Wait, that's not right either. The tensor is:
    # T[i,j,k] = 1 if we think of it as the "structure constants"
    # Actually, the matrix multiplication tensor has:
    # T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # No, that's wrong. Let me reconsider.
    #
    # The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form, this is a trilinear form:
    # sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # where T[i,j,k] = 1 for all valid indices.
    #
    # So the decomposition requires:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Start with rank = n*m*p - 1 and go down
    # Use a simple approach: try random initializations and see if we can verify
    
    # For small problems, we can try a few different ranks
    # Known lower bounds: rank >= n + m + p - 2 (for rectangular matmul, this is a conjecture)
    # For 3x4x6: lower bound is 3+4+6-2 = 11 (very loose)
    # Actually the Strassen rank for 2x2x2 is 7
    # For our problem, let's just try to find any rank < 72
    
    # Try ranks from 71 down to some minimum
    # We'll use a simple random search approach
    
    rng = np.random.RandomState(42)
    
    min_rank_to_try = max(n + m + p - 2, 10)
    
    for rank in range(rank_schoolbook - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random initializations at this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10.0:
                break
            
            # Random initialization
            U_r = rng.randn(n * m, rank)
            V_r = rng.randn(m * p, rank)
            W_r = rng.randn(p * n, rank)
            
            # Check if this decomposes the tensor
            # We need: for all i,j,k: sum_r U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] == 1
            
            # Efficient check: compute the tensor and compare
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        s = np.dot(np.dot(U_r[i*m+j], V_r[j*p+k]), W_r[k*n+i])
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
            
            if ok:
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                # Convert to list format
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    
    # Return schoolbook if no better found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
