"""
5x5 matrix multiplication - rank reduction via structured algebraic search.
Combines schoolbook baseline with Strassen-style block decomposition for 2x2
blocks and randomized verification. Attempts to find rank < 125.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    C[i,k] = sum_c W[k*n+i, c] * (A_flat . U[:,c]) * (B_flat . V[:,c])
    where A_flat has A[i,j] at position i*m+j, B_flat has B[j,k] at j*p+k.
    """
    r = U.shape[1]
    # Random test matrices with fixed seed for reproducibility
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()  # length n*m, index i*m+j
    B_flat = B.flatten()  # length m*p, index j*p+k
    
    # Compute C from decomposition
    L = A_flat @ U  # shape (r,)
    M = B_flat @ V  # shape (r,)
    
    # C_flat[k*n+i] = sum_c W[k*n+i, c] * L[c] * M[c]
    C_flat = W @ (L * M)  # shape (p*n,)
    C_computed = C_flat.reshape(p, n).T  # C[i,k]
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _strassen_2x2_decomposition():
    """
    Return the Strassen 2x2 decomposition as U, V, W factors.
    For 2x2 matrices, rank 7 decomposition.
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    # M1 = (A11-A22)(B12+B22)
    U[0, 0] = 1.0; U[3, 0] = -1.0
    V[1, 0] = 1.0; V[3, 0] = 1.0
    
    # M2 = (A11+A22)B22
    U[0, 1] = 1.0; U[3, 1] = 1.0
    V[3, 1] = 1.0
    
    # M3 = A11(B11-B22)
    U[0, 2] = 1.0
    V[0, 2] = 1.0; V[3, 2] = -1.0
    
    # M4 = A22(B11+B22)
    U[3, 3] = 1.0
    V[0, 3] = 1.0; V[3, 3] = 1.0
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    
    # W: C[i,k] coefficients, W[k*n+i, c]
    # C[0,0] = M1+M4-M5+M7 -> W[0, c]
    W[0, 0] = 1.0; W[0, 3] = 1.0; W[0, 4] = -1.0; W[0, 6] = 1.0
    
    # C[0,1] = M3+M5 -> W[1*2+0, c] = W[2, c]
    W[2, 2] = 1.0; W[2, 4] = 1.0
    
    # C[1,0] = M2+M4 -> W[0*2+1, c] = W[1, c]
    W[1, 1] = 1.0; W[1, 3] = 1.0
    
    # C[1,1] = M1-M2+M3+M6 -> W[1*2+1, c] = W[3, c]
    W[3, 0] = 1.0; W[3, 1] = -1.0; W[3, 2] = 1.0; W[3, 5] = 1.0
    
    return U, V, W

def _embed_2x2_block(U2, V2, W2, i0, j0, k0, n, m, p):
    """
    Embed a 2x2 Strassen decomposition into a larger 5x5 matrix multiplication.
    The block covers A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2].
    """
    U_full = np.zeros((n*m, 7), dtype=float)
    V_full = np.zeros((m*p, 7), dtype=float)
    W_full = np.zeros((p*n, 7), dtype=float)
    
    # Map 2x2 indices to full matrix indices
    for local_i in range(2):
        for local_j in range(2):
            global_i = i0 + local_i
            global_j = j0 + local_j
            local_idx = local_i * 2 + local_j
            global_idx = global_i * m + global_j
            U_full[global_idx, :] = U2[local_idx, :]
    
    for local_j in range(2):
        for local_k in range(2):
            global_j = j0 + local_j
            global_k = k0 + local_k
            local_idx = local_j * 2 + local_k
            global_idx = global_j * p + global_k
            V_full[global_idx, :] = V2[local_idx, :]
    
    for local_k in range(2):
        for local_i in range(2):
            global_k = k0 + local_k
            global_i = i0 + local_i
            local_idx = local_k * 2 + local_i
            global_idx = global_k * n + global_i
            W_full[global_idx, :] = W2[local_idx, :]
    
    return U_full, V_full, W_full

def _combine_blocks(block_list, n, m, p):
    """
    Combine multiple 2x2 block decompositions into a full 5x5 decomposition.
    Each block is a (U, V, W) triple for a 2x2 sub-problem.
    """
    total_rank = 0
    for _, _, _ in block_list:
        total_rank += 7  # Each 2x2 block uses rank 7
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    offset = 0
    for U_block, V_block, W_block in block_list:
        U[:, offset:offset+7] = U_block
        V[:, offset:offset+7] = V_block
        W[:, offset:offset+7] = W_block
        offset += 7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for 120s limit
    
    # First, get schoolbook as fallback
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # Strategy 1: Use Strassen-style 2x2 blocks to reduce rank
    # For 5x5, we can place 2x2 blocks in non-overlapping positions
    # Possible 2x2 block positions in a 5x5 matrix:
    # Rows can start at 0,1,2,3 and cols at 0,1,2,3
    # We need to cover as much of the 5x5 as possible with non-overlapping 2x2 blocks
    
    # Let's try a tiling strategy:
    # Place 2x2 blocks at (0,0), (0,2), (0,3), (2,0), (2,2), (2,3), (3,0), (3,2), (3,3)
    # This gives 9 blocks * 7 = 63 rank, but many will be redundant
    
    # Actually, for a proper decomposition, we need to be more careful.
    # Let's try a simpler approach: use 2x2 Strassen for a 2x2 subblock and
    # schoolbook for the rest, then see if we can merge.
    
    # Try a hybrid: 2x2 Strassen block at top-left, schoolbook for remaining
    # This won't directly give a valid decomposition because the blocks interact.
    
    # Better approach: Try to find a valid decomposition using the tensor structure.
    # For now, let's try randomized search with verification.
    
    np.random.seed(12345)
    
    # Try ranks from 100 down to best_rank-1
    # This is expensive, so we limit attempts
    max_attempts = 50
    attempts = 0
    
    for r in range(100, 125):  # Try ranks 100 to 124
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            continue
        
        for attempt in range(max_attempts // 10):
            if time.time() - start_time > time_limit:
                break
            attempts += 1
            
            # Random integer/half-integer initialization
            scale = 1.0
            U = np.random.choice([-1, 0, 1], size=(n*m, r)) * scale
            V = np.random.choice([-1, 0, 1], size=(m*p, r)) * scale
            W = np.random.choice([-1, 0, 1], size=(p*n, r)) * scale
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank <= r:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
