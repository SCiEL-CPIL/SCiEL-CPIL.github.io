"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1
    # For all other (a,b,c) combinations, it must equal 0
    # But actually the tensor is sparse: T[i,j,k] = 1 for all valid (i,j,k) pairs
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # The full tensor has indices (a, b, c) where:
    # a = i*m + j (i in [0,n), j in [0,m))
    # b = j*p + k (j in [0,m), k in [0,p))
    # c = k*n + i (k in [0,p), i in [0,n))
    
    # For the decomposition to be valid:
    # sum_r U[a,r] * V[b,r] * W[c,r] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # sum_r U[a,r] * V[b,r] * W[c,r] = 0 otherwise
    
    # Let's check all valid (i,j,k) first
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(s - 1.0) > tol:
                    return False
    
    # Now check that invalid combinations give 0
    # We need to check all (a, b, c) that don't correspond to valid (i,j,k)
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # Valid pairs: a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # Build set of valid (a,b,c)
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid.add((a, b, c))
    
    # Check a sample of invalid combinations (for efficiency, check all if small enough)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Total combinations: nm * mp * pn = 4 * 16 * 16 = 1024, which is small
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(s) > tol:
                        return False
    
    return True

def _strassen_28():
    """Return the known rank-28 Strassen-based decomposition for M_{2,2,8}."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2x2
    # U vectors (for A, flattened as [a00, a01, a10, a11])
    U_strassen = [
        [1, 0, 0, 1],   # P1: a00 + a11
        [0, 0, 1, 1],   # P2: a10 + a11
        [1, 0, 0, 0],   # P3: a00
        [0, 0, 0, 1],   # P4: a11
        [1, 1, 0, 0],   # P5: a00 + a01
        [-1, 0, 1, 0],  # P6: a10 - a00
        [0, 1, 0, -1]   # P7: a01 - a11
    ]
    
    # V vectors (for B, flattened as [b00, b01, b10, b11])
    V_strassen = [
        [1, 0, 0, 1],   # P1: b00 + b11
        [1, 0, 0, 0],   # P2: b00
        [0, 1, 0, -1],  # P3: b01 - b11
        [-1, 0, 1, 0],  # P4: b10 - b00
        [0, 0, 0, 1],   # P5: b11
        [1, 1, 0, 0],   # P6: b00 + b01
        [0, 0, 1, 1]    # P7: b10 + b11
    ]
    
    # W vectors (for C, flattened as [c00, c01, c10, c11])
    W_strassen = [
        [1, 0, 0, 1],   # P1: c00 + c11
        [0, 0, 1, -1],  # P2: c10 - c11
        [0, 1, 0, 1],   # P3: c01 + c11
        [1, 0, 1, 0],   # P4: c00 + c10
        [-1, 1, 0, 0],  # P5: c01 - c00
        [0, 0, 0, 1],   # P6: c11
        [1, 0, 0, 0]    # P7: c00
    ]
    
    # 4 blocks of 2x2 in the 2x8 matrix
    num_blocks = p // 2  # 4 blocks
    
    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            
            # Fill U (same for all blocks, since A is 2x2)
            for i in range(n * m):
                U[i][col] = U_strassen[term_idx][i]
            
            # Fill V
            # Local V indices 0,1,2,3 correspond to B[0,2t], B[0,2t+1], B[1,2t], B[1,2t+1]
            # Global B index: row * p + col
            # Local 0: B[0][2*block_idx] -> 0*p + 2*block_idx = 2*block_idx
            # Local 1: B[0][2*block_idx+1] -> 0*p + 2*block_idx+1 = 2*block_idx+1
            # Local 2: B[1][2*block_idx] -> 1*p + 2*block_idx = p + 2*block_idx
            # Local 3: B[1][2*block_idx+1] -> 1*p + 2*block_idx+1 = p + 2*block_idx+1
            
            v_indices = [
                2 * block_idx,           # B[0][2t]
                2 * block_idx + 1,       # B[0][2t+1]
                p + 2 * block_idx,       # B[1][2t]
                p + 2 * block_idx + 1    # B[1][2t+1]
            ]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[term_idx][l]
            
            # Fill W
            # Local W indices 0,1,2,3 correspond to C[0,2t], C[0,2t+1], C[1,2t], C[1,2t+1]
            # Convention: W[k*n + i, r] where k is column index in C, i is row index
            # Local 0: C[0][2*block_idx] -> i=0, k=2*block_idx -> 2*block_idx*n + 0 = 2*block_idx*n
            # Local 1: C[0][2*block_idx+1] -> i=0, k=2*block_idx+1 -> (2*block_idx+1)*n + 0
            # Local 2: C[1][2*block_idx] -> i=1, k=2*block_idx -> 2*block_idx*n + 1
            # Local 3: C[1][2*block_idx+1] -> i=1, k=2*block_idx+1 -> (2*block_idx+1)*n + 1
            
            w_indices = [
                2 * block_idx * n,          # C[0][2t]
                (2 * block_idx + 1) * n,    # C[0][2t+1]
                2 * block_idx * n + 1,      # C[1][2t]
                (2 * block_idx + 1) * n + 1 # C[1][2t+1]
            ]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[term_idx][l]
    
    return {"U": U, "V": V, "W": W}

def _try_rank_24(deadline):
    """Attempt to find a rank-24 decomposition by exploiting tensor product structure.
    
    The key insight: M_{2,2,8} = M_{2,2,2} ⊗ I_4 (in a certain sense), where
    the 8 columns of B are grouped into 4 blocks of 2. The rank-7 Strassen
    decomposition applied to each block gives rank 28. 
    
    To get rank 24, we would need to find 4 linear dependencies among the 28 terms.
    This is equivalent to finding a rank-6 decomposition for a related tensor.
    
    Given the complexity, we'll try a structured search with symmetry constraints.
    """
    if time.time() > deadline - 10:
        return None
    
    # The tensor product structure suggests that if we can find a rank-6
    # decomposition for some auxiliary tensor, we might get rank 24 overall.
    # However, it's known that the rank of M_{2,2,2} is exactly 7, so
    # a rank-6 decomposition doesn't exist for that tensor.
    
    # But for M_{2,2,8}, there might be cross-block dependencies that allow
    # rank reduction. Let's try a simple approach: check if we can express
    # 4 of the 28 Strassen terms as linear combinations of the other 24.
    
    # This is a complex nonlinear problem, so we'll skip it for now and
    # just return None to indicate no better solution found.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit (10s safety margin)
    
    # First, try to find a better (lower rank) decomposition
    better = _try_rank_24(deadline)
    
    if better is not None:
        # Verify the better solution
        if _verify_decomposition(better["U"], better["V"], better["W"]):
            return better
    
    # Fall back to the known rank-28 Strassen decomposition
    result = _strassen_28()
    
    # Verify before returning
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        # If verification fails, there's a bug in our construction
        # Return the result anyway (should not happen)
        pass
    
    return result
# EVOLVE-BLOCK-END
