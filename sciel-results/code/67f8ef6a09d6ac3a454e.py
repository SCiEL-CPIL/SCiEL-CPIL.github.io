"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1;  U[3, 0] = 1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1
    
    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1;  W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[2, 2] = 1;  W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[0, 3] = 1;  W[1, 3] = 1
    
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[2, 4] = 1
    
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2x3_rank11():
    """
    Build a rank-11 decomposition for 2x2 x 2x3 matrix multiplication.
    Uses 7 Strassen terms for columns 0-1 and 4 schoolbook terms for column 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # First 7 terms: Strassen for columns 0,1
    U[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j * p + k_local, 0:7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[k_local * n + i, 0:7] = W_str[k_local * 2 + i, :]
    
    # Terms 7-10: Schoolbook for column 2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U[0, 7] = 1; V[0, 7] = 1; W[4, 7] = 1  # A[0,0]*B[0,2] -> C[0,2]
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1  # A[0,1]*B[1,2] -> C[0,2]
    U[2, 9] = 1; V[4, 9] = 1; W[5, 9] = 1  # A[1,0]*B[0,2] -> C[1,2]
    U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1  # A[1,1]*B[1,2] -> C[1,2]
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _find_2x2x3_rank10():
    """
    Search for a rank-10 decomposition of 2x2x3 matrix multiplication.
    Uses a structured search over small integer patterns.
    Returns (U, V, W) of shapes (4,10), (6,10), (6,10) or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # We need: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # U: (4,10), V: (6,10), W: (6,10)
    
    # Strategy: Try rank-10 decompositions where the first 7 terms form a 
    # "modified Strassen" and 3 additional terms handle column 2.
    # Key insight: We can allow the extra 3 terms to have small contributions
    # to columns 0 and 1 that cancel with modifications to the Strassen terms.
    
    # Alternative: Try a completely different rank-10 structure.
    # Based on literature, the border rank of 2x2x3 is 10, and rank might be 11.
    # Let's try to verify if rank 10 is achievable.
    
    # Let's try a specific known construction:
    # Use the fact that 2x2x3 can be seen as 2x2x2 + 2x2x1.
    # Rank(2x2x2) = 7 (Strassen), Rank(2x2x1) = 4 (schoolbook).
    # But we might be able to do better by sharing.
    
    # Try: 6 terms for "almost" 2x2x2 + 4 terms for column 2 with sharing.
    
    # Let me try a direct search with constrained entries.
    # Fix V to have a specific structure and solve for U, W.
    
    # Approach: Try V with entries in {-1, 0, 1}, then solve linear system for U*W.
    
    # For efficiency, try a few specific V patterns.
    
    # Pattern 1: V is block-diagonal with 2x2 Strassen block + 2 schoolbook columns
    # This gives rank 7+4=11, not 10.
    
    # Pattern 2: Try to find a rank-10 by allowing some cross-terms.
    
    # Let's try a randomized search with verification.
    np.random.seed(42)
    
    for attempt in range(10000):
        # Generate random sparse pattern
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Fill with random ±1 entries, sparse
        for r in range(rank):
            # U column: 1-3 nonzeros
            n_u = np.random.randint(1, 4)
            u_rows = np.random.choice(4, n_u, replace=False)
            for row in u_rows:
                U[row, r] = np.random.choice([-1, 1])
            
            # V column: 1-3 nonzeros
            n_v = np.random.randint(1, 4)
            v_rows = np.random.choice(6, n_v, replace=False)
            for row in v_rows:
                V[row, r] = np.random.choice([-1, 1])
            
            # W column: 1-2 nonzeros
            n_w = np.random.randint(1, 3)
            w_rows = np.random.choice(6, n_w, replace=False)
            for row in w_rows:
                W[row, r] = np.random.choice([-1, 1])
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    # Random search failed. Try a more structured approach.
    # Use the known Strassen decomposition for 2x2x2 and try to extend it.
    
    # Actually, let's try a specific candidate from the literature.
    # The rank of 2x2x3 matrix multiplication is known to be 11 (not 10).
    # See: Bostan, Giesbrecht, Salvy (2007) or similar references.
    # If rank is 11, then we can't achieve 52 with this approach.
    
    # Let's verify: is the rank of 2x2x3 really 11?
    # The lower bound for 2x2x3 is ceil(4*3/2) = 6 (trivial), 
    # but the actual rank is known to be 11.
    
    # So rank 52 is not achievable with this block structure.
    # Let's return None.
    return None

def _build_rank52():
    """
    Build a rank-52 decomposition if possible, otherwise fall back to rank-53.
    """
    n, m, p = DIMENSIONS
    
    # Try to find rank-10 for 2x2x3
    result = _find_2x2x3_rank10()
    if result is not None:
        U3, V3, W3 = result
        # Build full decomposition: 6 Strassen blocks (cols 0-11) + 1 rank-10 block (cols 12-14)
        U_str, V_str, W_str = _build_strassen_2x2x2()
        total_rank = 6 * 7 + 10  # 52
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col_offset = 0
        for b in range(6):
            start_k = b * 2
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        
        # Last block: 2x2x3 rank-10 for cols 12,13,14
        U[:, col_offset:col_offset+10] = U3
        for j in range(m):
            for k_local in range(3):
                k_global = 12 + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V3[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = 12 + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W3[k_local * 2 + i, :]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    # Fallback to rank 53
    return _build_rank53()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank 52 first
    U, V, W = _build_rank52()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to rank 53
        U, V, W = _build_rank53()
        best_rank = 53
        best_U, best_V, best_W = U, V, W
        
        if not _verify_decomposition(U, V, W, n, m, p):
            # Final fallback: schoolbook
            U, V, W = _build_schoolbook(n, m, p)
            best_rank = 60
            best_U, best_V, best_W = U, V, W
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
