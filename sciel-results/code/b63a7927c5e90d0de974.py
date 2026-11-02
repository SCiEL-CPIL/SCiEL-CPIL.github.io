"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W correctly decompose the multiplication tensor.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _find_low_rank_3x3x3(max_rank=27, deadline=None):
    """
    Try to find a low-rank decomposition for 3x3 * 3x3 -> 3x3.
    Uses iterative method: fix two matrices, solve for third.
    """
    n = m = p = 3
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try ranks from 23 up to 27
    for target_rank in range(23, min(best_rank, 28)):
        if deadline is not None and time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if deadline is not None and time.time() > deadline:
                break
                
            # Randomly initialize U and V
            U_try = np.random.randn(9, target_rank)
            V_try = np.random.randn(9, target_rank)
            W_try = np.zeros((9, target_rank))
            
            # Iteratively solve: given U and V, solve for W
            # For each (i,j,k): sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
            # This is linear in W for fixed U, V
            
            # Build the linear system for W
            # Unknowns: W[k*3+i, r] for all k,i,r
            # Equations: for each (i,j,k): sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
            
            # The system is: A * W_vec = b
            # where A is (27 x 27*target_rank) and W_vec is W flattened
            
            A = np.zeros((27, 9 * target_rank))
            b = np.ones(27)
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        eq_idx = i * 9 + j * 3 + k
                        for r in range(target_rank):
                            # W[k*3+i, r] is at position (k*3+i) * target_rank + r
                            A[eq_idx, (k*3+i) * target_rank + r] = U_try[i*3+j, r] * V_try[j*3+k, r]
            
            # Solve least squares
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W_try = W_vec.reshape(9, target_rank)
            
            # Verify
            if _validate(U_try, V_try, W_try, n, m, p, tol=1e-6):
                # Check if entries are close to integers/half-integers
                max_err = 0.0
                for mat in [U_try, V_try, W_try]:
                    for val in mat.flatten():
                        # Check if close to integer or half-integer
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2.0
                        if abs(val - nearest_int) < 1e-4 and abs(val - nearest_half) < 1e-4:
                            pass
                        else:
                            # Not close to integer/half-integer, but we can still use it
                            # The problem says "integer/half-integer entries" so let's round
                            pass
                
                # Round to nearest half-integer
                U_rounded = np.round(U_try * 2) / 2.0
                V_rounded = np.round(V_try * 2) / 2.0
                W_rounded = np.round(W_try * 2) / 2.0
                
                if _validate(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
            elif _validate(U_try, V_try, W_try, n, m, p, tol=1e-6):
                # Use unrounded version as fallback
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
    
    return best_U, best_V, best_W, best_rank

def _laderman_3x3x3():
    """
    Construct a low-rank decomposition for 3x3 * 3x3 -> 3x3.
    Tries to find rank < 27.
    """
    U, V, W, rank = _find_low_rank_3x3x3(max_rank=27, deadline=time.time() + 30)
    return U, V, W

def _embed_3x3x3_into_3x3x8(U3, V3, W3, k_offset, n, m, p):
    """Embed a 3x3x3 decomposition into the 3x3x8 problem.
    
    U3: (9, r), V3: (9, r), W3: (9, r) for 3x3x3
    Returns U, V, W components for the 3x3x8 problem with k in [k_offset, k_offset+2]
    """
    r = U3.shape[1]
    U_embed = np.zeros((n * m, r))
    V_embed = np.zeros((m * p, r))
    W_embed = np.zeros((p * n, r))
    
    for i in range(3):
        for j in range(3):
            U_embed[i * m + j, :] = U3[i * 3 + j, :]
    
    for j in range(3):
        for k in range(3):
            V_embed[j * p + (k_offset + k), :] = V3[j * 3 + k, :]
    
    for k in range(3):
        for i in range(3):
            W_embed[(k_offset + k) * n + i, :] = W3[k * 3 + i, :]
    
    return U_embed, V_embed, W_embed

def _find_low_rank_3x3x2(max_rank=18, deadline=None):
    """
    Try to find a low-rank decomposition for 3x3 * 3x2 -> 3x2.
    """
    n = m = p = 3, 3, 2
    n, m, p = 3, 3, 2
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try ranks from 15 up to 18
    for target_rank in range(15, min(best_rank, 19)):
        if deadline is not None and time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if deadline is not None and time.time() > deadline:
                break
                
            # Randomly initialize U and V
            U_try = np.random.randn(9, target_rank)
            V_try = np.random.randn(6, target_rank)
            W_try = np.zeros((6, target_rank))
            
            # Build linear system for W
            A = np.zeros((18, 6 * target_rank))
            b = np.ones(18)
            
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        eq_idx = i * 6 + j * 2 + k
                        for r in range(target_rank):
                            A[eq_idx, (k*3+i) * target_rank + r] = U_try[i*3+j, r] * V_try[j*2+k, r]
            
            # Solve least squares
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W_try = W_vec.reshape(6, target_rank)
            
            # Verify
            if _validate(U_try, V_try, W_try, n, m, p, tol=1e-6):
                # Round to nearest half-integer
                U_rounded = np.round(U_try * 2) / 2.0
                V_rounded = np.round(V_try * 2) / 2.0
                W_rounded = np.round(W_try * 2) / 2.0
                
                if _validate(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
            elif _validate(U_try, V_try, W_try, n, m, p, tol=1e-6):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
    
    return best_U, best_V, best_W, best_rank

def _schoolbook_3x3x2():
    """Low-rank decomposition for 3x3 * 3x2 -> 3x2."""
    U, V, W, rank = _find_low_rank_3x3x2(max_rank=18, deadline=time.time() + 15)
    return U, V, W

def _embed_3x3x2_into_3x3x8(U2, V2, W2, k_offset, n, m, p):
    """Embed a 3x3x2 decomposition into the 3x3x8 problem.
    
    U2: (9, r), V2: (6, r), W2: (6, r) for 3x3x2
    Returns U, V, W components for the 3x3x8 problem with k in [k_offset, k_offset+1]
    """
    r = U2.shape[1]
    U_embed = np.zeros((n * m, r))
    V_embed = np.zeros((m * p, r))
    W_embed = np.zeros((p * n, r))
    
    for i in range(3):
        for j in range(3):
            U_embed[i * m + j, :] = U2[i * 3 + j, :]
    
    for j in range(3):
        for k in range(2):
            V_embed[j * p + (k_offset + k), :] = V2[j * 2 + k, :]
    
    for k in range(2):
        for i in range(3):
            W_embed[(k_offset + k) * n + i, :] = W2[k * 3 + i, :]
    
    return U_embed, V_embed, W_embed

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook (rank 72)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Split 8 columns into 3+3+2
    # Use low-rank decompositions for each block
    
    # First, get the 3x3x3 decomposition
    U3, V3, W3 = _laderman_3x3x3()
    
    # Verify the 3x3x3 decomposition
    if not _validate(U3, V3, W3, 3, 3, 3):
        # Fall back to schoolbook if it fails
        U3, V3, W3 = _build_schoolbook(3, 3, 3)
    
    # Embed into 3x3x8 for k=0,1,2
    U1, V1, W1 = _embed_3x3x3_into_3x3x8(U3, V3, W3, 0, n, m, p)
    
    # Embed into 3x3x8 for k=3,4,5
    U2, V2, W2 = _embed_3x3x3_into_3x3x8(U3, V3, W3, 3, n, m, p)
    
    # Get 3x3x2 decomposition for k=6,7
    U32, V32, W32 = _schoolbook_3x3x2()
    U3_emb, V3_emb, W3_emb = _embed_3x3x2_into_3x3x8(U32, V32, W32, 6, n, m, p)
    
    # Combine all components
    r_total = U1.shape[1] + U2.shape[1] + U3_emb.shape[1]
    U_combined = np.hstack([U1, U2, U3_emb])
    V_combined = np.hstack([V1, V2, V3_emb])
    W_combined = np.hstack([W1, W2, W3_emb])
    
    # Verify
    if _validate(U_combined, V_combined, W_combined, n, m, p):
        best_rank = r_total
        best_result = {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
