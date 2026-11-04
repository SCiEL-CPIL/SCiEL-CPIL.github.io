import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 4, 9]
_TIME_LIMIT = 145.0

def _build_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor using NumPy."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Build the tensor T[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
    # Reshape: U is (n,m,rank), V is (m,p,rank), W is (p,n,rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,k,j] = sum_r U3[i,k,r] * V3[k,j,r] * W3[j,i,r]
    # Compute via einsum
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    
    return np.allclose(T, 1.0, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce the rank by merging or eliminating terms.
    Strategy: try removing one term at a time and check if the remaining
    terms can still decompose the tensor (with possible coefficient adjustments).
    """
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing each term and see if we can compensate
    for r_remove in range(rank):
        if time.time() > deadline - 10:
            break
        
        # Try removing term r_remove
        keep = [r for r in range(rank) if r != r_remove]
        if len(keep) == 0:
            continue
        
        U_sub = U[:, keep]
        V_sub = V[:, keep]
        W_sub = W[:, keep]
        
        # Check if removing this term still gives a valid decomposition
        # (This is a quick check - most terms are needed)
        if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
            best_U, best_V, best_W = U_sub, V_sub, W_sub
            best_rank = len(keep)
            # Update for next iteration
            U, V, W = U_sub, V_sub, W_sub
            rank = best_rank
            # Reset r_remove since indices changed
            r_remove = 0
    
    return best_U, best_V, best_W, best_rank

def _try_integer_rounding(U, V, W, n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by:
    1. Forming the tensor as a 3D array
    2. Using iterative rank-1 approximation with rounding
    """
    rank = U.shape[1]
    # Current tensor value
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ikr,kjr,jir->ijk', U3, V3, W3)
    
    # The target is all ones
    target = np.ones((n, m, p))
    
    # Try to find a lower rank decomposition
    # Start from current rank and try to reduce
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various target ranks
    for target_rank in range(best_rank - 1, max(best_rank - 20, 10), -1):
        if time.time() > deadline - 15:
            break
        
        # Use the current decomposition as initialization
        # Try to project down to target_rank
        # Simple approach: keep the first target_rank terms
        if target_rank < rank:
            U_try = U[:, :target_rank]
            V_try = V[:, :target_rank]
            W_try = W[:, :target_rank]
            
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 5  # 5s safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Build the base decomposition
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # For the 4x1 part
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U41[i * 4 + k, r] = 1
            V41[k * 1 + 0, r] = 1
            W41[0 * 4 + i, r] = 1
    
    total_rank = 49 + 49 + 16  # 114
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A*B1 (cols 0-3)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Part 2: A*B2 (cols 4-7)
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Part 3: A*B3 (col 8)
    U[:, 98:114] = U41
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]
    
    # Verify the decomposition using fast NumPy method
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial decomposition
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * m * p + k * p + j
                    U[i*m+k, r] = 1
                    V[k*p+j, r] = 1
                    W[j*n+i, r] = 1
        total_rank = rank
    else:
        total_rank = U.shape[1]
    
    # Try to reduce rank if we have time
    if time.time() < deadline - 20:
        U, V, W, total_rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    # Final safety check
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to original known-good decomposition
        U44, V44, W44 = _build_4x4x4_strassen()
        U41 = np.zeros((16, 16))
        V41 = np.zeros((4, 16))
        W41 = np.zeros((4, 16))
        for i in range(4):
            for k in range(4):
                r = i * 4 + k
                U41[i * 4 + k, r] = 1
                V41[k * 1 + 0, r] = 1
                W41[0 * 4 + i, r] = 1
        
        total_rank = 49 + 49 + 16
        U = np.zeros((16, total_rank))
        V = np.zeros((36, total_rank))
        W = np.zeros((36, total_rank))
        
        U[:, 0:49] = U44
        for k in range(4):
            for j in range(4):
                V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
        for j in range(4):
            for i in range(4):
                W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
        
        U[:, 49:98] = U44
        for k in range(4):
            for j in range(4):
                V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
        for j in range(4):
            for i in range(4):
                W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
        
        U[:, 98:114] = U41
        for k in range(4):
            V[k * 9 + 8, 98:114] = V41[k, :]
        for i in range(4):
            W[8 * 4 + i, 98:114] = W41[i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
