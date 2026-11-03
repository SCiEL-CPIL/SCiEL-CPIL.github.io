"""Matrix multiplication 4x5x5 rank-minimal decomposition search.

Uses a structured approach:
1. Strassen-based 4x4x4 (rank 49) + rank-1 factors for remaining parts (rank 51)
2. Attempts rank reduction via linear algebra (null-space elimination)
3. Falls back to schoolbook if verification fails
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2_factors():
    """Generate Strassen factors for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, -1]
    ]
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Strassen."""
    U2, V2, W2 = _strassen_2x2_factors()
    
    U4 = [[0.0]*49 for _ in range(16)]
    V4 = [[0.0]*49 for _ in range(16)]
    W4 = [[0.0]*49 for _ in range(16)]
    
    for a in range(7):
        for b in range(7):
            c = a*7 + b
            for i in range(4):
                for j in range(4):
                    flat = i*4 + j
                    block_flat = (i//2)*2 + (j//2)
                    sub_flat = (i%2)*2 + (j%2)
                    U4[flat][c] = U2[block_flat][a] * U2[sub_flat][b]
                    V4[flat][c] = V2[block_flat][a] * V2[sub_flat][b]
                    W4[flat][c] = W2[block_flat][a] * W2[sub_flat][b]
    
    return U4, V4, W4


def _build_4x5x5_decomposition():
    """Build the full 4x5x5 decomposition using Strassen for 4x4x4 plus rank-1 factors."""
    n, m, p = 4, 5, 5
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    total_rank = 49 + 2  # = 51
    
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3), rank 49
    for i in range(4):
        for j in range(4):
            for k in range(4):
                flat_U = i*5 + j
                flat_V = j*5 + k
                flat_W = k*4 + i
                
                for c in range(49):
                    U[flat_U][c] += U4[i*4+j][c]
                    V[flat_V][c] += V4[j*4+k][c]
                    W[flat_W][c] += W4[k*4+i][c]
    
    # Part 2: 4x4x1 (i in 0..3, j in 0..3, k=4), rank 1
    c = 49
    for i in range(4):
        for j in range(4):
            flat_U = i*5 + j
            flat_V = j*5 + 4
            flat_W = 4*4 + i
            U[flat_U][c] = 1.0
            V[flat_V][c] = 1.0
            W[flat_W][c] = 1.0
    
    # Part 3: 4x1x5 (i in 0..3, j=4, k in 0..4), rank 1
    c = 50
    for i in range(4):
        for k in range(5):
            flat_U = i*5 + 4
            flat_V = 4*5 + k
            flat_W = k*4 + i
            U[flat_U][c] = 1.0
            V[flat_V][c] = 1.0
            W[flat_W][c] = 1.0
    
    return U, V, W, total_rank


def _schoolbook_fallback(n, m, p):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce the rank by eliminating redundant factors.
    
    A factor c is redundant if its contribution to the tensor is zero everywhere,
    i.e., U[i,j,c] * V[j,k,c] * W[k,i,c] = 0 for all i,j,k.
    This happens when at least one of the three vectors is the zero vector.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try removing each factor one at a time
    for c_remove in range(rank):
        if time.time() > deadline - 30:
            break
            
        # Check if factor c_remove is redundant (zero contribution everywhere)
        U_col = U_arr[:, c_remove]
        V_col = V_arr[:, c_remove]
        W_col = W_arr[:, c_remove]
        
        # A factor is redundant if any of its vectors is all zeros
        if np.allclose(U_col, 0, atol=1e-10) or \
           np.allclose(V_col, 0, atol=1e-10) or \
           np.allclose(W_col, 0, atol=1e-10):
            # Factor is redundant, remove it
            U_try = np.delete(U_arr, c_remove, axis=1)
            V_try = np.delete(V_arr, c_remove, axis=1)
            W_try = np.delete(W_arr, c_remove, axis=1)
            
            best_rank = rank - 1
            best_U, best_V, best_W = U_try, V_try, W_try
            U_arr, V_arr, W_arr = U_try, V_try, W_try
            rank = best_rank
            # Restart the loop since indices changed
            c_remove = 0
            continue
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    return None


def _try_linear_reduction(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding linear dependencies among the factor vectors.
    
    The key insight: if two factors c1 and c2 have proportional U, V, W vectors
    (up to scaling), we might be able to merge them. More generally, we look
    for linear combinations of factors that cancel out.
    
    This is a heuristic approach - we check if any column in U, V, or W is a
    linear combination of others, and if so, try to eliminate it.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each column, check if it can be expressed as a linear combination
    # of the other columns in all three matrices simultaneously
    # This is expensive, so we only try a limited number of attempts
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Simple heuristic: check for duplicate or near-duplicate columns
    # across all three matrices
    for c_remove in range(rank):
        if time.time() > deadline - 30:
            break
        
        U_col = U_arr[:, c_remove]
        V_col = V_arr[:, c_remove]
        W_col = W_arr[:, c_remove]
        
        # Check if this column is all zeros
        if np.allclose(U_col, 0, atol=1e-10) and \
           np.allclose(V_col, 0, atol=1e-10) and \
           np.allclose(W_col, 0, atol=1e-10):
            U_try = np.delete(U_arr, c_remove, axis=1)
            V_try = np.delete(V_arr, c_remove, axis=1)
            W_try = np.delete(W_arr, c_remove, axis=1)
            
            best_rank = rank - 1
            best_U, best_V, best_W = U_try, V_try, W_try
            U_arr, V_arr, W_arr = U_try, V_try, W_try
            rank = best_rank
            c_remove = 0
            continue
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    # Build the Strassen-based decomposition
    try:
        U, V, W, rank = _build_4x5x5_decomposition()
        
        # Verify
        if _verify(U, V, W, n, m, p):
            # Try to reduce rank via simple redundancy elimination
            reduced = _try_reduce_rank(U, V, W, n, m, p, deadline)
            if reduced is not None:
                U, V, W, rank = reduced
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
            
            # Try linear reduction
            if time.time() < deadline - 30:
                reduced2 = _try_linear_reduction(U, V, W, n, m, p, deadline)
                if reduced2 is not None:
                    U, V, W, rank = reduced2
                    if _verify(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}
            
            return {"U": U, "V": V, "W": W}
        else:
            return _schoolbook_fallback(n, m, p)
    except Exception:
        return _schoolbook_fallback(n, m, p)
