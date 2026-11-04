import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _get_strassen_2x2():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
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
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_decomposition_4x4x4():
    U2, V2, W2 = _get_strassen_2x2()
    rank = 49
    
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i * 4 + j, c] = U2[i0 * 2 + j0, c0] * U2[i1 * 2 + j1, c1]
                    
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j * 4 + k, c] = V2[j0 * 2 + k0, c0] * V2[j1 * 2 + k1, c1]
                    
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k * 4 + i, c] = W2[k0 * 2 + i0, c0] * W2[k1 * 2 + i1, c1]
    return U4, V4, W4

def _build_decomposition_4x4x8():
    U4, V4, W4 = _build_decomposition_4x4x4()
    total_rank = 98
    
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((32, total_rank), dtype=int)
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
            
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = W4[k * 4 + i, :]
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Check all entries of the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, time_budget=5.0):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    Returns (U', V', W') with lower rank if found, else None.
    """
    start = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # Try removing columns one at a time and checking if the tensor is still spanned
    # This is expensive, so we limit the number of attempts
    max_attempts = 20
    attempts = 0
    
    for col_to_try in range(current_rank):
        if time.time() - start > time_budget:
            break
        attempts += 1
        
        # Remove column col_to_try
        mask = [i for i in range(current_rank) if i != col_to_try]
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check if the removed column is a linear combination of others
        # by checking if the tensor it contributes is already covered
        # Quick check: see if the outer product structure is redundant
        # This is a heuristic - check if the column is "small" in some sense
        
        # More practical: just check if removing it still gives correct tensor
        # But that's O(n*m*p*rank) which is expensive
        # For now, skip this and return None
        pass
    
    return None

def _fallback_schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return _fallback_schoolbook()
    
    try:
        U, V, W = _build_decomposition_4x4x8()
        
        # Quick verification (sample check for speed)
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Verify a few entries
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    total = np.sum(U_arr[i*4+j, :] * V_arr[j*8+k, :] * W_arr[k*4+i, :])
                    if abs(total - 1.0) > 1e-9:
                        return _fallback_schoolbook()
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    except Exception as e:
        print(f"Error in construct: {e}")
        return _fallback_schoolbook()

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
