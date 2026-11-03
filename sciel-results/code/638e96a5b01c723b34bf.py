# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 3, 6
    
    # Target: Find a rank R decomposition for 3x3x3, then use 2R for 3x3x6
    # Schoolbook 3x3x3 is 27.
    # Try to find rank 26, 25, 24, 23
    
    best_rank_333 = 27
    best_decomp_333 = _schoolbook_333()
    
    # Search for lower rank in 3x3x3
    for rank in range(26, 22, -1):
        if time.time() > deadline - 30:
            break
        
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(attempt * 104729 + rank * 13 + 3000)
            
            # Use integers {-1, 0, 1}
            U = rng.choice([-1, 0, 1], size=(9, rank))
            V = rng.choice([-1, 0, 1], size=(9, rank))
            W = rng.choice([-1, 0, 1], size=(9, rank))
            
            # Verify 3x3x3
            if _verify_333(U, V, W):
                best_rank_333 = rank
                best_decomp_333 = {"U": U, "V": V, "W": W}
                break
        
        if best_rank_333 == rank:
            # Found a decomposition for this rank, continue to next lower rank
            continue
        # If not found, we might break early if time is tight, but let's try a few more
    
    # Construct 3x3x6 decomposition from 3x3x3
    # Rank will be 2 * best_rank_333
    final_rank = 2 * best_rank_333
    
    U_small = best_decomp_333["U"]
    V_small = best_decomp_333["V"]
    W_small = best_decomp_333["W"]
    
    # U_full: (9, 2*R)
    # First R columns: U_small
    # Last R columns: U_small
    U_full = np.hstack([U_small, U_small])
    
    # V_full: (18, 2*R)
    # First 9 rows, first R cols: V_small
    # First 9 rows, last R cols: 0
    # Last 9 rows, first R cols: 0
    # Last 9 rows, last R cols: V_small
    V_full = np.zeros((18, 2 * best_rank_333))
    V_full[0:9, 0:best_rank_333] = V_small
    V_full[9:18, best_rank_333:2*best_rank_333] = V_small
    
    # W_full: (18, 2*R)
    # First 9 rows, first R cols: W_small
    # First 9 rows, last R cols: 0
    # Last 9 rows, first R cols: 0
    # Last 9 rows, last R cols: W_small
    W_full = np.zeros((18, 2 * best_rank_333))
    W_full[0:9, 0:best_rank_333] = W_small
    W_full[9:18, best_rank_333:2*best_rank_333] = W_small
    
    result = {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
    
    # Verify final result
    if _verify_336(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook_336()

def _schoolbook_333():
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _schoolbook_336():
    rank = 54
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[i * 3 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_333(U, V, W):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    M = np.zeros((9, 9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
    M_rec = np.zeros_like(M)
    for t in range(rank):
        M_rec += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    return np.allclose(M, M_rec, atol=1e-9)

def _verify_336(U, V, W):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    M = np.zeros((9, 18, 18), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(6):
                a = i * 3 + j
                b = j * 6 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
    M_rec = np.zeros_like(M)
    for t in range(rank):
        M_rec += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    return np.allclose(M, M_rec, atol=1e-9)
# EVOLVE-BLOCK-END
