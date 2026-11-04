import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition for n x m x p matrix multiplication.
    
    U: shape (n*m, rank)  -- entries for A[i,j] indexed as i*m+j
    V: shape (m*p, rank)  -- entries for B[j,k] indexed as j*p+k
    W: shape (p*n, rank)  -- entries for C[i,k] indexed as k*n+i (trace convention)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def _strassen_2x2_2x2_UVW():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
    Rank = 7.
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, -1, 0],
    ]
    
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    Returns True if valid, False otherwise.
    """
    try:
        np.random.seed(12345)
        
        for trial in range(5):
            A = np.random.randint(-10, 10, (n, m), dtype=float)
            B = np.random.randint(-10, 10, (m, p), dtype=float)
            C_expected = A @ B
            
            rank = len(U[0])
            C_computed = np.zeros((n, p), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_val = W[k * n + i]
                        for r in range(rank):
                            s += u_row[r] * v_row[r] * w_val[r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    except Exception:
        return False

def _strassen_4x4_4x4_UVW():
    """
    Return the Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    Uses block Strassen: treat 4x4 as 2x2 blocks of 2x2 matrices.
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    block_A_coeffs = {
        (0,0): [1, 0, 1, 0, 1, 0, 0],
        (0,1): [0, 0, 0, 0, 1, 0, 1],
        (1,0): [0, 1, 0, 0, 0, -1, 0],
        (1,1): [1, 1, 0, 1, 0, -1, 0],
    }
    
    block_B_coeffs = {
        (0,0): [1, 1, 0, 0, 0, 1, 0],
        (0,1): [0, 0, 1, 0, 0, 1, 0],
        (1,0): [0, 0, 0, 1, 0, 0, 1],
        (1,1): [1, -1, 0, 0, 1, 0, 1],
    }
    
    block_C_coeffs = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    for i in range(4):
        for j in range(4):
            bi, bj = i // 2, j // 2
            ii, jj = i % 2, j % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    U[i*4+j][combined_r] = block_A_coeffs[(bi,bj)][rb] * U2[inner_idx][ri]
    
    for j in range(4):
        for k in range(4):
            bi, bj = j // 2, k // 2
            ii, jj = j % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    V[j*4+k][combined_r] = block_B_coeffs[(bi,bj)][rb] * V2[inner_idx][ri]
    
    for i in range(4):
        for k in range(4):
            bi, bj = i // 2, k // 2
            ii, jj = i % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    W[k*4+i][combined_r] = block_C_coeffs[(bi,bj)][rb] * W2[inner_idx][ri]
    
    return U, V, W

def _extend_4x4_4x4_to_4x4_7():
    """
    Extend the 4x4*4x4 -> 4x4 Strassen decomposition (rank 49) to 4x4*4x7 -> 4x7.
    Uses trivial decomposition (rank 48) for the last 3 columns.
    Total rank = 49 + 48 = 97.
    """
    U4, V4, W4 = _strassen_4x4_4x4_UVW()
    U3, V3, W3 = _trivial_decomposition(4, 4, 3)
    
    U = [row4 + row3 for row4, row3 in zip(U4, U3)]
    V = [row4 + row3 for row4, row3 in zip(V4, V3)]
    
    W = [[0] * 97 for _ in range(28)]
    
    for i in range(16):
        for r in range(49):
            W[i][r] = W4[i][r]
    
    for i3 in range(12):
        k3, i_val = i3 // 4, i3 % 4
        row_in_W = (4 + k3) * 4 + i_val
        for r in range(48):
            W[row_in_W][49 + r] = W3[i3][r]
    
    return U, V, W

def _random_search_for_better_rank(n, m, p, target_rank, time_budget=30):
    """
    Attempt to find a decomposition with rank = target_rank using random initialization
    and verification. This is a heuristic search that may or may not succeed.
    
    For the 4x4x7 problem, we know rank 97 works. We try to find something better.
    The theoretical lower bound is around 75-80 based on known results.
    """
    start = time.time()
    
    # Try a few different random approaches
    for attempt in range(3):
        if time.time() - start > time_budget:
            break
        
        try:
            # Initialize with small random values
            np.random.seed(42 + attempt)
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Verify
            if _verify_decomposition(n, m, p, U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
                
        except Exception:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 7
    
    # Start with the trivial decomposition (rank = 4*4*7 = 112)
    U, V, W = _trivial_decomposition(n, m, p)
    
    # Try the extended Strassen decomposition (rank 97)
    try:
        U_strassen, V_strassen, W_strassen = _extend_4x4_4x4_to_4x4_7()
        
        # Verify
        if _verify_decomposition(n, m, p, U_strassen, V_strassen, W_strassen):
            U, V, W = U_strassen, V_strassen, W_strassen
    except Exception:
        pass
    
    # Try to find a better decomposition if time permits
    if time.time() - start_time < deadline - 20:
        current_rank = len(U[0]) if U else 112
        
        # Try ranks between 80 and current_rank - 1
        for target_rank in range(80, current_rank):
            if time.time() - start_time > deadline - 10:
                break
            
            result = _random_search_for_better_rank(n, m, p, target_rank, time_budget=5)
            if result is not None:
                U, V, W = result
                break
    
    # Verify final result
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to trivial
        U, V, W = _trivial_decomposition(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
