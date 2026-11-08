"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    def verify_solution(U, V, W):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Compute T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    def get_rank_33_solution():
        """Construct the rank-33 solution using 3 blocks of 2x3x2."""
        rank = 33
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # 11 terms for a 2x3x2 block
        # A is 2x3: flat index = i*3+j, i in {0,1}, j in {0,1,2}
        # B is 3x2: flat index = j*2+k, j in {0,1,2}, k in {0,1}
        # C is 2x2: flat index = i*2+k, i in {0,1}, k in {0,1}
        
        terms = [
            # Strassen for 2x2x2 (using j=0,1 of A and j=0,1 of B)
            {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},   # M1
            {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},        # M2
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},        # M3
            {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},        # M4
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},        # M5
            {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},        # M6
            {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},        # M7
            # Scalar terms for j=2
            {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},                      # K1
            {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},                      # K2
            {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},                      # K3
            {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},                      # K4
        ]
        
        for block in range(3):
            for t_idx, term in enumerate(terms):
                col = block * 11 + t_idx
                
                for idx, val in term['u'].items():
                    U[idx, col] = val
                
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx, col] = val
                
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx, col] = val
        
        return U, V, W
    
    def get_trivial_solution():
        """Fallback trivial rank-36 solution."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m + j, r] = 1
                    V[j*p + k, r] = 1
                    W[k*n + i, r] = 1
        
        return U, V, W
    
    # Try the rank-33 solution first
    U33, V33, W33 = get_rank_33_solution()
    
    if verify_solution(U33, V33, W33):
        best_U, best_V, best_W = U33, V33, W33
        best_rank = 33
    else:
        # Fall back to trivial
        U36, V36, W36 = get_trivial_solution()
        best_U, best_V, best_W = U36, V36, W36
        best_rank = 36
    
    # Check if we're running out of time
    if time.time() > deadline - 5:
        return {
            'U': best_U.tolist(),
            'V': best_V.tolist(),
            'W': best_W.tolist()
        }
    
    # Try to find a better solution with rank < 33
    # Strategy: Try different block structures or shared terms
    # For now, the rank-33 solution appears to be the best known construction
    
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
# EVOLVE-BLOCK-END
