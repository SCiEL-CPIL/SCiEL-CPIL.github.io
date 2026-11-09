import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def verify_block(U, V, W, n, m, p, rank):
    """
    Verify M_{n,m,p} decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Correct verification:
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k
    # We can vectorize this check.
    # For each r, we have a rank-1 tensor.
    # We sum them up.
    # Result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Efficient way:
    # Iterate over r? No, iterate over i,j,k?
    # n*m*p is small (2*4*5 = 40).
    # rank is ~40.
    # 40 * 40 = 1600 ops. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                if not np.allclose(s, 1.0):
                    return False
    return True

def find_block_decomposition(n, m, p, target_rank, time_limit=50):
    start_time = time.time()
    # Random search
    for _ in range(50000):
        if time.time() - start_time > time_limit:
            return None
            
        U = np.random.randint(0, 2, (n * m, target_rank)) * 2 - 1
        V = np.random.randint(0, 2, (m * p, target_rank)) * 2 - 1
        W = np.random.randint(0, 2, (p * n, target_rank)) * 2 - 1
        
        if verify_block(U, V, W, n, m, p, target_rank):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Try to find rank 38 decomposition
    # Strategy: Find rank 19 for M_{2,2,5} and duplicate
    # M_{2,4,5} is two copies of M_{2,2,5}
    
    # Search for M_{2,2,5} rank 19
    n_sub, m_sub, p_sub = 2, 2, 5
    target_rank_sub = 19
    
    # Use most of the time for this search
    res = find_block_decomposition(n_sub, m_sub, p_sub, target_rank_sub, time_limit=100)
    
    if res is not None:
        U_sub, V_sub, W_sub = res
        
        # Construct M_{2,4,5} from two blocks
        n_tot, m_tot, p_tot = n, m, p
        rank_tot = target_rank_sub * 2
        
        U_tot = np.zeros((n_tot * m_tot, rank_tot), dtype=int)
        V_tot = np.zeros((m_tot * p_tot, rank_tot), dtype=int)
        W_tot = np.zeros((p_tot * n_tot, rank_tot), dtype=int)
        
        # Block 1: j in {0, 1}
        for i in range(n_tot):
            for j_sub in range(m_sub):
                j = j_sub
                U_tot[i * m_tot + j, 0:target_rank_sub] = U_sub[i * m_sub + j_sub, :]
                
        # Block 2: j in {2, 3}
        for i in range(n_tot):
            for j_sub in range(m_sub):
                j = j_sub + m_sub
                U_tot[i * m_tot + j, target_rank_sub:rank_tot] = U_sub[i * m_sub + j_sub, :]
                
        # V Block 1: j in {0, 1}
        for j_sub in range(m_sub):
            j = j_sub
            for k in range(p_tot):
                V_tot[j * p_tot + k, 0:target_rank_sub] = V_sub[j_sub * p_sub + k, :]
                
        # V Block 2: j in {2, 3}
        for j_sub in range(m_sub):
            j = j_sub + m_sub
            for k in range(p_tot):
                V_tot[j * p_tot + k, target_rank_sub:rank_tot] = V_sub[j_sub * p_sub + k, :]
                
        # W Block 1 and 2
        for k in range(p_tot):
            for i in range(n_tot):
                W_tot[k * n_tot + i, 0:target_rank_sub] = W_sub[k * n_sub + i, :]
                W_tot[k * n_tot + i, target_rank_sub:rank_tot] = W_sub[k * n_sub + i, :]
        
        # Verify full solution
        if verify_block(U_tot, V_tot, W_tot, n_tot, m_tot, p_tot, rank_tot):
            return {
                "U": U_tot.tolist(),
                "V": V_tot.tolist(),
                "W": W_tot.tolist()
            }
            
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Verify schoolbook just in case
    if verify_block(U, V, W, n, m, p, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Should not happen
        raise ValueError("Schoolbook decomposition failed verification")
