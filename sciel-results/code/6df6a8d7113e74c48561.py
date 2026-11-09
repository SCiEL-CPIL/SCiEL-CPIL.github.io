import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def verify_block(U, V, W, n, m, p, rank):
    """
    Vectorized verification of M_{n,m,p} decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    # Reshape to 3D tensors
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # We need to check if sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # This is equivalent to checking the element-wise product sum.
    
    # Create a tensor of ones of shape (n, m, p)
    # The condition is:
    # For each j, k:
    #   sum_r U[:, j, :] * V[j, k, :] * W[k, :, :] == ones(n)
    
    # Let's iterate over j and k as in the original, but use numpy for the inner sum
    for j in range(m):
        for k in range(p):
            # term_r = U[:, j, :] * V[j, k, :] * W[k, :, :]
            # result_i = sum_r term_r
            term = U_r[:, j, :] * V_r[j, k, :] * W_r[k, :, :]
            result = np.sum(term, axis=1)
            if not np.allclose(result, np.ones(n)):
                return False
    return True

def find_block_decomposition(n, m, p, target_ranks, time_limit=50):
    start_time = time.time()
    
    # Try ranks in increasing order
    for target_rank in target_ranks:
        # Random search
        # Use a more diverse set of random integers
        # Values in {-1, 0, 1} are common for small ranks
        for _ in range(20000):
            if time.time() - start_time > time_limit:
                return None
            
            U = np.random.randint(-1, 2, (n * m, target_rank))
            V = np.random.randint(-1, 2, (m * p, target_rank))
            W = np.random.randint(-1, 2, (p * n, target_rank))
            
            if verify_block(U, V, W, n, m, p, target_rank):
                return U, V, W, target_rank
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    deadline = time.time() + 110
    
    # Strategy: Find rank-minimal decomposition for M_{2,2,5}
    # Known lower bound for M_{2,2,5} is 17.
    # Try ranks 17, 18, 19, 20
    n_sub, m_sub, p_sub = 2, 2, 5
    target_ranks = [17, 18, 19, 20]
    
    res = find_block_decomposition(n_sub, m_sub, p_sub, target_ranks, time_limit=60)
    
    if res is not None:
        U_sub, V_sub, W_sub, rank_sub = res
        
        # Construct M_{2,4,5} from two blocks
        n_tot, m_tot, p_tot = n, m, p
        rank_tot = rank_sub * 2
        
        U_tot = np.zeros((n_tot * m_tot, rank_tot), dtype=int)
        V_tot = np.zeros((m_tot * p_tot, rank_tot), dtype=int)
        W_tot = np.zeros((p_tot * n_tot, rank_tot), dtype=int)
        
        # Block 1: j in {0, 1}
        for i in range(n_tot):
            for j_sub in range(m_sub):
                j = j_sub
                U_tot[i * m_tot + j, 0:rank_sub] = U_sub[i * m_sub + j_sub, :]
                
        # Block 2: j in {2, 3}
        for i in range(n_tot):
            for j_sub in range(m_sub):
                j = j_sub + m_sub
                U_tot[i * m_tot + j, rank_sub:rank_tot] = U_sub[i * m_sub + j_sub, :]
                
        # V Block 1: j in {0, 1}
        for j_sub in range(m_sub):
            j = j_sub
            for k in range(p_tot):
                V_tot[j * p_tot + k, 0:rank_sub] = V_sub[j_sub * p_sub + k, :]
                
        # V Block 2: j in {2, 3}
        for j_sub in range(m_sub):
            j = j_sub + m_sub
            for k in range(p_tot):
                V_tot[j * p_tot + k, rank_sub:rank_tot] = V_sub[j_sub * p_sub + k, :]
                
        # W Block 1 and 2
        for k in range(p_tot):
            for i in range(n_tot):
                W_tot[k * n_tot + i, 0:rank_sub] = W_sub[k * p_sub + i, :]
                W_tot[k * n_tot + i, rank_sub:rank_tot] = W_sub[k * p_sub + i, :]
        
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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
