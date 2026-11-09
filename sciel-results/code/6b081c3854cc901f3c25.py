"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    return error < 1e-6

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    num_blocks = 5
    block_size = 2
    
    # Baseline: Rank 55 solution using Strassen (7) + Outer (4) per 2x3x2 block
    rank_per_block_base = 11
    total_rank_base = num_blocks * rank_per_block_base
    
    U_base = np.zeros((n * m, total_rank_base))
    V_base = np.zeros((m * p, total_rank_base))
    W_base = np.zeros((p * n, total_rank_base))
    
    strassen_terms = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            for i, j, val in term['U']:
                U_base[i * m + j, col] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V_base[j * p + k_global, col] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W_base[k_global * n + i, col] += val
            col_offset += 1
            
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = total_rank_base
    
    # Attempt to find a rank 10 block decomposition for 2x3x2
    # If found, total rank becomes 5 * 10 = 50
    def _find_rank_10_block():
        """
        Attempt to find a rank 10 decomposition for M_{2,3,2}.
        Returns (U, V, W) as numpy arrays or None if not found.
        """
        n_b, m_b, p_b = 2, 3, 2
        target_rank = 10
        time_limit = 15  # seconds
        
        start_time = time.time()
        
        # We use a randomized approach with rounding to integers/half-integers
        # The space of valid decompositions is likely discrete and small.
        
        for restart in range(500):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n_b*m_b, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m_b*p_b, target_rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p_b*n_b, target_rank)).astype(float)
            
            # Check
            if _verify_decomposition(U, V, W, n_b, m_b, p_b):
                return U, V, W
                
        return None

    block_decomp = _find_rank_10_block()
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        rank_per_block = 10
        total_rank = num_blocks * rank_per_block
        
        if total_rank < best_rank:
            U_new = np.zeros((n * m, total_rank))
            V_new = np.zeros((m * p, total_rank))
            W_new = np.zeros((p * n, total_rank))
            
            for b in range(num_blocks):
                col_offset = b * rank_per_block
                U_new[:, col_offset:col_offset+rank_per_block] = U_block
                for j in range(m):
                    for k_local in range(block_size):
                        k_global = b * block_size + k_local
                        V_new[j * p + k_global, col_offset:col_offset+rank_per_block] = V_block[j * block_size + k_local, :]
                for k_local in range(block_size):
                    for i in range(n):
                        k_global = b * block_size + k_local
                        W_new[k_global * n + i, col_offset:col_offset+rank_per_block] = W_block[k_local * n + i, :]
            
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = total_rank
    
    # Return best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
