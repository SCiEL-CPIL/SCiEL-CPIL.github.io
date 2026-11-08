"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    def get_rank_33_solution():
        rank = 33
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        terms = []
        # Strassen terms for 2x2x2 part
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
        # Scalar terms
        terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
        terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
        terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
        terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
        
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
        
        return {'U': U, 'V': V, 'W': W}
    
    def verify_solution(sol):
        U = np.array(sol['U'])
        V = np.array(sol['V'])
        W = np.array(sol['W'])
        
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    T[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        
        return np.allclose(T, 1.0, atol=1e-9)
    
    def get_trivial_solution():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {'U': U, 'V': V, 'W': W}
    
    best_solution = get_rank_33_solution()
    best_rank = 33
    
    if not verify_solution(best_solution):
        best_solution = get_trivial_solution()
        best_rank = n*m*p
    
    # Try to find a lower rank solution by checking for linear dependencies
    # The 33 terms might have a linear dependency that allows rank reduction
    U33, V33, W33 = best_solution['U'], best_solution['V'], best_solution['W']
    
    # Build the tensor as a vector and check if the 33 rank-1 tensors are linearly independent
    # If they are not, we might be able to reduce the rank
    
    # For each term r, the rank-1 tensor is T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Flatten this to a vector of length n*m*p = 36
    
    rank1_tensors = []
    for r in range(33):
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] = U33[i*m+j, r] * V33[j*p+k, r] * W33[k*n+i, r]
        rank1_tensors.append(tensor.flatten())
    
    rank1_matrix = np.array(rank1_tensors).T  # 36 x 33
    
    # Check the rank of this matrix
    # If rank < 33, there's a linear dependency
    try:
        _, singular_values, _ = np.linalg.svd(rank1_matrix, full_matrices=False)
        rank_of_terms = np.sum(singular_values > 1e-10)
        
        if rank_of_terms < 33:
            # There's a linear dependency. Try to find it.
            # The null space of rank1_matrix.T gives the dependency
            _, _, Vt = np.linalg.svd(rank1_matrix, full_matrices=True)
            null_space = Vt[rank_of_terms:, :]
            
            # Try to use the dependency to reduce rank
            # This is complex; for now, just note it
            pass
    except:
        pass
    
    # Alternative approach: try random perturbations to find a lower rank solution
    # This is unlikely to work well for exact solutions, but worth trying
    
    # For now, return the best solution we have
    return {
        'U': best_solution['U'].tolist(),
        'V': best_solution['V'].tolist(),
        'W': best_solution['W'].tolist()
    }
# EVOLVE-BLOCK-END
