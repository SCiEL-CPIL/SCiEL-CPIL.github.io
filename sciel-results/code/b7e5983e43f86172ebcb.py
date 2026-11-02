import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We try to solve the sub-problem for 2 x 3 * 3 x 3
    # n_sub=2, m_sub=3, p_sub=3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Target tensor T for sub-problem
    # U dim: n_sub * m_sub = 6
    # V dim: m_sub * p_sub = 9
    # W dim: p_sub * n_sub = 6
    dim_u = n_sub * m_sub
    dim_v = m_sub * p_sub
    dim_w = p_sub * n_sub
    
    # Create target tensor T (flattened or as indices)
    # T[u, v, w] = 1 if indices match
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    target_indices = []
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                u = i * m_sub + j
                v = j * p_sub + k
                w = k * n_sub + i
                target_indices.append((u, v, w))
    
    # Function to check if a decomposition is valid
    def check_decomposition(U, V, W, rank):
        # Reconstruct T
        T = np.zeros((dim_u, dim_v, dim_w))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Check against target
        for u, v, w in target_indices:
            if abs(T[u, v, w] - 1.0) > 1e-9:
                return False
        # Check other entries are 0
        # Iterate all u,v,w
        for u in range(dim_u):
            for v in range(dim_v):
                for w in range(dim_w):
                    if (u, v, w) not in target_indices:
                        if abs(T[u, v, w]) > 1e-9:
                            return False
        return True

    # Try to find a low rank decomposition using ALS
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s for final construction and safety
    
    # Try ranks from 12 upwards
    for rank in range(12, 20):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: Fix V, W. Solve for U.
                # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # For each u, we have a vector t_u of length dim_v * dim_w
                # t_u[v, w] = sum_r U[u, r] (V[v, r] W[w, r])
                # This is a linear system: U[u, :] * (V * W)^T = t_u
                # Where V * W is an outer product matrix of size dim_v x dim_w for each r?
                # Actually, let M_r = V[:, r] * W[:, r].T (dim_v x dim_w)
                # Then sum_r U[u, r] M_r = T_u
                # We can stack M_r into a matrix M of size (dim_v * dim_w) x rank
                # Then U[u, :] M = T_u_flat
                # U[u, :] = T_u_flat M^+ (pseudo-inverse)
                
                # Construct M
                # M is (dim_v * dim_w) x rank
                M = np.zeros((dim_v * dim_w, rank))
                for r in range(rank):
                    M[:, r] = (V[:, r] * W[:, r].T).flatten()
                
                # Target T flattened for each u
                # We only care about non-zero entries?
                # No, we must satisfy the whole tensor.
                # But the tensor is sparse.
                # We can solve using only the non-zero constraints?
                # If we use only non-zero constraints, we might get a solution that is 0 elsewhere?
                # Not necessarily. We should use all constraints or project.
                # Given the sparsity, using all constraints is safer for exactness.
                # But dim_u=6, dim_v=9, dim_w=6. 6*9*6 = 324.
                # M is 324 x rank.
                
                T_flat = np.zeros((dim_u, dim_v * dim_w))
                for u in range(dim_u):
                    for v in range(dim_v):
                        for w in range(dim_w):
                            idx = v * dim_w + w
                            if (u, v, w) in target_indices:
                                T_flat[u, idx] = 1.0
                            else:
                                T_flat[u, idx] = 0.0
                
                # Solve U
                # U @ M.T = T_flat
                # U = T_flat @ pinv(M.T)
                try:
                    U_new = T_flat @ np.linalg.pinv(M.T)
                except:
                    U_new = U
                    
                # Update V
                M_v = np.zeros((dim_u * dim_w, rank))
                for r in range(rank):
                    M_v[:, r] = (U[:, r] * W[:, r].T).flatten()
                
                T_flat_v = np.zeros((dim_v, dim_u * dim_w))
                for v in range(dim_v):
                    for u in range(dim_u):
                        for w in range(dim_w):
                            idx = u * dim_w + w
                            if (u, v, w) in target_indices:
                                T_flat_v[v, idx] = 1.0
                            else:
                                T_flat_v[v, idx] = 0.0
                try:
                    V_new = T_flat_v @ np.linalg.pinv(M_v.T)
                except:
                    V_new = V
                    
                # Update W
                M_w = np.zeros((dim_u * dim_v, rank))
                for r in range(rank):
                    M_w[:, r] = (U[:, r] * V[:, r].T).flatten()
                
                T_flat_w = np.zeros((dim_w, dim_u * dim_v))
                for w in range(dim_w):
                    for u in range(dim_u):
                        for v in range(dim_v):
                            idx = u * dim_v + v
                            if (u, v, w) in target_indices:
                                T_flat_w[w, idx] = 1.0
                            else:
                                T_flat_w[w, idx] = 0.0
                try:
                    W_new = T_flat_w @ np.linalg.pinv(M_w.T)
                except:
                    W_new = W
                    
                # Check convergence
                if np.allclose(U, U_new) and np.allclose(V, V_new) and np.allclose(W, W_new):
                    break
                    
                U, V, W = U_new, V_new, W_new
                
            # Check validity
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break # Found for this rank, try next rank or stop?
                # We want minimal rank, so we can stop searching higher ranks
                # But we are iterating rank from low to high.
                # So if found, we can break the rank loop.
                break
        
        if best_rank is not None:
            break

    # If no low rank found, use schoolbook for the sub-problem?
    # Or just return schoolbook for the whole problem.
    if best_rank is None:
        # Fall back to schoolbook for 2x3x9
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Construct the full 2x3x9 decomposition using the found 2x3x3 block
    # n=2, m=3, p=9
    # p is 3 * p_sub
    # We repeat the block 3 times.
    
    rank_full = 3 * best_rank
    U_full = [[0] * rank_full for _ in range(n * m)]
    V_full = [[0] * rank_full for _ in range(m * p)]
    W_full = [[0] * rank_full for _ in range(p * n)]
    
    for block in range(3):
        # Map indices
        # U is same for all blocks, so we just copy U_full columns
        # For block b, the columns in U_full are block*best_rank to (block+1)*best_rank - 1
        # But U is the same?
        # Wait, in the decomposition C = AB, A is shared.
        # So U_full[:, block*best_rank : (block+1)*best_rank] should be the same as best_U?
        # Yes.
        for u in range(n * m):
            for r in range(best_rank):
                U_full[u][block * best_rank + r] = best_U[u, r]
                
        for v in range(m * p_sub): # v index in sub-problem is j * p_sub + k
            # In full problem, v index is j * p + k_full
            # k_full = block * p_sub + k_sub
            # So for each j, k_sub:
            # v_full = j * p + (block * p_sub + k_sub)
            # v_sub = j * p_sub + k_sub
            for j in range(m_sub):
                for k_sub in range(p_sub):
                    v_sub = j * p_sub + k_sub
                    v_full = j * p + (block * p_sub + k_sub)
                    for r in range(best_rank):
                        V_full[v_full][block * best_rank + r] = best_V[v_sub, r]
                        
        for w in range(p_sub * n_sub): # w index in sub-problem is k * n_sub + i
            # In full problem, w index is k_full * n + i
            # k_full = block * p_sub + k_sub
            # w_full = (block * p_sub + k_sub) * n + i
            # w_sub = k_sub * n_sub + i
            for k_sub in range(p_sub):
                for i in range(n_sub):
                    w_sub = k_sub * n_sub + i
                    w_full = (block * p_sub + k_sub) * n + i
                    for r in range(best_rank):
                        W_full[w_full][block * best_rank + r] = best_W[w_sub, r]

    return {"U": U_full, "V": V_full, "W": W_full}
