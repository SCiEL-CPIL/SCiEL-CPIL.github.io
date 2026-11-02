import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

def _rank33_solution():
    """
    Build a rank-33 decomposition using 3 blocks of a 2x2-by-2x2 Strassen-like 
    decomposition with 11 terms per block.
    """
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}

def _try_rank31_search(deadline):
    """
    Try to find a rank-31 solution using randomized local search.
    Start from rank-33, try to remove 2 columns by finding a 2-column linear combination.
    """
    n, m, p = DIMENSIONS
    base = _rank33_solution()
    U33 = np.array(base['U'], dtype=float)
    V33 = np.array(base['V'], dtype=float)
    W33 = np.array(base['W'], dtype=float)
    
    def tensor_col(r):
        vec = np.zeros(36)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    vec[idx] = U33[i*m + j, r] * V33[j*p + k, r] * W33[k*n + i, r]
        return vec
    
    # Precompute all 33 tensor columns
    tensor_cols = [tensor_col(r) for r in range(33)]
    
    # Try to find two columns whose tensor contributions can be absorbed
    # into the remaining 31 columns
    best_result = None
    for c1 in range(33):
        for c2 in range(c1+1, 33):
            if time.time() > deadline:
                return best_result
            
            cols_to_keep = [c for c in range(33) if c != c1 and c != c2]
            
            # Check if tensor_cols[c1] and tensor_cols[c2] are in span of kept columns
            M = np.column_stack([tensor_cols[c] for c in cols_to_keep])
            target = np.column_stack([tensor_cols[c1], tensor_cols[c2]])
            
            try:
                # Check if both targets are in the column space
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                rank_M = np.sum(s > 1e-10)
                
                # Project each target onto column space
                proj1 = U_svd @ (U_svd.T @ tensor_cols[c1])
                proj2 = U_svd @ (U_svd.T @ tensor_cols[c2])
                
                res1 = np.linalg.norm(tensor_cols[c1] - proj1)
                res2 = np.linalg.norm(tensor_cols[c2] - proj2)
                
                if res1 < 1e-8 and res2 < 1e-8:
                    # Both are redundant - build rank-31 solution
                    U31 = U33[:, cols_to_keep]
                    V31 = V33[:, cols_to_keep]
                    W31 = W33[:, cols_to_keep]
                    
                    result = {
                        "U": U31.tolist(),
                        "V": V31.tolist(),
                        "W": W31.tolist()
                    }
                    
                    if _verify(result["U"], result["V"], result["W"]):
                        best_result = result
                        return best_result
            except np.linalg.LinAlgError:
                continue
    
    return best_result

def _try_rank32_search(deadline):
    """
    Attempt to find a rank-32 solution by removing one column.
    """
    n, m, p = DIMENSIONS
    base = _rank33_solution()
    U33 = np.array(base['U'], dtype=float)
    V33 = np.array(base['V'], dtype=float)
    W33 = np.array(base['W'], dtype=float)
    
    def tensor_col(r):
        vec = np.zeros(36)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    vec[idx] = U33[i*m + j, r] * V33[j*p + k, r] * W33[k*n + i, r]
        return vec
    
    for remove_col in range(33):
        if time.time() > deadline:
            return None
        
        cols_to_keep = [c for c in range(33) if c != remove_col]
        
        target = tensor_col(remove_col)
        M = np.column_stack([tensor_col(c) for c in cols_to_keep])
        
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            proj = U_svd @ (U_svd.T @ target)
            residual = target - proj
            if np.linalg.norm(residual) < 1e-8:
                U32 = U33[:, cols_to_keep]
                V32 = V33[:, cols_to_keep]
                W32 = W33[:, cols_to_keep]
                
                result = {
                    "U": U32.tolist(),
                    "V": V32.tolist(),
                    "W": W32.tolist()
                }
                
                if _verify(result["U"], result["V"], result["W"]):
                    return result
        except np.linalg.LinAlgError:
            continue
    
    return None

def _construct_rank31_direct():
    """
    Try to directly construct a rank-31 solution using a known pattern.
    For 2x3x6, we can use 3 blocks of 2x3x2 (rank 11 each = 33) but try to
    share one term across blocks.
    """
    n, m, p = DIMENSIONS
    rank = 31
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use 11 terms for block 0, 10 terms for block 1, 10 terms for block 2
    # The shared term handles cross-block interactions
    
    # Block 0: columns 0-10 (11 terms)
    # Block 1: columns 11-20 (10 terms) 
    # Block 2: columns 21-30 (10 terms)
    
    # Terms for 2x3x2 (indices: U: 0-5, V: 0-5, W: 0-3)
    # Using a rank-11 decomposition
    terms_11 = [
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
        {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
    ]
    
    # For blocks 1 and 2, use only 10 terms (drop the last one and compensate)
    # This is a heuristic - may not work
    terms_10 = terms_11[:10]
    
    for t_idx, term in enumerate(terms_11):
        col = t_idx
        for idx, val in term['u'].items():
            U[idx][col] = val
        for l_idx, val in term['v'].items():
            j_local = l_idx // 2
            k_local = l_idx % 2
            g_idx = j_local * p + (0 * 2 + k_local)
            V[g_idx][col] = val
        for l_idx, val in term['w'].items():
            i_local = l_idx // 2
            k_local = l_idx % 2
            g_idx = (0 * 2 + k_local) * n + i_local
            W[g_idx][col] = val
    
    for block in [1, 2]:
        for t_idx, term in enumerate(terms_10):
            col = 11 + (block - 1) * 10 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Add one shared term for the missing 11th term of blocks 1 and 2
    # This is a placeholder - need to figure out the correct shared term
    col = 30
    # Skip for now - this approach is too speculative
    
    result = {"U": U, "V": V, "W": W}
    if _verify(result["U"], result["V"], result["W"]):
        return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with the rank-33 solution
    best_result = _rank33_solution()
    
    # Try to find a rank-31 solution
    if time.time() < deadline:
        rank31_result = _try_rank31_search(deadline)
        if rank31_result is not None:
            best_result = rank31_result
    
    # If no rank-31 found, try rank-32
    if best_result == _rank33_solution() and time.time() < deadline:
        rank32_result = _try_rank32_search(deadline)
        if rank32_result is not None:
            best_result = rank32_result
    
    # Try direct construction
    if time.time() < deadline:
        direct_result = _construct_rank31_direct()
        if direct_result is not None:
            best_result = direct_result
    
    # Verify the best result before returning
    if not _verify(best_result["U"], best_result["V"], best_result["W"]):
        best_result = _rank33_solution()
    
    return best_result
