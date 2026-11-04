import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _build_strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2 matrix multiplication."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'U': {(0,0):1, (1,1):1},
        'V': {(0,0):1, (1,1):1},
        'W': {(0,0):1, (1,1):1}
    })
    # M2 = (A10+A11)B00
    terms.append({
        'U': {(1,0):1, (1,1):1},
        'V': {(0,0):1},
        'W': {(1,0):1, (1,1):-1}
    })
    # M3 = A00(B01-B11)
    terms.append({
        'U': {(0,0):1},
        'V': {(0,1):1, (1,1):-1},
        'W': {(0,1):1, (1,1):1}
    })
    # M4 = A11(B10-B00)
    terms.append({
        'U': {(1,1):1},
        'V': {(1,0):1, (0,0):-1},
        'W': {(0,0):1, (1,0):1}
    })
    # M5 = (A00+A01)B11
    terms.append({
        'U': {(0,0):1, (0,1):1},
        'V': {(1,1):1},
        'W': {(0,0):-1, (0,1):1}
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'U': {(1,0):1, (0,0):-1},
        'V': {(0,0):1, (0,1):1},
        'W': {(1,1):1}
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'U': {(0,1):1, (1,1):-1},
        'V': {(1,0):1, (1,1):1},
        'W': {(0,0):1}
    })
    return terms

def _build_structured_decomposition(n, m, p, split_rows, split_cols, split_mid):
    """
    Build a structured decomposition using Strassen on 2x2 blocks.
    """
    strassen_terms = _build_strassen_2x2_terms()
    
    # Generate cumulative splits
    row_splits = [0]
    for s in split_rows:
        row_splits.append(row_splits[-1] + s)
    
    col_splits = [0]
    for s in split_cols:
        col_splits.append(col_splits[-1] + s)
    
    mid_splits = [0]
    for s in split_mid:
        mid_splits.append(mid_splits[-1] + s)
    
    U_list = []
    V_list = []
    W_list = []
    block_info = []  # Store (row_start, mid_start_a, mid_start_b, col_start) for each term
    
    for ri in range(len(split_rows)):
        for ci in range(len(split_cols)):
            for mi in range(len(split_mid)):
                for mj in range(len(split_mid)):
                    row_start = row_splits[ri]
                    row_end = row_splits[ri+1]
                    mid_start_a = mid_splits[mi]
                    mid_end_a = mid_splits[mi+1]
                    mid_start_b = mid_splits[mj]
                    mid_end_b = mid_splits[mj+1]
                    col_start = col_splits[ci]
                    col_end = col_splits[ci+1]
                    
                    row_size = row_end - row_start
                    mid_size_a = mid_end_a - mid_start_a
                    mid_size_b = mid_end_b - mid_start_b
                    col_size = col_end - col_start
                    
                    if mid_size_a != mid_size_b:
                        continue
                    
                    if row_size == 2 and mid_size_a == 2 and col_size == 2:
                        # 2x2 * 2x2 -> 2x2: Strassen, rank 7
                        for term in strassen_terms:
                            U_block = np.zeros((row_size, mid_size_a))
                            V_block = np.zeros((mid_size_b, col_size))
                            W_block = np.zeros((col_size, row_size))
                            
                            for (ri_, mi_), val in term['U'].items():
                                U_block[ri_, mi_] = val
                            for (mi_, ci_), val in term['V'].items():
                                V_block[mi_, ci_] = val
                            for (ci_, ri_), val in term['W'].items():
                                W_block[ci_, ri_] = val
                            
                            U_list.append(U_block)
                            V_list.append(V_block)
                            W_list.append(W_block)
                            block_info.append((row_start, mid_start_a, mid_start_b, col_start))
                    elif row_size == 1 and mid_size_a == 2 and col_size == 2:
                        # 1x2 * 2x2 -> 1x2: rank 4
                        for mi_ in range(2):
                            for ci_ in range(2):
                                U_block = np.zeros((1, 2))
                                V_block = np.zeros((2, 2))
                                W_block = np.zeros((2, 1))
                                U_block[0, mi_] = 1
                                V_block[mi_, ci_] = 1
                                W_block[ci_, 0] = 1
                                U_list.append(U_block)
                                V_list.append(V_block)
                                W_list.append(W_block)
                                block_info.append((row_start, mid_start_a, mid_start_b, col_start))
                    elif row_size == 2 and mid_size_a == 1 and col_size == 2:
                        # 2x1 * 1x2 -> 2x2: rank 4
                        for ri_ in range(2):
                            for ci_ in range(2):
                                U_block = np.zeros((2, 1))
                                V_block = np.zeros((1, 2))
                                W_block = np.zeros((2, 2))
                                U_block[ri_, 0] = 1
                                V_block[0, ci_] = 1
                                W_block[ci_, ri_] = 1
                                U_list.append(U_block)
                                V_list.append(V_block)
                                W_list.append(W_block)
                                block_info.append((row_start, mid_start_a, mid_start_b, col_start))
                    elif row_size == 1 and mid_size_a == 1 and col_size == 1:
                        # 1x1 * 1x1 -> 1x1: rank 1
                        U_block = np.ones((1, 1))
                        V_block = np.ones((1, 1))
                        W_block = np.ones((1, 1))
                        U_list.append(U_block)
                        V_list.append(V_block)
                        W_list.append(W_block)
                        block_info.append((row_start, mid_start_a, mid_start_b, col_start))
                    else:
                        # Fallback: schoolbook
                        for ri_ in range(row_size):
                            for mi_ in range(mid_size_a):
                                for ci_ in range(col_size):
                                    U_block = np.zeros((row_size, mid_size_a))
                                    V_block = np.zeros((mid_size_b, col_size))
                                    W_block = np.zeros((col_size, row_size))
                                    U_block[ri_, mi_] = 1
                                    V_block[mi_, ci_] = 1
                                    W_block[ci_, ri_] = 1
                                    U_list.append(U_block)
                                    V_list.append(V_block)
                                    W_list.append(W_block)
                                    block_info.append((row_start, mid_start_a, mid_start_b, col_start))
    
    if len(U_list) == 0:
        return None
    
    rank = len(U_list)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx in range(rank):
        U_block = U_list[idx]
        V_block = V_list[idx]
        W_block = W_list[idx]
        row_start, mid_start_a, mid_start_b, col_start = block_info[idx]
        
        for i in range(U_block.shape[0]):
            for j in range(U_block.shape[1]):
                U[(row_start + i) * m + (mid_start_a + j), idx] = U_block[i, j]
        
        for j in range(V_block.shape[0]):
            for k in range(V_block.shape[1]):
                V[(mid_start_b + j) * p + (col_start + k), idx] = V_block[j, k]
        
        for k in range(W_block.shape[0]):
            for i in range(W_block.shape[1]):
                W[(col_start + k) * n + (row_start + i), idx] = W_block[k, i]
    
    return U, V, W

def _try_all_block_partitions(n, m, p):
    """Try all reasonable block partitions and return the best one."""
    best_rank = float('inf')
    best_result = None
    
    row_splits_options = [[2, 1], [1, 2], [1, 1, 1]]
    col_splits_options = [[2, 2], [1, 3], [3, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1], [1, 1, 1, 1]]
    mid_splits_options = [[2, 2], [1, 3], [3, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1], [1, 1, 1, 1]]
    
    for row_split in row_splits_options:
        for col_split in col_splits_options:
            for mid_split in mid_splits_options:
                if sum(row_split) != n or sum(col_split) != p or sum(mid_split) != m:
                    continue
                
                result = _build_structured_decomposition(n, m, p, row_split, col_split, mid_split)
                if result is None:
                    continue
                
                U, V, W = result
                rank = U.shape[1]
                
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
    
    return best_result, best_rank

def _local_search_reduce_rank(U, V, W, n, m, p, max_trials=500, time_budget=30):
    """Try to reduce rank by removing terms."""
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    start_time = time.time()
    
    for trial in range(max_trials):
        if time.time() - start_time > time_budget:
            break
        
        if best_rank == 0:
            break
        
        col_to_remove = np.random.randint(0, best_rank)
        
        U_test = np.delete(best_U, col_to_remove, axis=1)
        V_test = np.delete(best_V, col_to_remove, axis=1)
        W_test = np.delete(best_W, col_to_remove, axis=1)
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank -= 1
    
    return best_U, best_V, best_W, best_rank

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    best_result, best_rank = _try_all_block_partitions(n, m, p)
    
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    
    if time.time() < deadline - 20:
        U, V, W, best_rank = _local_search_reduce_rank(U, V, W, n, m, p, 
                                                        max_trials=500, 
                                                        time_budget=30)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
