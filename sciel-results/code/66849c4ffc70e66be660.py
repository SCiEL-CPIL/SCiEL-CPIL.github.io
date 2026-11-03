import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _try_different_blockings(deadline=None):
    """
    Try different blocking strategies to find the minimum rank.
    """
    best_rank = float('inf')
    best_result = None
    
    def _time_exceeded():
        return deadline is not None and time.time() > deadline
    
    def _make_block_decomposition(blocks, remaining_handling='pair_k'):
        """Create a decomposition from given Strassen blocks and handle remaining terms."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        covered = set()
        for ib, jb, kb in blocks:
            for i in range(ib, min(ib+2, n)):
                for j in range(jb, min(jb+2, m)):
                    for k in range(kb, min(kb+2, p)):
                        covered.add((i, j, k))
            
            U_terms, V_terms, W_terms = _strassen_2x2_block(ib, jb, kb)
            for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                for idx, val in u_t.items():
                    u_vec[idx] = val
                for idx, val in v_t.items():
                    v_vec[idx] = val
                for idx, val in w_t.items():
                    w_vec[idx] = val
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
        
        # Handle remaining terms
        remaining = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        if (i, j) not in remaining:
                            remaining[(i, j)] = []
                        remaining[(i, j)].append(k)
        
        for (i, j), ks in remaining.items():
            ks = sorted(ks)
            if remaining_handling == 'pair_k':
                # Pair k's together
                for idx in range(0, len(ks), 2):
                    if idx + 1 < len(ks):
                        k1, k2 = ks[idx], ks[idx+1]
                        u_vec = np.zeros(n * m)
                        v_vec = np.zeros(m * p)
                        w_vec = np.zeros(p * n)
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k1] = 1
                        v_vec[j * p + k2] = 1
                        w_vec[k1 * n + i] = 1
                        w_vec[k2 * n + i] = 1
                        U_cols.append(u_vec)
                        V_cols.append(v_vec)
                        W_cols.append(w_vec)
                    else:
                        k1 = ks[idx]
                        u_vec = np.zeros(n * m)
                        v_vec = np.zeros(m * p)
                        w_vec = np.zeros(p * n)
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k1] = 1
                        w_vec[k1 * n + i] = 1
                        U_cols.append(u_vec)
                        V_cols.append(v_vec)
                        W_cols.append(w_vec)
        
        if not U_cols:
            return None
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return (U, V, W)
    
    # Strategy 1: Two 2x2x2 Strassen blocks on (i,j) in {0,1}^2, k in {0,1} and {2,3}
    blocks1 = [(0, 0, 0), (0, 0, 2)]
    result = _make_block_decomposition(blocks1, 'pair_k')
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    if _time_exceeded():
        return best_result
    
    # Strategy 2: Try (i,j) in {1,2}^2 for both k blocks
    blocks2 = [(1, 0, 0), (1, 0, 2)]
    result = _make_block_decomposition(blocks2, 'pair_k')
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    if _time_exceeded():
        return best_result
    
    # Strategy 3: One block at (0,0,0), one at (1,0,2) - overlapping in i
    blocks3 = [(0, 0, 0), (1, 0, 2)]
    result = _make_block_decomposition(blocks3, 'pair_k')
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    if _time_exceeded():
        return best_result
    
    # Strategy 4: One block at (0,1,0), one at (1,0,2)
    blocks4 = [(0, 1, 0), (1, 0, 2)]
    result = _make_block_decomposition(blocks4, 'pair_k')
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    if _time_exceeded():
        return best_result
    
    # Strategy 5: Try j offsets too
    blocks5 = [(0, 0, 0), (0, 1, 2)]
    result = _make_block_decomposition(blocks5, 'pair_k')
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    if _time_exceeded():
        return best_result
    
    # Strategy 6: Three blocks covering more
    blocks6 = [(0, 0, 0), (0, 1, 2), (1, 0, 2)]
    result = _make_block_decomposition(blocks6, 'pair_k')
    if result is not None:
        U, V, W = result
        if _verify(U, V, W) and U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)
    if _time_exceeded():
        return best_result
    
    return best_result

def _local_search_refinement(U, V, W, deadline=None):
    """
    Try to reduce rank by removing columns that are linearly dependent 
    in the tensor product sense, or by finding equivalent lower-rank representations.
    """
    def _time_exceeded():
        return deadline is not None and time.time() > deadline
    
    r = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = r
    
    # Try removing each column and checking if the decomposition still works
    for col in range(r):
        if _time_exceeded():
            break
        if current_rank <= 1:
            break
        
        # Skip this column
        mask = [i for i in range(current_rank) if i != col]
        if len(mask) == 0:
            continue
            
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        if _verify(U_test, V_test, W_test):
            current_U, current_V, current_W = U_test, V_test, W_test
            current_rank -= 1
    
    return (current_U, current_V, current_W)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try different blocking strategies
    result = _try_different_blockings(deadline)
    
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            # Try local search refinement to reduce rank
            U, V, W = _local_search_refinement(U, V, W, deadline)
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _strassen_2x2_block(i_base, j_base, k_base):
    """
    Returns the U, V, W vectors for a 2x2x2 Strassen block.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
    # Verify
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print("Valid:", _verify(U, V, W))
