import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if len(U) == 0:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _verify_full(U, V, W)

def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # result[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    result = np.einsum('uc,vc,wc->uvw', U_arr, V_arr, W_arr)
    
    if np.max(np.abs(result - T)) > 1e-9:
        return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy broadcasting."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    
    Ue = U_arr.reshape(n, m, 1, r)
    Ve = V_arr.reshape(1, m, p, r)
    We = W_arr.T.reshape(n, 1, p, r)
    
    T = np.sum(Ue * Ve * We, axis=3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _block_partition_search():
    """
    Block partition with Strassen for 2x2 blocks.
    A is 3x4: top 2 rows use Strassen, bottom row uses direct.
    Total rank: 4*7 + 4*4 = 44
    """
    n, m, p = DIMENSIONS
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        def A_idx(r, c):
            return r * m + c
        def B_idx(r, c):
            return r * p + c
        def C_idx(r, c):
            return r * n + c
        
        # M1 = (P00+P11)(Q00+Q11)
        U[A_idx(a_r0, a_c0)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0, b_c0)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M2 = (P10+P11)Q00
        U[A_idx(a_r0+1, a_c0)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0, b_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] -= 1
        col += 1
        
        # M3 = P00(Q01-Q11)
        U[A_idx(a_r0, a_c0)][col] += 1
        V[B_idx(b_r0, b_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] -= 1
        W[C_idx(c_r0, c_c0+1)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M4 = P11(Q10-Q00)
        U[A_idx(a_r0+1, a_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0)][col] += 1
        V[B_idx(b_r0, b_c0)][col] -= 1
        W[C_idx(c_r0, c_c0)][col] += 1
        W[C_idx(c_r0+1, c_c0)][col] += 1
        col += 1
        
        # M5 = (P00+P01)Q11
        U[A_idx(a_r0, a_c0)][col] += 1
        U[A_idx(a_r0, a_c0+1)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] -= 1
        W[C_idx(c_r0, c_c0+1)][col] += 1
        col += 1
        
        # M6 = (P10-P00)(Q00+Q01)
        U[A_idx(a_r0+1, a_c0)][col] += 1
        U[A_idx(a_r0, a_c0)][col] -= 1
        V[B_idx(b_r0, b_c0)][col] += 1
        V[B_idx(b_r0, b_c0+1)][col] += 1
        W[C_idx(c_r0+1, c_c0+1)][col] += 1
        col += 1
        
        # M7 = (P01-P11)(Q10+Q11)
        U[A_idx(a_r0, a_c0+1)][col] += 1
        U[A_idx(a_r0+1, a_c0+1)][col] -= 1
        V[B_idx(b_r0+1, b_c0)][col] += 1
        V[B_idx(b_r0+1, b_c0+1)][col] += 1
        W[C_idx(c_r0, c_c0)][col] += 1
        col += 1
    
    def add_direct_1x2(a_r, a_c0, b_r0, b_c0, c_r, c_c0):
        nonlocal col
        U[a_r * m + a_c0][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_r * n + c_c0][col] += 1
        col += 1
        
        U[a_r * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        W[c_r * n + c_c0][col] += 1
        col += 1
        
        U[a_r * m + a_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        W[c_r * n + (c_c0+1)][col] += 1
        col += 1
        
        U[a_r * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        W[c_r * n + (c_c0+1)][col] += 1
        col += 1
    
    # Four 2x2*2x2 Strassen products for top 2 rows
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    
    # Four 1x2*2x2 direct products for bottom row
    add_direct_1x2(2, 0, 0, 0, 2, 0)
    add_direct_1x2(2, 2, 2, 0, 2, 0)
    add_direct_1x2(2, 0, 0, 2, 2, 2)
    add_direct_1x2(2, 2, 2, 2, 2, 2)
    
    return U, V, W, col

def _try_reduce_rank(U, V, W, deadline):
    """Try to reduce rank by finding linear dependencies among rank-1 terms."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    def find_proportional_pair(U1, V1, W1):
        r = U1.shape[1]
        for i in range(r):
            if time.time() > deadline - 10:
                return None
            for j in range(i + 1, r):
                u_i = U1[:, i]
                u_j = U1[:, j]
                if np.max(np.abs(u_i)) < 1e-12 or np.max(np.abs(u_j)) < 1e-12:
                    continue
                
                nonzero = np.abs(u_i) > 1e-12
                if np.sum(nonzero) == 0:
                    continue
                alpha = u_i[nonzero][0] / u_j[nonzero][0]
                if np.max(np.abs(u_i - alpha * u_j)) > 1e-8:
                    continue
                
                v_i = V1[:, i]
                v_j = V1[:, j]
                nonzero_v = np.abs(v_i) > 1e-12
                if np.sum(nonzero_v) == 0:
                    continue
                beta = v_i[nonzero_v][0] / v_j[nonzero_v][0]
                if np.max(np.abs(v_i - beta * v_j)) > 1e-8:
                    continue
                
                w_i = W1[:, i]
                w_j = W1[:, j]
                nonzero_w = np.abs(w_i) > 1e-12
                if np.sum(nonzero_w) == 0:
                    continue
                gamma = w_i[nonzero_w][0] / w_j[nonzero_w][0]
                if np.max(np.abs(w_i - gamma * w_j)) > 1e-8:
                    continue
                
                if abs(alpha * beta * gamma - 1) > 1e-6:
                    continue
                
                return (i, j, alpha, beta, gamma)
        return None
    
    U_new = U_arr.copy()
    V_new = V_arr.copy()
    W_new = W_arr.copy()
    kept = list(range(rank))
    
    changed = True
    while changed:
        changed = False
        pair = find_proportional_pair(
            U_new[:, kept], V_new[:, kept], W_new[:, kept])
        if pair is None:
            break
        
        i, j, alpha, beta, gamma = pair
        # j is the index in the current 'kept' list (sub-array index)
        # We need to remove the original column index corresponding to kept[j]
        original_j = kept[j]
        kept = [k for k in kept if k != original_j]
        changed = True
    
    if len(kept) < rank:
        U_final = U_new[:, kept]
        V_final = V_new[:, kept]
        W_final = W_new[:, kept]
        
        U_list = U_final.tolist()
        V_list = V_final.tolist()
        W_list = W_final.tolist()
        
        if _verify_full(U_list, V_list, W_list):
            return U_list, V_list, W_list, len(kept)
    
    return U, V, W, rank

def _try_linear_dependence_removal(U, V, W, deadline):
    """
    More aggressive rank reduction: check if any column is a linear combination
    of others in a way that preserves the tensor identity.
    
    For each column c, check if removing it can be compensated by scaling
    other columns. This is a linear algebra approach.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Try removing each column and see if the remaining can be rescaled
    # to still satisfy the tensor identity
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for c_remove in range(rank):
        if time.time() > deadline - 15:
            break
        
        # Remove column c_remove
        keep = [c for c in range(rank) if c != c_remove]
        U_sub = U_arr[:, keep]
        V_sub = V_arr[:, keep]
        W_sub = W_arr[:, keep]
        
        # Try to find scalings s_c such that the tensor identity still holds
        # This is a complex optimization; for now just try uniform scaling
        # or check if the removed column was already "absorbed"
        
        # Quick check: if the removed column is zero in all three matrices, remove it
        # Note: We must check the original array U_arr, not U_sub, because
        # U_sub has c_remove removed, so its indices are shifted/shortened.
        if np.allclose(U_arr[:, c_remove], 0) and np.allclose(V_arr[:, c_remove], 0) and np.allclose(W_arr[:, c_remove], 0):
            U_new = U_arr[:, keep]
            V_new = V_arr[:, keep]
            W_new = W_arr[:, keep]
            if _verify_full(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                best_rank = len(keep)
                best_U = U_new.tolist()
                best_V = V_new.tolist()
                best_W = W_new.tolist()
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    
    return U, V, W, rank

def _random_search_low_rank(n, m, p, max_rank, time_limit, rng):
    """Random search for low-rank decompositions with half-integer entries."""
    start_time = time.time()
    best = None
    best_rank = n * m * p + 1
    
    min_possible = max(n * p, m * p, n * m)
    
    for target_rank in range(min_possible, min(max_rank, min_possible + 15)):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(30):
            if time.time() - start_time > time_limit:
                break
            
            max_val = 2
            U_rand = [[rng.randint(-max_val, max_val) for _ in range(target_rank)] 
                      for _ in range(n * m)]
            V_rand = [[rng.randint(-max_val, max_val) for _ in range(target_rank)] 
                      for _ in range(m * p)]
            W_rand = [[rng.randint(-max_val, max_val) for _ in range(target_rank)] 
                      for _ in range(p * n)]
            
            # Scale by 0.5 for half-integer entries
            U_rand = [[x / 2.0 for x in row] for row in U_rand]
            V_rand = [[x / 2.0 for x in row] for row in V_rand]
            W_rand = [[x / 2.0 for x in row] for row in W_rand]
            
            if _verify_fast(U_rand, V_rand, W_rand, n, m, p):
                best = (U_rand, V_rand, W_rand, target_rank)
                best_rank = target_rank
                break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try block partition with Strassen (should give rank 44)
    if time.time() < deadline - 20:
        U_block, V_block, W_block, rank_block = _block_partition_search()
        if _verify_full(U_block, V_block, W_block):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = rank_block
    
    # Try to reduce rank by finding linear dependencies
    if time.time() < deadline - 10:
        U_red, V_red, W_red, rank_red = _try_reduce_rank(best_U, best_V, best_W, deadline)
        if rank_red < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = rank_red
    
    # Try more aggressive linear dependence removal
    if time.time() < deadline - 10:
        U_red2, V_red2, W_red2, rank_red2 = _try_linear_dependence_removal(best_U, best_V, best_W, deadline)
        if rank_red2 < best_rank:
            best_U, best_V, best_W = U_red2, V_red2, W_red2
            best_rank = rank_red2
    
    # Try random search for even lower ranks
    if time.time() < deadline - 5:
        rng = random.Random(42)
        random_result = _random_search_low_rank(n, m, p, best_rank, min(20, deadline - time.time() - 5), rng)
        if random_result is not None:
            U_rand, V_rand, W_rand, rank_rand = random_result
            if rank_rand < best_rank:
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = rank_rand
    
    return {"U": best_U, "V": best_V, "W": best_W}
