"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact using vectorized NumPy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    S = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if np.max(np.abs(S - 1.0)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    Handles overlapping blocks by summing contributions.
    strassen_blocks: list of (i0, j0, k0)
    """
    # For overlapping blocks, we need to be more careful.
    # Each block contributes to its 8 (i,j,k) positions.
    # If two blocks overlap, the overlapping (i,j,k) positions get contributions from both.
    # We'll build the full tensor and verify.
    
    # Strategy: for each block, add its Strassen contribution.
    # For (i,j,k) covered by exactly one block: use Strassen.
    # For (i,j,k) covered by multiple blocks: this is tricky.
    # For now, only support non-overlapping or handle by direct construction.
    
    # Check for overlaps
    covered_count = np.zeros((n, m, p), dtype=int)
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_count[i0+di, j0+dj, k0+dk] += 1
    
    # If there are overlaps, we can't simply use the embedding approach.
    # Return None to signal this case needs special handling.
    if np.max(covered_count) > 1:
        return None, None, None
    
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    sU, sV, sW = _strassen_2x2_2()
    sU_arr = np.array(sU, dtype=np.float64)
    sV_arr = np.array(sV, dtype=np.float64)
    sW_arr = np.array(sW, dtype=np.float64)
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col+c] = sU_arr[di*2+dj, c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk), col+c] = sV_arr[dj*2+dk, c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di), col+c] = sW_arr[dk*2+di, c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _blocks_overlap(pos1, pos2):
    """Check if two Strassen blocks overlap."""
    for d in range(3):
        if abs(pos1[d] - pos2[d]) < 2:
            return True
    return False

def _try_strassen_placements(n, m, p):
    """Try all valid Strassen block placements including overlapping combinations."""
    best_rank = n * m * p
    best_result = None
    
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))
    
    def _try_set(blocks):
        nonlocal best_rank, best_result
        if _time_left() < 3:
            return
        U, V, W = _build_strassen_embedding(n, m, p, blocks)
        if U is None:
            return
        if _verify(U, V, W, n, m, p):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
    
    # Single blocks
    for pos in all_positions:
        _try_set([pos])
    
    # Pairs of non-overlapping blocks
    for idx1 in range(len(all_positions)):
        if _time_left() < 5:
            break
        for idx2 in range(idx1 + 1, len(all_positions)):
            if _blocks_overlap(all_positions[idx1], all_positions[idx2]):
                continue
            _try_set([all_positions[idx1], all_positions[idx2]])
    
    # Triples of mutually non-overlapping blocks
    if _time_left() > 15:
        for idx1 in range(len(all_positions)):
            if _time_left() < 5:
                break
            for idx2 in range(idx1 + 1, len(all_positions)):
                if _blocks_overlap(all_positions[idx1], all_positions[idx2]):
                    continue
                for idx3 in range(idx2 + 1, len(all_positions)):
                    if _blocks_overlap(all_positions[idx1], all_positions[idx3]) or \
                       _blocks_overlap(all_positions[idx2], all_positions[idx3]):
                        continue
                    _try_set([all_positions[idx1], all_positions[idx2], all_positions[idx3]])
    
    # Quadruples of mutually non-overlapping blocks
    if _time_left() > 10:
        for idx1 in range(len(all_positions)):
            if _time_left() < 5:
                break
            for idx2 in range(idx1 + 1, len(all_positions)):
                if _blocks_overlap(all_positions[idx1], all_positions[idx2]):
                    continue
                for idx3 in range(idx2 + 1, len(all_positions)):
                    if _blocks_overlap(all_positions[idx1], all_positions[idx3]) or \
                       _blocks_overlap(all_positions[idx2], all_positions[idx3]):
                        continue
                    for idx4 in range(idx3 + 1, len(all_positions)):
                        if _blocks_overlap(all_positions[idx1], all_positions[idx4]) or \
                           _blocks_overlap(all_positions[idx2], all_positions[idx4]) or \
                           _blocks_overlap(all_positions[idx3], all_positions[idx4]):
                            continue
                        _try_set([all_positions[idx1], all_positions[idx2], all_positions[idx3], all_positions[idx4]])
    
    # Try overlapping block combinations (5+ blocks) - verify each
    if _time_left() > 20:
        # Generate all subsets of size 5-8 from positions that share i0,j0
        # For each (i0,j0), try all combinations of k0 values
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                k_positions = [k for k in range(p - 1)]
                # Try all subsets of k_positions of size 5, 6, 7
                from itertools import combinations
                for size in [5, 6, 7]:
                    if _time_left() < 10:
                        break
                    for combo in combinations(k_positions, size):
                        blocks = [(i0, j0, k) for k in combo]
                        _try_set(blocks)
        
        # Try mixed: blocks at different (i0,j0) with same k range
        if _time_left() > 15:
            # (0,0) with k=0,2,4,6 and (1,1) with k=1,3,5,7 - check if valid
            test_blocks = [(0,0,0),(0,0,2),(0,0,4),(0,0,6),(1,1,1),(1,1,3),(1,1,5),(1,1,7)]
            _try_set(test_blocks)
            
            test_blocks2 = [(0,0,0),(0,0,2),(0,0,4),(0,0,6),(1,1,0),(1,1,2),(1,1,4),(1,1,6)]
            _try_set(test_blocks2)
            
            # Try all 8 blocks: 4 at (0,0) and 4 at (1,1)
            for k_offset in range(8):
                if _time_left() < 10:
                    break
                blocks = [(0,0,(k_offset+kk)%7) for kk in range(4)] + [(1,1,(k_offset+kk+1)%7) for kk in range(4)]
                # Deduplicate
                blocks = list(set(blocks))
                if len(blocks) >= 5:
                    _try_set(blocks)
    
    return best_result, best_rank

def _gradient_descent_3x3x4(max_rank=35, min_rank=18, n_restarts=5, max_steps=500, tol=1e-5):
    """
    Try to find low-rank decomposition for 3x3x4 using gradient descent.
    Returns (U, V, W) as numpy arrays or None.
    """
    n_sub, m_sub, p_sub = 3, 3, 4
    dim_u = n_sub * m_sub  # 9
    dim_v = m_sub * p_sub  # 12
    dim_w = p_sub * n_sub  # 12
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                idx_u = i * m_sub + j
                idx_v = j * p_sub + k
                idx_w = k * n_sub + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    for r_sub in range(min_rank, max_rank + 1):
        if _time_left() < 10:
            break
        
        for restart in range(n_restarts):
            if _time_left() < 5:
                break
            
            np.random.seed(42 + restart)
            U_sub = np.random.randn(dim_u, r_sub) * 0.5
            V_sub = np.random.randn(dim_v, r_sub) * 0.5
            W_sub = np.random.randn(dim_w, r_sub) * 0.5
            
            lr = 0.05
            best_err = float('inf')
            
            for step in range(max_steps):
                if _time_left() < 5:
                    break
                
                # Compute S = sum_c U[:,c] V[:,c] W[:,c]
                S = np.einsum('ak,bk,ck->abc', U_sub, V_sub, W_sub)
                R = T - S
                err = np.linalg.norm(R)
                
                if err < tol:
                    return U_sub, V_sub, W_sub, r_sub
                
                if err < best_err:
                    best_err = err
                
                # Gradients
                grad_u = -np.einsum('abc,bk,ck->ak', R, V_sub, W_sub)
                grad_v = -np.einsum('abc,ak,ck->bk', R, U_sub, W_sub)
                grad_w = -np.einsum('abc,ak,bk->ck', R, U_sub, V_sub)
                
                U_sub -= lr * grad_u
                V_sub -= lr * grad_v
                W_sub -= lr * grad_w
                
                # Decay learning rate
                if step % 100 == 99:
                    lr *= 0.7
    
    return None

def _expand_3x3x4_to_3x3x8(U_sub, V_sub, W_sub, r_sub):
    """
    Expand a 3x3x4 decomposition to 3x3x8 by duplicating for k=4..7.
    """
    n, m, p = 3, 3, 8
    p_sub = 4
    rank_full = 2 * r_sub
    
    U_full = np.zeros((n * m, rank_full))
    V_full = np.zeros((m * p, rank_full))
    W_full = np.zeros((p * n, rank_full))
    
    # First half: k=0..3
    for c in range(r_sub):
        U_full[:, c] = U_sub[:, c]
        for j in range(m):
            for k in range(p_sub):
                V_full[j * p + k, c] = V_sub[j * p_sub + k, c]
        for k in range(p_sub):
            for i in range(n):
                W_full[k * n + i, c] = W_sub[k * n_sub + i, c]
    
    # Second half: k=4..7
    for c in range(r_sub):
        c2 = c + r_sub
        U_full[:, c2] = U_sub[:, c]
        for j in range(m):
            for k in range(p_sub):
                V_full[j * p + (k + p_sub), c2] = V_sub[j * p_sub + k, c]
        for k in range(p_sub):
            for i in range(n):
                W_full[(k + p_sub) * n + i, c2] = W_sub[k * n_sub + i, c]
    
    return U_full, V_full, W_full

def _try_reduce_rank(U, V, W, n, m, p):
    """
    Try to reduce rank by checking if any rank term is redundant.
    For each column c, check if removing it still gives a valid decomposition
    by adjusting other columns. Returns reduced (U, V, W) or None.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check each column for redundancy
    for c_remove in range(rank):
        if _time_left() < 5:
            break
        # Try removing column c_remove
        U_new = np.delete(U_arr, c_remove, axis=1)
        V_new = np.delete(V_arr, c_remove, axis=1)
        W_new = np.delete(W_arr, c_remove, axis=1)
        
        # Check if the remaining terms still sum correctly
        U3 = U_new.reshape(n, m, U_new.shape[1])
        V3 = V_new.reshape(m, p, V_new.shape[1])
        W3 = W_new.reshape(p, n, W_new.shape[1])
        S = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if np.max(np.abs(S - 1.0)) < 1e-6:
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def _dft_based_search(n, m, p, max_rank=67, min_rank=55, n_restarts=10, max_steps=200):
    """
    Use DFT structure along k to guide a randomized search for low-rank decomposition.
    The tensor is linear in k, so we exploit this structure.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    # DFT matrix for k dimension
    omega = np.exp(2j * np.pi / p)
    F = np.array([[omega**(k*kk) for kk in range(p)] for k in range(p)])
    F_inv = F.conj().T / p
    
    for r in range(min_rank, max_rank + 1):
        if _time_left() < 15:
            break
        
        for restart in range(n_restarts):
            if _time_left() < 8:
                break
            
            np.random.seed(1000 + restart * 17)
            
            # Initialize with DFT-inspired structure
            # U: (n*m, r) - initialize with random
            U_sub = np.random.randn(dim_u, r) * 0.3
            V_sub = np.random.randn(dim_v, r) * 0.3
            W_sub = np.random.randn(dim_w, r) * 0.3
            
            lr = 0.02
            best_err = float('inf')
            stuck = 0
            
            for step in range(max_steps):
                if _time_left() < 5:
                    break
                
                S = np.einsum('ak,bk,ck->abc', U_sub, V_sub, W_sub)
                R = T - S
                err = np.linalg.norm(R)
                
                if err < 1e-8:
                    return U_sub, V_sub, W_sub, r
                
                if err < best_err:
                    best_err = err
                    stuck = 0
                else:
                    stuck += 1
                
                if stuck > 50:
                    break
                
                grad_u = -np.einsum('abc,bk,ck->ak', R, V_sub, W_sub)
                grad_v = -np.einsum('abc,ak,ck->bk', R, U_sub, W_sub)
                grad_w = -np.einsum('abc,ak,bk->ck', R, U_sub, V_sub)
                
                U_sub -= lr * grad_u
                V_sub -= lr * grad_v
                W_sub -= lr * grad_w
                
                if step % 50 == 49:
                    lr *= 0.8
    
    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Try Strassen embeddings
    if _time_left() > 20:
        try:
            result, rank = _try_strassen_placements(n, m, p)
            if result is not None and rank < best_rank:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
        except Exception:
            pass
    
    # Try to reduce rank of the best Strassen result
    if _time_left() > 10 and best_rank < n * m * p:
        try:
            U, V, W = best_result["U"], best_result["V"], best_result["W"]
            reduced = _try_reduce_rank(U, V, W, n, m, p)
            if reduced is not None:
                U_r, V_r, W_r = reduced
                if _verify(U_r, V_r, W_r, n, m, p):
                    new_rank = len(U_r[0])
                    if new_rank < best_rank:
                        best_result = {"U": U_r, "V": V_r, "W": W_r}
                        best_rank = new_rank
        except Exception:
            pass
    
    # Try DFT-guided gradient descent
    if _time_left() > 40:
        try:
            gd_result = _dft_based_search(n, m, p, max_rank=best_rank - 1, min_rank=max(50, best_rank - 5), n_restarts=8, max_steps=150)
            if gd_result is not None:
                U_sub, V_sub, W_sub, r_sub = gd_result
                U_list = U_sub.tolist()
                V_list = V_sub.tolist()
                W_list = W_sub.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    if r_sub < best_rank:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r_sub
        except Exception:
            pass
    
    # Try gradient descent on 3x3x4, then expand to 3x3x8
    if _time_left() > 30:
        try:
            gd_result = _gradient_descent_3x3x4(max_rank=35, min_rank=18, n_restarts=3, max_steps=300)
            if gd_result is not None:
                U_sub, V_sub, W_sub, r_sub = gd_result
                U_full, V_full, W_full = _expand_3x3x4_to_3x3x8(U_sub, V_sub, W_sub, r_sub)
                
                # Verify
                U_list = U_full.tolist()
                V_list = V_full.tolist()
                W_list = W_full.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    rank_full = 2 * r_sub
                    if rank_full < best_rank:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank_full
        except Exception:
            pass
    
    return best_result
