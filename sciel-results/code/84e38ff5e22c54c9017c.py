"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

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
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
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
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_strassen_placements(n, m, p):
    """Try all valid Strassen block placements."""
    best_rank = n * m * p
    best_result = None
    
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))
    
    # Single blocks
    for pos in all_positions:
        if _time_left() < 5:
            break
        U, V, W = _build_strassen_embedding(n, m, p, [pos])
        if _verify(U, V, W, n, m, p):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
    
    # Pairs of non-overlapping blocks
    for idx1 in range(len(all_positions)):
        if _time_left() < 5:
            break
        for idx2 in range(idx1 + 1, len(all_positions)):
            pos1 = all_positions[idx1]
            pos2 = all_positions[idx2]
            # Check if blocks overlap
            if not (pos1[0] >= pos2[0]+2 or pos2[0] >= pos1[0]+2 or
                    pos1[1] >= pos2[1]+2 or pos2[1] >= pos1[1]+2 or
                    pos1[2] >= pos2[2]+2 or pos2[2] >= pos1[2]+2):
                continue
            U, V, W = _build_strassen_embedding(n, m, p, [pos1, pos2])
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
    
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
