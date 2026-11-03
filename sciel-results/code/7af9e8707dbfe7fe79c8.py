import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for the (i,j,k) triple."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    expected = _build_tensor(n, m, p)
    return np.allclose(T, expected, atol=tol)

def _naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7) for A indices (i*2+j), V has shape (4,7) for B (j*2+k), W has shape (4,7) for C (k*2+i).
    """
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-d)(f+g), M7=(b-d)(e+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, -1, -1], # d
    ]
    V = [
        [1, 1, 0, -1, 0, 0, 1],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 1, 0],   # g
        [1, 0, -1, 0, 1, 0, 1],  # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
    ]
    return U, V, W

def _strassen_221():
    """Rank-4 decomposition for 2x2x1 matrix multiplication (C is 2x2, but we're computing A(2x2)*B(2x1) -> C(2x1)).
    Actually for 2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank = 4 (naive).
    Wait, let me reconsider. The tensor is for multiplying 2x2 by 2x1 to get 2x1.
    T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1}
    U has shape (2*2, rank) = (4, rank)
    V has shape (2*1, rank) = (2, rank)
    W has shape (1*2, rank) = (2, rank)
    Naive rank = 2*2*1 = 4. Can we do better? Probably not for this small case.
    """
    # Naive: 4 terms
    U = [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]
    V = [[1,0,1,0],
         [0,1,0,1]]
    W = [[1,0,0,0],
         [0,1,0,0]]
    # Wait, let me redo this properly.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # So we need 4 terms.
    U = [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]]
    V = [[1,0,1,0],
         [0,1,0,1]]
    W = [[1,0,0,0],
         [0,1,0,0]]
    return U, V, W

def _strassen_122():
    """Rank-4 decomposition for 1x2x2 matrix multiplication.
    A is 1x2, B is 2x2, C is 1x2. Naive rank = 1*2*2 = 4.
    """
    U = [[1,0,0,0],
         [0,1,0,0]]
    V = [[1,0,1,0],
         [0,1,0,1]]
    W = [[1,0,0,0],
         [0,1,0,0]]
    return U, W

def _strassen_212():
    """Rank-4 decomposition for 2x1x2 matrix multiplication.
    A is 2x1, B is 1x2, C is 2x2. Naive rank = 2*1*2 = 4.
    """
    U = [[1,0,0,0],
         [0,1,0,0]]
    V = [[1,0,0,0],
         [0,1,0,0]]
    W = [[1,0,0,0],
         [0,1,0,0]]
    return U, V, W

def _strassen_121():
    """Rank-2 decomposition for 1x2x1 matrix multiplication.
    A is 1x2, B is 2x1, C is 1x1. Naive rank = 1*2*1 = 2.
    """
    U = [[1,0],
         [0,1]]
    V = [[1,0],
         [0,1]]
    W = [[1,0],
         [0,1]]
    return U, V, W

def _strassen_211():
    """Rank-2 decomposition for 2x1x1 matrix multiplication.
    A is 2x1, B is 1x1, C is 2x1. Naive rank = 2*1*1 = 2.
    """
    U = [[1,0],
         [0,1]]
    V = [[1,0],
         [0,1]]
    W = [[1,0],
         [0,1]]
    return U, V, W

def _strassen_112():
    """Rank-2 decomposition for 1x1x2 matrix multiplication.
    A is 1x1, B is 1x2, C is 1x2. Naive rank = 1*1*2 = 2.
    """
    U = [[1,0],
         [0,1]]
    V = [[1,0],
         [0,1]]
    W = [[1,0],
         [0,1]]
    return U, V, W

def _strassen_111():
    """Rank-1 decomposition for 1x1x1 matrix multiplication.
    A is 1x1, B is 1x1, C is 1x1. Naive rank = 1*1*1 = 1.
    """
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return U, V, W

def _try_strassen_block_decomposition(n, m, p):
    """Try to decompose n×m×p by splitting dimensions and using Strassen on 2×2×2 blocks."""
    def split_dim(d):
        parts = []
        while d > 2:
            parts.append(2)
            d -= 2
        parts.append(d)
        return parts
    
    ni = split_dim(n)
    mj = split_dim(m)
    pk = split_dim(p)
    
    total_rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ii, ni_size in enumerate(ni):
        for jj, mj_size in enumerate(mj):
            for kk, pk_size in enumerate(pk):
                i_off = sum(ni[:ii])
                j_off = sum(mj[:jj])
                k_off = sum(pk[:kk])
                
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Use Strassen rank 7
                    U_s, V_s, W_s = _strassen_222()
                    for r in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            for b in range(2):
                                u_col[(i_off + a) * m + (j_off + b)] = U_s[a * 2 + b][r]
                        for b in range(2):
                            for c in range(2):
                                v_col[(j_off + b) * p + (k_off + c)] = V_s[b * 2 + c][r]
                        for c in range(2):
                            for a in range(2):
                                w_col[(k_off + c) * n + (i_off + a)] = W_s[c * 2 + a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 7
                elif ni_size == 2 and mj_size == 2 and pk_size == 1:
                    # 2x2x1: rank 4
                    U_s, V_s, W_s = _strassen_221()
                    for r in range(4):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            for b in range(2):
                                u_col[(i_off + a) * m + (j_off + b)] = U_s[a * 2 + b][r]
                        for b in range(2):
                            v_col[(j_off + b) * p + k_off] = V_s[b][r]
                        for a in range(2):
                            w_col[k_off * n + (i_off + a)] = W_s[a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 4
                elif ni_size == 1 and mj_size == 2 and pk_size == 2:
                    # 1x2x2: rank 4
                    U_s, V_s, W_s = _strassen_122()
                    for r in range(4):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for b in range(2):
                            u_col[i_off * m + (j_off + b)] = U_s[b][r]
                        for b in range(2):
                            for c in range(2):
                                v_col[(j_off + b) * p + (k_off + c)] = V_s[b * 2 + c][r]
                        for c in range(2):
                            w_col[(k_off + c) * n + i_off] = W_s[c][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 4
                elif ni_size == 2 and mj_size == 1 and pk_size == 2:
                    # 2x1x2: rank 4
                    U_s, V_s, W_s = _strassen_212()
                    for r in range(4):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            u_col[(i_off + a) * m + j_off] = U_s[a][r]
                        for c in range(2):
                            v_col[j_off * p + (k_off + c)] = V_s[c][r]
                        for c in range(2):
                            for a in range(2):
                                w_col[(k_off + c) * n + (i_off + a)] = W_s[c * 2 + a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 4
                elif ni_size == 1 and mj_size == 2 and pk_size == 1:
                    # 1x2x1: rank 2
                    U_s, V_s, W_s = _strassen_121()
                    for r in range(2):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for b in range(2):
                            u_col[i_off * m + (j_off + b)] = U_s[b][r]
                        v_col[(j_off + 0) * p + k_off] = V_s[0][r]
                        v_col[(j_off + 1) * p + k_off] = V_s[1][r]
                        w_col[k_off * n + i_off] = W_s[0][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 2
                elif ni_size == 2 and mj_size == 1 and pk_size == 1:
                    # 2x1x1: rank 2
                    U_s, V_s, W_s = _strassen_211()
                    for r in range(2):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            u_col[(i_off + a) * m + j_off] = U_s[a][r]
                        v_col[j_off * p + k_off] = V_s[0][r]
                        w_col[k_off * n + (i_off + a)] = W_s[a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 2
                elif ni_size == 1 and mj_size == 1 and pk_size == 2:
                    # 1x1x2: rank 2
                    U_s, V_s, W_s = _strassen_112()
                    for r in range(2):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        u_col[i_off * m + j_off] = U_s[0][r]
                        for c in range(2):
                            v_col[j_off * p + (k_off + c)] = V_s[c][r]
                        for c in range(2):
                            w_col[(k_off + c) * n + i_off] = W_s[c][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 2
                elif ni_size == 1 and mj_size == 1 and pk_size == 1:
                    # 1x1x1: rank 1
                    U_s, V_s, W_s = _strassen_111()
                    u_col = [0.0] * (n * m)
                    v_col = [0.0] * (m * p)
                    w_col = [0.0] * (p * n)
                    u_col[i_off * m + j_off] = 1.0
                    v_col[j_off * p + k_off] = 1.0
                    w_col[k_off * n + i_off] = 1.0
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    total_rank += 1
                else:
                    # Naive for this block
                    block_rank = ni_size * mj_size * pk_size
                    for a in range(ni_size):
                        for b in range(mj_size):
                            for c in range(pk_size):
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                u_col[(i_off + a) * m + (j_off + b)] = 1.0
                                v_col[(j_off + b) * p + (k_off + c)] = 1.0
                                w_col[(k_off + c) * n + (i_off + a)] = 1.0
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
                    total_rank += block_rank
    
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_block_decomposition_v2(n, m, p):
    """Improved block decomposition: split n=3 as 2+1, m=4 as 2+2, p=7 as 2+2+2+1.
    This gives blocks:
    (2,2,2): 3 blocks -> 3*7 = 21
    (2,2,1): 1 block -> 4
    (1,2,2): 3 blocks -> 3*4 = 12
    (1,2,1): 1 block -> 2
    Total: 21+4+12+2 = 39? That seems too low. Let me recalculate.
    
    Actually wait. n=3 -> [2,1], m=4 -> [2,2], p=7 -> [2,2,2,1]
    Blocks:
    (2,2,2): ii=0,jj=0,kk=0 -> 7
    (2,2,2): ii=0,jj=0,kk=1 -> 7
    (2,2,2): ii=0,jj=0,kk=2 -> 7
    (2,2,1): ii=0,jj=0,kk=3 -> 4
    (2,2,2): ii=0,jj=1,kk=0 -> 7
    (2,2,2): ii=0,jj=1,kk=1 -> 7
    (2,2,2): ii=0,jj=1,kk=2 -> 7
    (2,2,1): ii=0,jj=1,kk=3 -> 4
    (1,2,2): ii=1,jj=0,kk=0 -> 4
    (1,2,2): ii=1,jj=0,kk=1 -> 4
    (1,2,2): ii=1,jj=0,kk=2 -> 4
    (1,2,1): ii=1,jj=0,kk=3 -> 2
    (1,2,2): ii=1,jj=1,kk=0 -> 4
    (1,2,2): ii=1,jj=1,kk=1 -> 4
    (1,2,2): ii=1,jj=1,kk=2 -> 4
    (1,2,1): ii=1,jj=1,kk=3 -> 2
    
    Total: 3*7 + 2*4 + 6*4 + 2*2 = 21 + 8 + 24 + 4 = 57
    """
    # Use the same logic as _try_strassen_block_decomposition but with explicit splits
    # The function already does this, so just call it
    return _try_strassen_block_decomposition(n, m, p)

def _try_lanczos_like_search(n, m, p, target_rank, max_iter=500, seed=42):
    """Try to find a rank decomposition using iterative refinement."""
    rng = np.random.RandomState(seed)
    best_score = float('inf')
    best_result = None
    
    for it in range(max_iter):
        if time.time() > _START_TIME + TIME_LIMIT - 15:
            break
        
        # Initialize with random values
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        # Normalize
        for r in range(target_rank):
            norm = np.linalg.norm(U[:, r])
            if norm > 1e-10:
                U[:, r] /= norm
                V[:, r] *= norm
        
        # Iterative refinement
        for inner in range(20):
            if time.time() > _START_TIME + TIME_LIMIT - 15:
                break
            
            # Compute current tensor
            T = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            expected = _build_tensor(n, m, p)
            error = np.linalg.norm(T - expected)
            
            if error < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            if error < best_score:
                best_score = error
                best_result = {"U": U.copy(), "V": V.copy(), "W": W.copy()}
        
        # Try to improve by adjusting one factor at a time
        # This is a simplified approach
    
    if best_result is not None:
        U = best_result["U"]
        V = best_result["V"]
        W = best_result["W"]
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_groebner_like_search(n, m, p, target_rank, max_iter=100, seed=42):
    """Try to find a rank decomposition using a more structured approach."""
    rng = np.random.RandomState(seed)
    
    for it in range(max_iter):
        if time.time() > _START_TIME + TIME_LIMIT - 15:
            break
        
        # Create a sparse random initialization
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for r in range(target_rank):
            # Each factor has a few non-zero entries
            u_idx = rng.choice(n * m, size=min(3, n * m), replace=False)
            v_idx = rng.choice(m * p, size=min(3, m * p), replace=False)
            w_idx = rng.choice(p * n, size=min(3, p * n), replace=False)
            
            U[u_idx, r] = rng.choice([-1, 1], size=len(u_idx))
            V[v_idx, r] = rng.choice([-1, 1], size=len(v_idx))
            W[w_idx, r] = rng.choice([-1, 1], size=len(w_idx))
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + TIME_LIMIT
    
    # First verify Strassen decomposition works
    try:
        U_s, V_s, W_s = _strassen_222()
        # Quick verify 2x2x2
        T = np.zeros((4, 4, 4))
        for r in range(7):
            T += np.einsum('a,b,c->abc', np.array(U_s)[:, r], np.array(V_s)[:, r], np.array(W_s)[:, r])
        expected = _build_tensor(2, 2, 2)
        strassen_ok = np.allclose(T, expected, atol=1e-9)
    except Exception:
        strassen_ok = False
    
    best_result = None
    best_rank = n * m * p
    
    if strassen_ok:
        try:
            result = _try_strassen_block_decomposition(n, m, p)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    if time.time() > deadline - 20:
        if best_result is not None:
            return best_result
        return _naive_decomposition(n, m, p)
    
    # Try random search for lower ranks
    for target_rank in range(max(1, best_rank - 5), best_rank + 1):
        if target_rank < 1 or time.time() > deadline - 10:
            break
        result = _try_groebner_like_search(n, m, p, target_rank, max_iter=200, seed=42 + target_rank)
        if result is not None:
            return result
    
    if best_result is not None:
        return best_result
    return _naive_decomposition(n, m, p)
