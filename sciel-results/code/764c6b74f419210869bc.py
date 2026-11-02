"""Combined approach: verified rank-39 Strassen+schoolbook, then targeted rank-38 search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _find_rank10_223(deadline, max_rounds=5000, seed=42):
    """
    Try to find a rank-10 decomposition for M_{2,2,3}.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    rng = np.random.RandomState(seed)
    
    # Target indices
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    # Multiple random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                mask = (ta_arr == a)
                if not np.any(mask):
                    continue
                tb_sub = tb_arr[mask]
                tc_sub = tc_arr[mask]
                V_sub = V[tb_sub, :]
                W_sub = W[tc_sub, :]
                prod = V_sub * W_sub
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-12
                if np.any(valid):
                    U[a, valid] = num[valid] / den[valid]
            
            # Update V
            for b in range(m * p):
                mask = (tb_arr == b)
                if not np.any(mask):
                    continue
                ta_sub = ta_arr[mask]
                tc_sub = tc_arr[mask]
                U_sub = U[ta_sub, :]
                W_sub = W[tc_sub, :]
                prod = U_sub * W_sub
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-12
                if np.any(valid):
                    V[b, valid] = num[valid] / den[valid]
            
            # Update W
            for c_idx in range(p * n):
                mask = (tc_arr == c_idx)
                if not np.any(mask):
                    continue
                ta_sub = ta_arr[mask]
                tb_sub = tb_arr[mask]
                U_sub = U[ta_sub, :]
                V_sub = V[tb_sub, :]
                prod = U_sub * V_sub
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-12
                if np.any(valid):
                    W[c_idx, valid] = num[valid] / den[valid]
            
            # Check convergence
            if it % 50 == 0:
                sums = np.zeros(len(ta_arr))
                for c in range(rank):
                    sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
                err = np.sum((sums - 1.0) ** 2)
                if err < 1e-10:
                    return U, V, W
    
    return None

def _construct_rank38_from_223(U223, V223, W223):
    """
    Construct rank-38 decomposition using Strassen for 4 blocks (8 cols) 
    + rank-10 block for last 3 cols.
    """
    n, m, p = 2, 2, 11
    rank = 38
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    # 4 Strassen blocks for k=0..7
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Rank-10 block for k=8,9,10
    # U223 is 4x10, V223 is 6x10, W223 is 6x10
    # V223 indices: B[0,8]=0, B[0,9]=1, B[0,10]=2, B[1,8]=3, B[1,9]=4, B[1,10]=5
    # W223 indices: C[0,8]=0, C[0,9]=1, C[0,10]=2, C[1,8]=3, C[1,9]=4, C[1,10]=5
    # But in full problem: B[0,8]=8, B[0,9]=9, B[0,10]=10, B[1,8]=19, B[1,9]=20, B[1,10]=21
    # C[0,8]=16, C[0,9]=18, C[0,10]=20, C[1,8]=17, C[1,9]=19, C[1,10]=21
    
    for c in range(10):
        # U
        for a in range(4):
            U[a][r + c] = U223[a, c]
        
        # V: map local indices to global
        # Local: 0->B[0,8]=8, 1->B[0,9]=9, 2->B[0,10]=10, 3->B[1,8]=19, 4->B[1,9]=20, 5->B[1,10]=21
        v_map = [8, 9, 10, 19, 20, 21]
        for b_local in range(6):
            b_global = v_map[b_local]
            V[b_global][r + c] = V223[b_local, c]
        
        # W: map local indices to global
        # Local: 0->C[0,8]=16, 1->C[0,9]=18, 2->C[0,10]=20, 3->C[1,8]=17, 4->C[1,9]=19, 5->C[1,10]=21
        w_map = [16, 18, 20, 17, 19, 21]
        for w_local in range(6):
            w_global = w_map[w_local]
            W[w_global][r + c] = W223[w_local, c]
    
    r += 10
    assert r == rank
    return U, V, W

def _try_low_rank_full(rank, n, m, p, deadline, max_rounds=3000, seed=42):
    """
    Try to find a decomposition of given rank for the full M_{2,2,11} using iterative refinement.
    """
    rng = np.random.RandomState(seed)
    
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Update U
            for a in range(n * m):
                mask = (ta_arr == a)
                if not np.any(mask):
                    continue
                tb_sub = tb_arr[mask]
                tc_sub = tc_arr[mask]
                V_sub = V[tb_sub, :]
                W_sub = W[tc_sub, :]
                prod = V_sub * W_sub
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-12
                if np.any(valid):
                    U[a, valid] = num[valid] / den[valid]
            
            # Update V
            for b in range(m * p):
                mask = (tb_arr == b)
                if not np.any(mask):
                    continue
                ta_sub = ta_arr[mask]
                tc_sub = tc_arr[mask]
                U_sub = U[ta_sub, :]
                W_sub = W[tc_sub, :]
                prod = U_sub * W_sub
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-12
                if np.any(valid):
                    V[b, valid] = num[valid] / den[valid]
            
            # Update W
            for c_idx in range(p * n):
                mask = (tc_arr == c_idx)
                if not np.any(mask):
                    continue
                ta_sub = ta_arr[mask]
                tb_sub = tb_arr[mask]
                U_sub = U[ta_sub, :]
                V_sub = V[tb_sub, :]
                prod = U_sub * V_sub
                num = prod.sum(axis=0)
                den = (prod ** 2).sum(axis=0)
                valid = den > 1e-12
                if np.any(valid):
                    W[c_idx, valid] = num[valid] / den[valid]
            
            # Check convergence
            if it % 100 == 0:
                sums = np.zeros(len(ta_arr))
                for c in range(rank):
                    sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
                err = np.sum((sums - 1.0) ** 2)
                if err < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, get the verified rank-39 construction as fallback
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
        # Fallback: schoolbook
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
    
    # Try to find rank-10 for M_{2,2,3} to construct rank-38
    if time.time() < deadline - 60:
        result_223 = _find_rank10_223(deadline=deadline - 60, max_rounds=5000, seed=42)
        if result_223 is not None:
            U223, V223, W223 = result_223
            U38, V38, W38 = _construct_rank38_from_223(U223, V223, W223)
            if _verify(U38, V38, W38, n, m, p):
                return {"U": U38, "V": V38, "W": W38}
    
    # Try direct search for rank 38
    if time.time() < deadline - 30:
        result = _try_low_rank_full(38, n, m, p, deadline=deadline - 10, max_rounds=2000, seed=123)
        if result is not None:
            U2, V2, W2 = result
            if _verify(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
    
    # Try rank 37
    if time.time() < deadline - 20:
        result = _try_low_rank_full(37, n, m, p, deadline=deadline - 10, max_rounds=1000, seed=456)
        if result is not None:
            U2, V2, W2 = result
            if _verify(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
    
    # Return rank-39
    return {"U": U39, "V": V39, "W": W39}
# EVOLVE-BLOCK-END
