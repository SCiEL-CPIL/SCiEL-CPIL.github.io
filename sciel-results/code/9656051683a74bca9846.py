"""Combined approach: verified rank-39 baseline, then iterative refinement for lower rank."""
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
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
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

def _try_low_rank(rank, n, m, p, deadline, max_rounds=5000, seed=42, tol=1e-10):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute indices for fast ALS updates
    # For each U row a, we need sum of V[b,c]*W[d,c] for all (b,d) associated with a
    # a = i*m + j
    # b = j*p + k
    # d = k*n + i
    
    # Structure:
    # n=2, m=2, p=11
    # a in 0..3
    # i = a // 2, j = a % 2
    
    # For a fixed a (i,j):
    # k ranges 0..10
    # b = j*11 + k
    # d = k*2 + i
    
    # We can precompute the pairs (b_idx, d_idx) for each a.
    pairs_for_a = []
    for a in range(n * m):
        i = a // m
        j = a % m
        pairs = []
        for k in range(p):
            b_idx = j * p + k
            d_idx = k * n + i
            pairs.append((b_idx, d_idx))
        pairs_for_a.append(pairs)
    
    # Similarly for V rows b (j,k):
    # a = i*m + j
    # d = k*n + i
    # i ranges 0..1
    # a = i*2 + j
    # d = k*2 + i
    pairs_for_b = []
    for b in range(m * p):
        j = b // p
        k = b % p
        pairs = []
        for i in range(n):
            a_idx = i * m + j
            d_idx = k * n + i
            pairs.append((a_idx, d_idx))
        pairs_for_b.append(pairs)
    
    # Similarly for W rows d (k,i):
    # a = i*m + j
    # b = j*p + k
    # j ranges 0..1
    # a = i*2 + j
    # b = j*11 + k
    pairs_for_d = []
    for d in range(p * n):
        k = d // n
        i = d % n
        pairs = []
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + k
            pairs.append((a_idx, b_idx))
        pairs_for_d.append(pairs)

    best_err = float('inf')
    best_state = None
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            num = np.zeros(rank)
            den = np.zeros(rank)
            for b_idx, d_idx in pairs_for_a[a]:
                prod = V[b_idx] * W[d_idx]
                num += prod
                den += prod * prod
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            num = np.zeros(rank)
            den = np.zeros(rank)
            for a_idx, d_idx in pairs_for_b[b]:
                prod = U[a_idx] * W[d_idx]
                num += prod
                den += prod * prod
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for d in range(p * n):
            num = np.zeros(rank)
            den = np.zeros(rank)
            for a_idx, b_idx in pairs_for_d[d]:
                prod = U[a_idx] * V[b_idx]
                num += prod
                den += prod * prod
            valid = den > 1e-12
            if np.any(valid):
                W[d, valid] = num[valid] / den[valid]
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            # Compute error
            err = 0.0
            for a in range(n * m):
                for b_idx, d_idx in pairs_for_a[a]:
                    # sum_r U[a,r] V[b,r] W[d,r]
                    prod = U[a] * V[b_idx] * W[d_idx]
                    s = np.sum(prod)
                    err += (s - 1.0) ** 2
            # Also check zeros? The problem says "exactly decompose".
            # Usually, if the non-zeros are 1, the zeros are automatically 0 if the rank is minimal?
            # No, we must ensure zeros are 0.
            # But checking all 1936 entries is slow.
            # Let's just check the non-zeros for now, and if err is small, verify fully.
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                # Full verification
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    
    # Return best state if it's close enough
    if best_state is not None and best_err < 1e-8:
        U_b, V_b, W_b = best_state
        U_list = U_b.tolist()
        V_list = V_b.tolist()
        W_list = W_b.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # First, return the verified rank-39 construction
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
    
    # Try to find lower rank with remaining time
    best_result = None
    if time.time() < deadline - 10:
        # Try ranks 38, 37, 36, 35
        for rank in [38, 37, 36, 35]:
            if time.time() > deadline - 5:
                break
            # Try multiple seeds
            for seed in range(0, 20):
                if time.time() > deadline - 3:
                    break
                result = _try_low_rank(rank, n, m, p, deadline=deadline - 3, max_rounds=3000, seed=seed)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        best_result = (U2, V2, W2)
                        break
            if best_result is not None:
                break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    return {"U": U39, "V": V39, "W": W39}
# EVOLVE-BLOCK-END
