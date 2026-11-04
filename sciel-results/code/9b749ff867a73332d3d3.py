import numpy as np
import time
import random
from itertools import product

N, M, P = 3, 5, 7
TENSOR_DIM = N * M * P  # 105

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon = U[i * m + j] @ (V[j * p + k] * W[k * n + i])
                if abs(recon - 1.0) > 1e-7:
                    return False
    return True

def _schoolbook(n=N, m=M, p=P):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    n = m = p = 2
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1
    return {"U": U, "V": V, "W": W}

def _build_block_decomp(n, m, p, n_parts, m_parts, p_parts):
    assert sum(n_parts) == n
    assert sum(m_parts) == m
    assert sum(p_parts) == p
    assert len(n_parts) == len(m_parts) == len(p_parts)
    total_rank = 0
    for bi in range(len(n_parts)):
        for bj in range(len(m_parts)):
            for bk in range(len(p_parts)):
                ni, mj, pk = n_parts[bi], m_parts[bj], p_parts[bk]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    for bi in range(len(n_parts)):
        ni = n_parts[bi]
        i0 = sum(n_parts[:bi])
        for bj in range(len(m_parts)):
            mj = m_parts[bj]
            j0 = sum(m_parts[:bj])
            for bk in range(len(p_parts)):
                pk = p_parts[bk]
                k0 = sum(p_parts[:bk])
                if ni == 2 and mj == 2 and pk == 2:
                    sub = _strassen_2x2x2()
                else:
                    sub = _schoolbook(ni, mj, pk)
                sub_rank = len(sub["U"][0])
                sub_U = sub["U"]
                sub_V = sub["V"]
                sub_W = sub["W"]
                for r in range(sub_rank):
                    for i in range(ni):
                        for j in range(mj):
                            U[(i0 + i) * m + (j0 + j)][col] = sub_U[i * mj + j][r]
                    for j in range(mj):
                        for k in range(pk):
                            V[(j0 + j) * p + (k0 + k)][col] = sub_V[j * pk + k][r]
                    for k in range(pk):
                        for i in range(ni):
                            W[(k0 + k) * n + (i0 + i)][col] = sub_W[k * ni + i][r]
                    col += 1
    while col > 0:
        if all(U[i][col-1] == 0 for i in range(n*m)) and \
           all(V[i][col-1] == 0 for i in range(m*p)) and \
           all(W[i][col-1] == 0 for i in range(p*n)):
            col -= 1
        else:
            break
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    return {"U": U, "V": V, "W": W}

def _get_all_splits(dim, max_part=3):
    results = []
    def helper(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            helper(remaining - part, current)
            current.pop()
    helper(dim, [])
    return results

def _greedy_block_search(n, m, p, time_budget=15):
    start_time = time.time()
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if time.time() - start_time > time_budget:
                    return best_result, best_rank
                if len(ns) != len(ms) or len(ns) != len(ps):
                    continue
                rank = 0
                for idx in range(len(ns)):
                    ni, mj, pk = ns[idx], ms[idx], ps[idx]
                    if ni == 2 and mj == 2 and pk == 2:
                        rank += 7
                    else:
                        rank += ni * mj * pk
                if rank < best_rank:
                    try:
                        result = _build_block_decomp(n, m, p, list(ns), list(ms), list(ps))
                        actual_rank = len(result["U"][0])
                        if actual_rank == rank and _verify(result["U"], result["V"], result["W"], n, m, p):
                            best_rank = rank
                            best_result = result
                    except:
                        pass
    return best_result, best_rank

def _structured_lift(n, m, p, pattern_seed, n_lifts=3, m_lifts=3, p_lifts=3):
    """
    Generate structured sparse U and V with a fixed pattern, then solve for W.
    
    The key structural insight: if we fix U and V to have a specific sparse pattern
    (e.g., each column of U has non-zeros only at positions sharing the same 'lift group'),
    then the system for W becomes linear.
    
    Specifically, for each rank term r:
      T_{ijk} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    If we fix U and V, then for each (k, n+i) position in W, the constraint is:
      sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1  for all (i,j)
    
    This is a linear system in W[k*n+i, :].
    """
    rng = np.random.RandomState(pattern_seed)
    
    # Define lift groups: partition indices into groups
    # For U: group by (i mod n_lifts, j mod m_lifts)
    # For V: group by (j mod m_lifts, k mod p_lifts)
    
    n_groups = n_lifts * m_lifts
    m_groups = m_lifts * p_lifts
    
    # Assign each (i,j) to a U-group
    u_group = {}
    for i in range(n):
        for j in range(m):
            u_group[(i, j)] = (i % n_lifts) * m_lifts + (j % m_lifts)
    
    # Assign each (j,k) to a V-group
    v_group = {}
    for j in range(m):
        for k in range(p):
            v_group[(j, k)] = (j % m_lifts) * p_lifts + (k % p_lifts)
    
    # Rank: number of distinct U-groups times something... 
    # Actually, let's use a simpler approach:
    # Each rank term r corresponds to a (u_group, v_group) pair where u_group and v_group share the same j-mod component.
    
    # Simpler: just use rank = n * m * p / (n_lifts * m_lifts) approximately
    # Let's use rank = n * m * p // (n_lifts * m_lifts) + some slack
    rank = (n * m * p) // (n_lifts * m_lifts)
    rank = max(rank, n * m)  # at least n*m terms
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    
    # Assign non-zeros: for each (i,j), pick a random rank term in its group
    for i in range(n):
        for j in range(m):
            g = u_group[(i, j)]
            # Pick a random column in the "group"
            # We'll assign columns 0..rank-1, and each (i,j) gets one non-zero
            col = rng.randint(0, rank)
            U[i * m + j, col] = 1.0
    
    for j in range(m):
        for k in range(p):
            g = v_group[(j, k)]
            col = rng.randint(0, rank)
            V[j * p + k, col] = 1.0
    
    # Now solve for W
    # For each (k, i) position in W (indexed as k*n+i):
    #   sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1  for all j
    # This gives m equations per (k,i) pair, with rank unknowns.
    # If rank >= m, we can solve (least squares if rank > m).
    
    W = np.zeros((p * n, rank))
    feasible = True
    
    for k in range(p):
        for i in range(n):
            # Build the m x rank matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            A = np.zeros((m, rank))
            for j in range(m):
                for r in range(rank):
                    A[j, r] = U[i * m + j, r] * V[j * p + k, r]
            
            # Solve A @ W_col = ones(m)
            b = np.ones(m)
            try:
                W_col, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = W_col
                # Check residual
                if residuals.size > 0 and residuals[0] > 1e-6:
                    feasible = False
            except:
                feasible = False
    
    if not feasible:
        return None
    
    # Verify
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
    
    return None

def _greedy_rank1_repair(U, V, W, n, m, p, max_terms=20, time_budget=30):
    """
    Given a near-solution, try to repair it by adding rank-1 terms to cover residuals.
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute residual tensor
    residual = np.ones((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] -= U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Greedily add rank-1 terms to reduce residual
    for term in range(max_terms):
        if time.time() - start_time > time_budget:
            break
        
        # Find the largest absolute residual
        idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        i, j, k = idx
        val = residual[i, j, k]
        
        if abs(val) < 1e-8:
            break
        
        # Add a rank-1 term that covers this residual
        # u has non-zero at (i,j), v at (j,k), w at (k,i)
        # Scale: u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=val
        # But this might break other equations... 
        # Actually, adding a rank-1 term changes all T_{i'j'k'} where 
        # u[i'j'] * v[j'k'] * w[k'i'] != 0
        # With our sparse pattern, only specific entries change.
        
        # For simplicity, just note this approach is heuristic.
        # In practice, this repair rarely works perfectly.
        pass
    
    return None

def _dft_based_attempt(n, m, p):
    """
    Attempt Winograd-style DFT approach.
    
    For cyclic convolution of length L, the DFT diagonalizes the convolution tensor.
    For matrix multiplication C = A*B, we can embed into a larger cyclic convolution.
    
    Key idea: pad A to be LxL (L >= n+m-1) and B to be LxL, then C is the 
    first n rows and p columns of the cyclic convolution result.
    
    For L=8 (next power of 2 >= 3+5-1=7):
    - Embed A (3x5) into 8x8 by zero-padding
    - Embed B (5x7) into 8x8 by zero-padding  
    - Cyclic convolution of two 8x8 matrices has rank related to DFT
    - 8-point DFT has 8 eigenvalues, so the convolution tensor has rank ~8*8=64?
    
    Actually, for 1D cyclic convolution of length L, the tensor rank is L (via DFT).
    For 2D, it's L^2. So for 8x8, rank would be 64, which is worse than 99.
    
    This approach doesn't help for our dimensions. Return None.
    """
    return None

def _fixed_uv_linear_solve(n, m, p, time_budget=20):
    """
    Main new strategy: Fix U and V to structured patterns, solve for W linearly.
    Try multiple patterns and seeds.
    """
    start_time = time.time()
    best = None
    best_rank = n * m * p
    
    # Try various lift configurations
    configs = [
        (1, 1, 1),  # no lifting, schoolbook-like
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
        (3, 2, 1),
        (2, 3, 1),
        (2, 1, 3),
        (3, 3, 1),
        (3, 1, 3),
        (1, 3, 3),
        (3, 3, 3),
    ]
    
    for (nl, ml, pl) in configs:
        if time.time() - start_time > time_budget:
            break
        
        for seed in range(5):
            if time.time() - start_time > time_budget:
                    break
            
            result = _structured_lift(n, m, p, seed * 1000 + nl * 100 + ml * 10 + pl, nl, ml, pl)
            if result is not None:
                r = result["rank"]
                if r < best_rank:
                    best_rank = r
                    best = result
    
    return best, best_rank

def construct():
    n, m, p = N, M, P
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Phase 1: Fast block partition search
    block_result, block_rank = _greedy_block_search(n, m, p, time_budget=10)
    if block_rank < best_rank:
        best_rank = block_rank
        best = block_result
    
    # Phase 2: NEW STRATEGY - Fixed UV linear solve
    remaining_time = deadline - time.time()
    if remaining_time > 25:
        solve_result, solve_rank = _fixed_uv_linear_solve(n, m, p, time_budget=min(remaining_time - 20, 50))
        if solve_result is not None and solve_rank < best_rank:
            best_rank = solve_rank
            best = solve_result
    
    # Phase 3: If we have time, try to improve via additional random structured attempts
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            for (nl, ml, pl) in [(2,2,2), (3,3,3), (2,3,2), (3,2,3)]:
                result = _structured_lift(n, m, p, seed * 10000 + nl * 1000 + ml * 100 + pl, nl, ml, pl)
                if result is not None:
                    r = result["rank"]
                    if r < best_rank:
                        best_rank = r
                        best = result
    
    # Final verification
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best
