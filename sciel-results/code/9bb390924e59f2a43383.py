import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _build_block_decomposition(n, m, p):
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_block(i_off, j_off, k_off, bn, bm, bp):
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    ig = i_off + i
                    jg = j_off + j
                    kg = k_off + k
                    add_term(
                        {ig * m + jg: 1},
                        {jg * p + kg: 1},
                        {kg * n + ig: 1}
                    )
    
    def strassen_block(i_off, j_off, k_off):
        def gi(i, j): return (i_off + i) * m + (j_off + j)
        def gj(j, k): return (j_off + j) * p + (k_off + k)
        def gk(k, i): return (k_off + k) * n + (i_off + i)
        
        add_term(
            {gi(0,0): 1, gi(1,1): 1},
            {gj(0,0): 1, gj(1,1): 1},
            {gk(0,0): 1, gk(1,1): 1}
        )
        add_term(
            {gi(1,0): 1, gi(1,1): 1},
            {gj(0,0): 1},
            {gk(0,1): 1, gk(1,1): -1}
        )
        add_term(
            {gi(0,0): 1},
            {gj(0,1): 1, gj(1,1): -1},
            {gk(1,0): 1, gk(1,1): 1}
        )
        add_term(
            {gi(1,1): 1},
            {gj(1,0): 1, gj(0,0): -1},
            {gk(0,0): 1, gk(0,1): 1}
        )
        add_term(
            {gi(0,0): 1, gi(0,1): 1},
            {gj(1,1): 1},
            {gk(0,0): -1, gk(1,0): 1}
        )
        add_term(
            {gi(1,0): 1, gi(0,0): -1},
            {gj(0,0): 1, gj(0,1): 1},
            {gk(1,1): 1}
        )
        add_term(
            {gi(0,1): 1, gi(1,1): -1},
            {gj(1,0): 1, gj(1,1): 1},
            {gk(0,0): 1}
        )
    
    i_parts = [(0, 2), (2, 2)]
    j_parts = [(0, 2), (2, 2), (4, 1)]
    k_parts = [(0, 2), (2, 2), (4, 1)]
    
    for i_off, i_sz in i_parts:
        for j_off, j_sz in j_parts:
            for k_off, k_sz in k_parts:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    strassen_block(i_off, j_off, k_off)
                else:
                    schoolbook_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding a lower-rank equivalent decomposition.
    Strategy: 
    1. Try removing individual columns (quick check)
    2. Use the structure: for each (i,j,k) constraint, the sum over c of 
       U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    3. Try to find a linear transformation that reduces effective rank
    """
    rank = U.shape[1]
    if rank < 2:
        return U, V, W
    
    # First pass: try removing each column individually
    kept = list(range(rank))
    for trial in range(rank):
        if trial not in kept:
            continue
        if len(kept) <= 1:
            break
        new_kept = [k for k in kept if k != trial]
        U_new = U[:, new_kept]
        V_new = V[:, new_kept]
        W_new = W[:, new_kept]
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            kept = new_kept
    
    if len(kept) < rank:
        U, V, W = U[:, kept], V[:, kept], W[:, kept]
        rank = len(kept)
    
    # Second pass: try to find a 2-column merge
    # If two columns c1, c2 can be combined into one column, we reduce rank by 1
    # This happens if there exists a scalar alpha such that replacing
    # U[:,c1] -> U[:,c1] + alpha*U[:,c2], etc. still works
    # But this is complex. Instead, try a different approach:
    # For small rank reductions, try all pairs of columns to see if they're "compatible"
    
    if time.time() > deadline:
        return U, V, W
    
    # Try to find if any two columns can be merged
    # Two columns c1, c2 can potentially be merged if the "cross terms" 
    # cancel out appropriately. This is hard to check directly.
    
    # Alternative: use the fact that we have a valid decomposition and try
    # to find a lower-rank one by solving a linear system.
    # 
    # Key insight: If we have a rank-r decomposition, we can check if rank r-1
    # is possible by seeing if the tensor lies in the variety of rank r-1 tensors.
    # For small r, we can try a direct approach.
    
    # Practical approach: try all subsets of size rank-1 and check if they span
    # the same tensor. This is expensive but rank is small (~92).
    
    # Better: use the structure of the problem. The matrix multiplication tensor
    # T[i,j,k] = delta(i,j,k) has a specific structure. 
    # 
    # Let's try a different strategy: use the known algebraic structure.
    # For 4x5x5, we can try to find if there's a decomposition with rank < current.
    # 
    # Since the search space is large, let's use a heuristic:
    # Try to identify if any schoolbook term can be "absorbed" into nearby 
    # Strassen terms by adjusting coefficients.
    
    return U, V, W

def _try_algebraic_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank using the algebraic structure of matrix multiplication.
    
    Key idea: The matrix multiplication tensor satisfies certain identities.
    If we can find a linear dependency among the rank-1 terms, we can reduce rank.
    
    For this specific problem, we'll try to use the fact that the tensor is
    structured and look for patterns in the U, V, W matrices.
    """
    rank = U.shape[1]
    if rank < 2:
        return U, V, W
    
    # Check for duplicate or proportional columns
    # If two columns are proportional in all three matrices, we can merge them
    merged_indices = list(range(rank))
    i = 0
    while i < len(merged_indices):
        j = i + 1
        while j < len(merged_indices):
            ci = merged_indices[i]
            cj = merged_indices[j]
            # Check if column cj is a scalar multiple of column ci in all three
            U_i = U[:, ci]
            U_j = U[:, cj]
            V_i = V[:, ci]
            V_j = V[:, cj]
            W_i = W[:, ci]
            W_j = W[:, cj]
            
            # Find alpha such that U_j ≈ alpha * U_i
            nonzero_i = np.nonzero(U_i)[0]
            if len(nonzero_i) > 0:
                alpha_candidates = []
                for idx in nonzero_i:
                    if U_i[idx] != 0:
                        alpha_candidates.append(U_j[idx] / U_i[idx])
                
                if alpha_candidates:
                    alpha = alpha_candidates[0]
                    # Check consistency
                    consistent = True
                    for idx in nonzero_i:
                        if abs(U_j[idx] - alpha * U_i[idx]) > 1e-9:
                            consistent = False
                            break
                    if consistent:
                        # Check V and W
                        nonzero_v = np.nonzero(V_i)[0]
                        v_ok = True
                        for idx in nonzero_v:
                            if abs(V_j[idx] - alpha * V_i[idx]) > 1e-9:
                                v_ok = False
                                break
                        nonzero_w = np.nonzero(W_i)[0]
                        w_ok = True
                        for idx in nonzero_w:
                            if abs(W_j[idx] - alpha * W_i[idx]) > 1e-9:
                                w_ok = False
                                break
                        
                        if v_ok and w_ok:
                            # Columns are proportional, remove cj
                            merged_indices = [k for k in merged_indices if k != cj]
                            j = i + 1
                            continue
            j += 1
        i += 1
    
    if len(merged_indices) < rank:
        U = U[:, merged_indices]
        V = V[:, merged_indices]
        W = W[:, merged_indices]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    terms = _build_block_decomposition(n, m, p)
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
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
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    if time.time() < deadline - 10:
        U, V, W = _try_algebraic_reduction(U, V, W, n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
