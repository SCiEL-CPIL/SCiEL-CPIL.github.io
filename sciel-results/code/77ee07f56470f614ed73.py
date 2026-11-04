import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _build_block_decomposition(n, m, p):
    """
    Build a valid decomposition using block-diagonal Strassen + schoolbook.
    Returns list of (U_sparse, V_sparse, W_sparse) where each is dict {idx: val}.
    """
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
        """2x2x2 Strassen block."""
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

def _try_svd_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank using SVD on the tensor structure.
    The idea: if we can find a linear transformation T such that
    U' = U @ T, V' = V @ T^{-1}, W' = W @ T^{-1} (or similar)
    that reduces the effective rank.
    
    More practically: build the tensor and find its CP decomposition rank.
    """
    start = time.time()
    if time.time() > deadline:
        return U, V, W
    
    rank = U.shape[1]
    
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want to find a lower-rank CP decomposition of this tensor.
    # 
    # Alternative: try to find a subspace of the rank dimension where the
    # decomposition still works.
    
    # Practical approach: try removing columns one at a time and see if we
    # can "repair" the decomposition by adjusting remaining columns.
    
    # This is expensive, so let's try a simpler heuristic:
    # Check if any two columns are linearly dependent in a way that allows
    # merging.
    
    # For now, return original if no quick improvement found
    if time.time() > deadline - 5:
        return U, V, W
    
    # Try: for each pair of columns, check if their removal can be compensated
    # by adjusting other columns. This is a nonlinear problem, so use a
    # simple heuristic: try random linear combinations.
    
    # Actually, let's try a different approach: use the fact that the
    # decomposition is block-diagonal. Within each 2x2x2 Strassen block,
    # the 7 terms are minimal. But across blocks, there might be redundancy.
    
    # The key insight: if two blocks share the same (i,j,k) indices in a way
    # that allows cancellation, we can reduce rank.
    
    # For 4x5x5 with partition [2,2]x[2,2,1]x[2,2,1]:
    # - 2 Strassen blocks (2x2x2 each)
    # - Schoolbook blocks for the remaining dimensions
    # 
    # The Strassen blocks operate on disjoint index sets, so there's no
    # cross-block cancellation possible with the current partitioning.
    
    # Let's try a different partitioning that might allow more Strassen
    # blocks or better overlap.
    
    return U, V, W

def _try_alternative_partition(n, m, p, deadline):
    """
    Try different block partitions to see if we can get a lower rank.
    """
    start = time.time()
    if time.time() > deadline:
        return None
    
    best_rank = float('inf')
    best_terms = None
    
    # Try different partitions of n=4, m=5, p=5
    # n can be [4], [2,2], [1,3], [3,1], [1,1,2], [1,2,1], [2,1,1], [1,1,1,1]
    # m can be [5], [2,3], [3,2], [1,4], [4,1], [2,2,1], [2,1,2], [1,2,2], etc.
    # p can be similar to m
    
    # Focus on partitions that maximize 2x2x2 Strassen blocks
    # For 2x2x2 Strassen, we need at least 2 in each dimension
    
    # Partition n=4 as [2,2] gives 2 blocks of size 2
    # Partition m=5 as [2,2,1] gives 2 blocks of size 2
    # Partition p=5 as [2,2,1] gives 2 blocks of size 2
    # This gives 2*2*2 = 8 potential 2x2x2 blocks, but only 2 are actually
    # 2x2x2 (the first two in each dimension)
    
    # Let's try: n=[2,2], m=[2,2,1], p=[2,2,1] (current)
    # vs: n=[2,2], m=[2,1,2], p=[2,1,2]
    # vs: n=[2,2], m=[1,2,2], p=[1,2,2]
    
    def build_decomp_for_partition(i_parts, j_parts, k_parts):
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
        
        for i_off, i_sz in i_parts:
            for j_off, j_sz in j_parts:
                for k_off, k_sz in k_parts:
                    if i_sz == 2 and j_sz == 2 and k_sz == 2:
                        strassen_block(i_off, j_off, k_off)
                    else:
                        schoolbook_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
        
        return terms
    
    # Try a few promising partitions
    partitions = [
        ([(0,2), (2,2)], [(0,2), (2,2), (4,1)], [(0,2), (2,2), (4,1)]),  # Current
        ([(0,2), (2,2)], [(0,2), (2,1), (3,2)], [(0,2), (2,1), (3,2)]),
        ([(0,2), (2,2)], [(0,1), (1,2), (3,2)], [(0,1), (1,2), (3,2)]),
    ]
    
    for i_parts, j_parts, k_parts in partitions:
        if time.time() > deadline - 10:
            break
        
        terms = build_decomp_for_partition(i_parts, j_parts, k_parts)
        r = len(terms)
        
        if r < best_rank:
            best_rank = r
            best_terms = terms
    
    if best_terms is not None and best_rank < 92:
        return best_terms
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Step 1: Build block decomposition
    terms = _build_block_decomposition(n, m, p)
    original_rank = len(terms)
    
    # Step 2: Try alternative partitions
    if time.time() < deadline - 30:
        alt_terms = _try_alternative_partition(n, m, p, deadline - 20)
        if alt_terms is not None and len(alt_terms) < original_rank:
            terms = alt_terms
            original_rank = len(terms)
    
    # Step 3: Convert to matrices
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    # Step 4: Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook
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

if __name__ == "__main__":
    construct()
