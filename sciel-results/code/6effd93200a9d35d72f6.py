"""Find rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        r = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def schoolbook():
        """Standard O(nmp) decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Strassen's 7 terms for 2x2x2
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (ii, jj, coeff)
    # v_entries: list of (jj, kk, coeff)
    # w_entries: list of (kk, ii, coeff)
    strassen_terms = [
        ([((0,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0), ((1,1), 1.0)]),
        ([((1,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0)],
         [((0,1), 1.0)]),
        ([((0,0), 1.0)],
         [((0,1), 1.0), ((1,1), -1.0)],
         [((1,0), 1.0), ((1,1), 1.0)]),
        ([((1,1), 1.0)],
         [((1,0), 1.0), ((0,0), -1.0)],
         [((0,0), 1.0), ((0,1), 1.0)]),
        ([((0,0), 1.0), ((0,1), 1.0)],
         [((1,1), 1.0)],
         [((1,0), 1.0), ((0,0), -1.0)]),
        ([((1,0), 1.0), ((0,0), -1.0)],
         [((0,0), 1.0), ((0,1), 1.0)],
         [((1,1), 1.0)]),
        ([((0,1), 1.0), ((1,1), -1.0)],
         [((1,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0)]),
    ]
    
    def build_decomposition(blocks):
        """Build decomposition from a list of 2x2x2 blocks.
        blocks: list of (i_off, j_off, k_off)
        Returns (U, V, W, rank) or None if invalid.
        """
        # Check all blocks are within bounds
        for i_off, j_off, k_off in blocks:
            if i_off + 1 >= n or j_off + 1 >= m or k_off + 1 >= p:
                return None
        
        num_blocks = len(blocks)
        # Compute covered triples
        covered = set()
        for i_off, j_off, k_off in blocks:
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        covered.add((i_off + ii, j_off + jj, k_off + kk))
        
        # Uncovered triples need schoolbook
        all_triples = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))
        uncovered = all_triples - covered
        rank = num_blocks * 7 + len(uncovered)
        
        if rank >= 60:
            return None
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i_off, j_off, k_off in blocks:
            for term in strassen_terms:
                u_entries, v_entries, w_entries = term
                
                for (ii, jj), coeff in u_entries:
                    i_global = i_off + ii
                    j_global = j_off + jj
                    a = i_global * m + j_global
                    U[a, col] = coeff
                
                for (jj, kk), coeff in v_entries:
                    j_global = j_off + jj
                    k_global = k_off + kk
                    b = j_global * p + k_global
                    V[b, col] = coeff
                
                for (kk, ii), coeff in w_entries:
                    k_global = k_off + kk
                    i_global = i_off + ii
                    c = k_global * n + i_global
                    W[c, col] = coeff
                
                col += 1
        
        for (i, j, k) in sorted(uncovered):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[c, col] = 1.0
            col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return U, V, W, rank
    
    def search_best_blocks(max_rank=55):
        """Search for block arrangements that give rank < max_rank."""
        # All possible 2x2x2 block positions
        all_blocks = []
        for i_off in range(n - 1):
            for j_off in range(m - 1):
                for k_off in range(p - 1):
                    all_blocks.append((i_off, j_off, k_off))
        
        best_result = None
        best_rank = max_rank
        
        # Try all subsets of blocks up to size 8
        # Start with the known good 4-block solution
        base_blocks = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
        result = build_decomposition(base_blocks)
        if result is not None:
            U, V, W, r = result
            if r < best_rank and verify(U, V, W, tol=1e-8):
                best_result = (U, V, W, r)
                best_rank = r
        
        # Try adding one more block to the base
        for extra_block in all_blocks:
            if extra_block in base_blocks:
                continue
            test_blocks = base_blocks + [extra_block]
            if len(test_blocks) > 8:
                continue
            result = build_decomposition(test_blocks)
            if result is not None:
                U, V, W, r = result
                if r < best_rank and verify(U, V, W, tol=1e-8):
                    best_result = (U, V, W, r)
                    best_rank = r
        
        # Try replacing one base block with another
        for idx in range(4):
            for replacement in all_blocks:
                if replacement in base_blocks:
                    continue
                test_blocks = base_blocks[:idx] + [replacement] + base_blocks[idx+1:]
                # Remove duplicates
                seen = set()
                unique_blocks = []
                for b in test_blocks:
                    if b not in seen:
                        seen.add(b)
                        unique_blocks.append(b)
                if len(unique_blocks) != 4:
                    continue
                result = build_decomposition(unique_blocks)
                if result is not None:
                    U, V, W, r = result
                    if r < best_rank and verify(U, V, W, tol=1e-8):
                        best_result = (U, V, W, r)
                        best_rank = r
        
        # Try 5-block configurations
        for i_off in range(n - 1):
            for j_off in range(m - 1):
                for k_off in range(p - 1):
                    new_block = (i_off, j_off, k_off)
                    if new_block in base_blocks:
                        continue
                    # Try all 5-block sets: base 4 + new
                    test_blocks = base_blocks + [new_block]
                    result = build_decomposition(test_blocks)
                    if result is not None:
                        U, V, W, r = result
                        if r < best_rank and verify(U, V, W, tol=1e-8):
                            best_result = (U, V, W, r)
                            best_rank = r
        
        return best_result
    
    # Try the known 4-block Strassen solution first
    base_blocks = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
    result = build_decomposition(base_blocks)
    
    if result is not None:
        U_str, V_str, W_str, rank_str = result
        if verify(U_str, V_str, W_str, tol=1e-8):
            best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
            best_rank = rank_str
        else:
            U_sb, V_sb, W_sb, rank_sb = schoolbook()
            best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
            best_rank = rank_sb
    else:
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
        best_rank = rank_sb
    
    # Search for better configurations
    if time.time() < deadline - 10:
        search_result = search_best_blocks(max_rank=best_rank)
        if search_result is not None:
            U_s, V_s, W_s, r_s = search_result
            if r_s < best_rank:
                best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
                best_rank = r_s
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
