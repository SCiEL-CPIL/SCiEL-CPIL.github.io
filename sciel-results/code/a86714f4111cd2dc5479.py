"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6
    total_entries = n * m * p

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all (i,j,k) entries
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Entry (i,j,k) corresponds to U[i*m+j], V[j*p+k], W[k*n+i]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def reduce_rank_by_dependencies(U, V, W, rank):
        """
        Check if the rank-1 terms are linearly dependent.
        If so, try to find a smaller set of terms that span the same tensor.
        This is a heuristic: we check the rank of the matrix of coefficients.
        If the rank of the 'coefficient matrix' (rows=entries, cols=terms) is < rank,
        we can potentially reduce the rank.
        """
        # Construct matrix M where M[entry, term] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We want to see if the columns are linearly independent.
        
        # Since rank is small (<100) and entries are 90, we can do this quickly.
        M = np.zeros((total_entries, rank))
        
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        entry_idx = (i * m + j) * p + k # Not quite, we need to map (i,j,k) to a unique index
                        # Let's use a flat index for (i,j,k)
                        # i: 0..n-1, j: 0..m-1, k: 0..p-1
                        # index = i * m * p + j * p + k
                        idx = i * m * p + j * p + k
                        M[idx, c] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        
        # Calculate rank of M
        # If rank(M) < rank, there are linear dependencies.
        # However, we need the specific tensor T (which is 1s on diagonal) to be representable.
        # The current decomposition represents T exactly.
        # If there are dependencies, we can eliminate terms.
        
        # For simplicity and speed, we only attempt reduction if rank is significantly high.
        # Gaussian elimination on M to find pivot columns.
        M_copy = M.copy()
        pivots = []
        rows = M_copy.shape[0]
        cols = M_copy.shape[1]
        
        for c in range(cols):
            # Find pivot in column c from row len(pivots) onwards
            pivot_row = -1
            for r in range(len(pivots), rows):
                if abs(M_copy[r, c]) > 1e-9:
                    pivot_row = r
                    break
            
            if pivot_row != -1:
                # Swap rows
                if pivot_row != len(pivots):
                    M_copy[len(pivots), :] = M_copy[pivot_row, :]
                    M_copy[pivot_row, :] = M_copy[len(pivots), :]
                
                # Normalize pivot row
                pivot_val = M_copy[len(pivots), c]
                if pivot_val != 0:
                    M_copy[len(pivots), :] /= pivot_val
                
                # Eliminate other rows
                for r in range(rows):
                    if r != len(pivots) and abs(M_copy[r, c]) > 1e-9:
                        factor = M_copy[r, c]
                        M_copy[r, :] -= factor * M_copy[len(pivots), :]
                
                pivots.append(c)
            
            if len(pivots) == rows:
                break
            
            if time.time() > deadline - 5:
                return None

        if len(pivots) < rank:
            # We found a smaller basis.
            # The new rank is len(pivots).
            # We need to construct the new U, V, W using only the pivot terms.
            # Note: This only works if the tensor T is in the span of the pivot columns.
            # Since the original decomposition sums to T, and T is in the span of all columns,
            # and the pivot columns form a basis for the column space, T is in the span of pivots.
            
            new_rank = len(pivots)
            U_new = [[0] * new_rank for _ in range(n * m)]
            V_new = [[0] * new_rank for _ in range(m * p)]
            W_new = [[0] * new_rank for _ in range(p * n)]
            
            for new_c, old_c in enumerate(pivots):
                for i in range(n):
                    U_new[i * m + j][new_c] = U[i * m + j][old_c] # Wait, j is not defined here
                    # Fix:
                    for j in range(m):
                        U_new[i * m + j][new_c] = U[i * m + j][old_c]
                for j in range(m):
                    for k in range(p):
                        V_new[j * p + k][new_c] = V[j * p + k][old_c]
                for k in range(p):
                    for i in range(n):
                        W_new[k * n + i][new_c] = W[k * n + i][old_c]
            
            # Verify the new decomposition
            if verify_numpy(U_new, V_new, W_new, new_rank):
                return {"U": U_new, "V": V_new, "W": W_new, "rank": new_rank}
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Define splits
    i_splits = [
        [[0, 1], [2]],
        [[0], [1, 2]],
        [[0], [1], [2]],
    ]
    j_splits = [
        [[0, 1], [2, 3], [4]],
        [[0], [1, 2], [3, 4]],
        [[0, 1], [2], [3, 4]],
        [[0], [1], [2, 3, 4]],
        [[0, 1], [2, 3, 4]],
        [[0], [1, 2, 3], [4]],
        [[0, 1, 2], [3, 4]],
        [[0], [1], [2], [3, 4]],
        [[0, 1], [2], [3], [4]],
        [[0], [1, 2], [3], [4]],
        [[0, 1], [2, 3, 4]],
        [[0], [1, 2, 3], [4]],
        [[0, 1, 2], [3], [4]],
        [[0], [1], [2, 3], [4]],
        [[0], [1], [2], [3], [4]],
    ]
    k_splits = [
        [[0, 1], [2, 3], [4, 5]],
        [[0], [1, 2], [3, 4, 5]],
        [[0, 1], [2], [3, 4, 5]],
        [[0], [1], [2, 3], [4, 5]],
        [[0, 1], [2, 3], [4], [5]],
        [[0], [1, 2], [3], [4, 5]],
        [[0, 1], [2], [3], [4, 5]],
        [[0], [1], [2], [3], [4, 5]],
        [[0, 1, 2], [3, 4, 5]],
        [[0], [1, 2, 3], [4, 5]],
        [[0], [1], [2, 3, 4], [5]],
        [[0, 1], [2, 3, 4], [5]],
        [[0], [1, 2], [3, 4], [5]],
        [[0, 1], [2], [3, 4], [5]],
        [[0], [1], [2], [3, 4], [5]],
        [[0], [1, 2], [3], [4], [5]],
        [[0, 1], [2], [3], [4], [5]],
        [[0], [1], [2], [3], [4], [5]],
        [[0, 1, 2], [3], [4], [5]],
        [[0], [1, 2, 3], [4], [5]],
        [[0], [1], [2, 3], [4], [5]],
        [[0], [1], [2], [3, 4], [5]],
        [[0, 1], [2, 3], [4], [5]],
        [[0], [1, 2], [3, 4], [5]],
    ]
    
    # Try all combinations
    for i_g in i_splits:
        for j_g in j_splits:
            for k_g in k_splits:
                if time.time() > deadline:
                    break
                
                # Calculate rank for this grouping
                total_rank = 0
                for ig in i_g:
                    for jg in j_g:
                        for kg in k_g:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                total_rank += 7
                            else:
                                total_rank += si * sj * sk
                
                if total_rank >= best_rank:
                    continue
                
                # Build the decomposition
                result = build_decomposition(i_g, j_g, k_g)
                
                # Verify
                if verify_numpy(result["U"], result["V"], result["W"], total_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = total_rank

    # Try to reduce rank of the best result found so far using dependency check
    if time.time() < deadline - 10:
        current_rank = len(best_result["U"][0])
        reduced = reduce_rank_by_dependencies(best_result["U"], best_result["V"], best_result["W"], current_rank)
        if reduced is not None and reduced["rank"] < best_rank:
            best_result = {"U": reduced["U"], "V": reduced["V"], "W": reduced["W"]}
            best_rank = reduced["rank"]

    return best_result
# EVOLVE-BLOCK-END
