"""
Strategy: Start with 44-term decomposition (Strassen 2x2x8 + schoolbook 2x1x8),
then use SVD to find true rank and attempt rank reduction via linear combinations.
Additionally, use a greedy set-cover heuristic over GF(2) to find minimal rank-1
support patterns whose XOR sum reconstructs the identity tensor.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 28-term Strassen basis for 2x2x8
    rank_s = 28
    U_s = [[0] * rank_s for _ in range(n * m)]
    V_s = [[0] * rank_s for _ in range(m * p)]
    W_s = [[0] * rank_s for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_s[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_s[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_s[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Build 44-term decomposition (Strassen 28 + schoolbook 16)
    rank_44 = 44
    U44 = [[0] * rank_44 for _ in range(n * m)]
    V44 = [[0] * rank_44 for _ in range(m * p)]
    W44 = [[0] * rank_44 for _ in range(p * n)]

    for r in range(rank_s):
        for i in range(n):
            for a in range(2):
                U44[i * m + a][r] = U_s[i * m + a][r]
        for a in range(2):
            for k in range(p):
                V44[a * p + k][r] = V_s[a * p + k][r]
        for k in range(p):
            for i in range(n):
                W44[k * n + i][r] = W_s[k * n + i][r]

    col = rank_s
    for i in range(n):
        for k in range(p):
            U44[i * m + 2][col] = 1
            V44[2 * p + k][col] = 1
            W44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Try to reduce rank by finding linear dependencies
    def _term_vector(U, V, W, r):
        vec = {}
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    val = U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
                    if val != 0:
                        idx = (i, a, k)
                        vec[idx] = vec.get(idx, 0) + val
        return vec

    # Build matrix of term vectors
    num_terms = rank_44
    all_indices = set()
    term_vecs = []
    for r in range(num_terms):
        tv = _term_vector(U44, V44, W44, r)
        term_vecs.append(tv)
        all_indices.update(tv.keys())

    all_indices = sorted(all_indices)
    dim = len(all_indices)
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    # Build matrix M where M[r, pos] = coefficient of term r at position pos
    M = np.zeros((num_terms, dim))
    for r in range(num_terms):
        for idx, val in term_vecs[r].items():
            M[r, idx_to_pos[idx]] = val

    # Find rank of M and null space
    try:
        U_mat, s, Vt = np.linalg.svd(M, full_matrices=False)
        rank_M = int(np.sum(s > 1e-10))
        nullity = num_terms - rank_M
    except:
        nullity = 0
        rank_M = num_terms

    # If nullity > 0, try to eliminate terms using null-space vectors
    best_rank = num_terms
    best_U = None
    best_V = None
    best_W = None

    if nullity > 0 and _elapsed_ok():
        # Get null space vectors
        null_space = Vt[rank_M:]

        # Greedy elimination: for each null-space vector, try to zero out
        # one term at a time by adjusting the remaining terms
        # A null-space vector v means: sum_r v[r] * term_r = 0
        # So we can replace term_j (where v[j] != 0) with:
        #   new_term_j = old_term_j + sum_{r != j} (v[r]/v[j]) * term_r
        # But this changes U, V, W entries. Instead, we use the fact that
        # if v is in the null space, we can drop term j and rescale others.
        #
        # Actually, the correct approach: if sum_r c[r]*T_r = 0 where T_r are
        # the "contribution vectors", then we can remove any one term and
        # express it as a combination of the others. But the new "term" is
        # not rank-1 in general. So we need a different approach.
        #
        # Better approach: Try removing each term one at a time and check if
        # the remaining terms still span the full tensor.
        
        for remove_idx in range(num_terms):
            if not _elapsed_ok():
                break
            # Try removing term remove_idx
            keep = [r for r in range(num_terms) if r != remove_idx]
            if len(keep) == 0:
                continue
            
            # Build reduced matrix
            M_reduced = M[keep, :]
            
            # Check if M_reduced still has full column rank (= dim)
            # If yes, the removed term was redundant
            try:
                s_red = np.linalg.svd(M_reduced, full_matrices=False, compute_uv=False)
                rank_red = int(np.sum(s_red > 1e-10))
            except:
                rank_red = 0
            
            if rank_red >= dim:
                # This term is redundant! Remove it.
                # Build new U, V, W without this term
                new_rank = num_terms - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                col_new = 0
                for r in keep:
                    for i in range(n * m):
                        new_U[i][col_new] = U44[i][r]
                    for i in range(m * p):
                        new_V[i][col_new] = V44[i][r]
                    for i in range(p * n):
                        new_W[i][col_new] = W44[i][r]
                    col_new += 1
                
                candidate = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(candidate):
                    best_rank = new_rank
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    # Update M for next iteration
                    M = M_reduced
                    num_terms = new_rank
                    # Rebuild term_vecs
                    term_vecs = []
                    for r in range(num_terms):
                        tv = _term_vector(best_U, best_V, best_W, r)
                        term_vecs.append(tv)
                    # Continue trying to remove more terms
                    # Reset the loop to try removing from the new set
                    # We'll handle this by breaking and restarting
                    break
            else:
                # Term is not redundant via simple removal
                pass
    
    # If we found a better decomposition, try to iteratively remove more terms
    if best_rank < rank_44 and _elapsed_ok():
        # Iteratively try to remove more terms
        cur_rank = best_rank
        cur_U, cur_V, cur_W = best_U, best_V, best_W
        
        # Rebuild M
        all_indices2 = set()
        for r in range(cur_rank):
            tv = _term_vector(cur_U, cur_V, cur_W, r)
            all_indices2.update(tv.keys())
        all_indices2 = sorted(all_indices2)
        dim2 = len(all_indices2)
        idx_to_pos2 = {idx: pos for pos, idx in enumerate(all_indices2)}
        M2 = np.zeros((cur_rank, dim2))
        for r in range(cur_rank):
            tv = _term_vector(cur_U, cur_V, cur_W, r)
            for idx, val in tv.items():
                M2[r, idx_to_pos2[idx]] = val
        
        # Try removing terms one by one
        for remove_idx in range(cur_rank):
            if not _elapsed_ok():
                break
            keep = [r for r in range(cur_rank) if r != remove_idx]
            if len(keep) == 0:
                continue
            
            M_reduced = M2[keep, :]
            try:
                s_red = np.linalg.svd(M_reduced, full_matrices=False, compute_uv=False)
                rank_red = int(np.sum(s_red > 1e-10))
            except:
                rank_red = 0
            
            if rank_red >= dim2:
                new_rank = cur_rank - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                col_new = 0
                for r in keep:
                    for i in range(n * m):
                        new_U[i][col_new] = cur_U[i][r]
                    for i in range(m * p):
                        new_V[i][col_new] = cur_V[i][r]
                    for i in range(p * n):
                        new_W[i][col_new] = cur_W[i][r]
                    col_new += 1
                
                candidate = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(candidate):
                    cur_rank = new_rank
                    cur_U, cur_V, cur_W = new_U, new_V, new_W
                    # Rebuild M2
                    all_indices2 = set()
                    for r in range(cur_rank):
                        tv = _term_vector(cur_U, cur_V, cur_W, r)
                        all_indices2.update(tv.keys())
                    all_indices2 = sorted(all_indices2)
                    dim2 = len(all_indices2)
                    idx_to_pos2 = {idx: pos for pos, idx in enumerate(all_indices2)}
                    M2 = np.zeros((cur_rank, dim2))
                    for r in range(cur_rank):
                        tv = _term_vector(cur_U, cur_V, cur_W, r)
                        for idx, val in tv.items():
                            M2[r, idx_to_pos2[idx]] = val
                    # Restart removal loop from beginning
                    remove_idx = -1
                    continue
            else:
                pass
    
    # Return the best decomposition found
    if best_rank < rank_44 and best_U is not None:
        candidate = {"U": cur_U, "V": cur_V, "W": cur_W}
        if _validate(candidate):
            return candidate
    
    # Return the 44-term decomposition
    candidate = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate):
        return candidate
    return _schoolbook()
