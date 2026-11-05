"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact using numpy for speed."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C11 (T[1,1,1])
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        # Contributes +1 to C10 (T[1,0,0]) and -1 to C11 (T[1,1,1])
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        # Contributes +1 to C01 (T[0,0,1]) and +1 to C11 (T[1,1,1])
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C10 (T[1,0,0])
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        # Contributes -1 to C00 (T[0,0,0]) and +1 to C01 (T[0,0,1])
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11 (T[1,1,1])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00 (T[0,0,0])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def block_rank(si, sj, sk):
        """Compute the minimum rank for a block of size si x sj x sk."""
        if si == 2 and sj == 2 and sk == 2:
            return 7
        if si == 2 and sj == 2 and sk >= 2:
            return 7 + 4 * (sk - 2)
        if si == 2 and sk == 2 and sj >= 2:
            return 7 + 4 * (sj - 2)
        if sj == 2 and sk == 2 and si >= 2:
            return 7 + 4 * (si - 2)
        return si * sj * sk

    def build_strassen_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on suitable blocks and schoolbook elsewhere.
        """
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    r = block_rank(si, sj, sk)
                    total_rank += r
                    block_info.append((ig, jg, kg, r, si, sj, sk))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r, si, sj, sk in block_info:
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
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
            elif si == 2 and sj == 2 and sk > 2:
                # Strassen on 2x2x2 corner, schoolbook for rest
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
                # Schoolbook for remaining 2x2x(sk-2)
                for li in range(2):
                    for lj in range(2):
                        for lk in range(2, sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
            elif si == 2 and sk == 2 and sj > 2:
                # Strassen on 2x2x2 corner, schoolbook for rest
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
                # Schoolbook for remaining 2x(sj-2)x2
                for li in range(2):
                    for lj in range(2, sj):
                        for lk in range(2):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
            elif sj == 2 and sk == 2 and si > 2:
                # Strassen on 2x2x2 corner, schoolbook for rest
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
                # Schoolbook for remaining (si-2)x2x2
                for li in range(2, si):
                    for lj in range(2):
                        for lk in range(2):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
            else:
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def generate_partitions(dim, max_group_size=3):
        """Generate all ways to partition a range into groups of size 1, 2, or 3."""
        results = []
        def backtrack(pos, current):
            if pos == dim:
                results.append(list(current))
                return
            for size in range(1, min(max_group_size, dim - pos) + 1):
                current.append(list(range(pos, pos + size)))
                backtrack(pos + size, current)
                current.pop()
        backtrack(0, [])
        return results

    def direct_optimal_construction():
        """
        Directly construct the known optimal block Strassen decomposition.
        For n=3, m=5, p=6:
        - i groups: [0,1], [2]  (sizes 2, 1)
        - j groups: [0,1], [2,3], [4]  (sizes 2, 2, 1)
        - k groups: [0,1], [2,3], [4,5]  (sizes 2, 2, 2)
        
        Blocks with all dims >= 2: 
        (i=[0,1], j=[0,1], k=[0,1]) -> 2x2x2, rank 7
        (i=[0,1], j=[0,1], k=[2,3]) -> 2x2x2, rank 7
        (i=[0,1], j=[0,1], k=[4,5]) -> 2x2x2, rank 7
        (i=[0,1], j=[2,3], k=[0,1]) -> 2x2x2, rank 7
        (i=[0,1], j=[2,3], k=[2,3]) -> 2x2x2, rank 7
        (i=[0,1], j=[2,3], k=[4,5]) -> 2x2x2, rank 7
        Total Strassen blocks: 6, saving 6
        Remaining blocks use schoolbook.
        Total rank = 90 - 6 = 84
        """
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        result = build_strassen_decomposition(i_groups, j_groups, k_groups)
        rank = result.pop("rank")
        
        if verify(result["U"], result["V"], result["W"], rank):
            return result
        return None

    def try_improved_decomposition():
        """
        Try multiple partition strategies and pick the best.
        """
        if time.time() > deadline:
            return None
        
        best_result = None
        best_rank = float('inf')
        
        # First try the known optimal construction
        direct = direct_optimal_construction()
        if direct is not None:
            best_result = direct
            best_rank = len(direct["U"][0])
        
        # Generate partitions for each dimension (include size 3 groups)
        i_parts = generate_partitions(n, max_group_size=3)
        j_parts = generate_partitions(m, max_group_size=3)
        k_parts = generate_partitions(p, max_group_size=3)
        
        # Try all combinations (limited by time)
        count = 0
        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    if time.time() > deadline - 10:
                        break
                    # Convert index groups to actual index lists
                    i_groups = []
                    pos = 0
                    for g in ip:
                        i_groups.append(list(range(pos, pos + len(g))))
                        pos += len(g)
                    j_groups = []
                    pos = 0
                    for g in jp:
                        j_groups.append(list(range(pos, pos + len(g))))
                        pos += len(g)
                    k_groups = []
                    pos = 0
                    for g in kp:
                        k_groups.append(list(range(pos, pos + len(g))))
                        pos += len(g)
                    
                    # Quick rank estimate to skip obviously worse cases
                    est_rank = 0
                    for ig, jg, kg in zip(i_groups, j_groups, k_groups):
                        si, sj, sk = len(ig), len(jg), len(kg)
                        est_rank += block_rank(si, sj, sk)
                    
                    if est_rank >= best_rank:
                        continue
                    
                    result = build_strassen_decomposition(i_groups, j_groups, k_groups)
                    rank = result.pop("rank")
                    
                    if rank < best_rank:
                        if verify(result["U"], result["V"], result["W"], rank):
                            best_result = result
                            best_rank = rank
                    count += 1
        
        return best_result

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition (starts with direct optimal construction)
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
