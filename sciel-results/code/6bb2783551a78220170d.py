"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
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
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_hybrid_decomposition():
        """
        Build a decomposition using a hybrid approach:
        - Use Strassen on 2x2x2 blocks where beneficial
        - Use schoolbook for remaining parts
        - Carefully combine to avoid double-counting
        
        Partition strategy for 3x5x6:
        - i: {0,1}, {2}
        - j: {0,1}, {2,3}, {4}  
        - k: {0,1}, {2,3}, {4,5}
        
        We'll use Strassen for the (2,2,2) block and schoolbook for the rest,
        but we need to be careful about overlapping contributions.
        """
        strassen_terms = strassen_2x2x2_terms()
        
        # Define the partition
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # Count total rank
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen for this block
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
                # Schoolbook for this block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def build_optimized_decomposition():
        """
        Try to find an optimized decomposition with rank < 90.
        Uses a more clever partitioning that exploits the 3x5x6 structure.
        """
        if time.time() > deadline:
            return None
        
        # Strategy: Use multiple overlapping Strassen blocks
        # For 3x5x6, we can use:
        # - Block A: rows {0,1}, cols {0,1}, k {0,1} -> Strassen (7)
        # - Block B: rows {0,1}, cols {2,3}, k {2,3} -> Strassen (7)
        # - Block C: rows {0,1}, cols {4}, k {4,5} -> schoolbook (4)
        # - Block D: rows {2}, cols {0,1}, k {0,1} -> schoolbook (4)
        # - Block E: rows {2}, cols {2,3}, k {2,3} -> schoolbook (4)
        # - Block F: rows {2}, cols {4}, k {4,5} -> schoolbook (2)
        # Total: 7+7+4+4+4+2 = 28... that's way too low, something's wrong
        
        # Actually, the issue is that we're not covering all (i,j,k) tuples.
        # Let me use the simple block approach which gives rank 90,
        # and then try to optimize.
        
        result = build_hybrid_decomposition()
        rank = result["rank"]
        
        if rank < n * m * p:
            # Verify
            if verify(result["U"], result["V"], result["W"], rank):
                result.pop("rank")
                return result
        
        return None

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen for all possible 2x2x2 blocks.
        This is a more aggressive approach that tries to use Strassen
        wherever possible.
        """
        if time.time() > deadline:
            return None
        
        strassen_terms = strassen_2x2x2_terms()
        
        # Try all possible partitions of i, j, k into groups of size 1 or 2
        # For n=3, m=5, p=6
        # i can be: [0,1,2], [0,1],[2], [0],[1,2], [0],[1],[2]
        # j can be: [0,1,2,3,4], [0,1],[2,3],[4], [0,1],[2],[3,4], etc.
        # p can be: [0,1,2,3,4,5], [0,1],[2,3],[4,5], etc.
        
        # Let's try a specific partition that maximizes Strassen usage
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # Count total rank
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen for this block
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
                # Schoolbook for this block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_tensor_rank_minimization():
        """
        Try to minimize the tensor rank by removing linearly dependent terms.
        This is a heuristic approach that checks if any term can be removed
        without affecting the decomposition.
        """
        if time.time() > deadline:
            return None
        
        # Get the hybrid decomposition
        result = build_hybrid_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        rank = result["rank"]
        
        # Try to remove each term and see if the decomposition still works
        for c in range(rank):
            if time.time() > deadline:
                return None
            
            # Temporarily remove term c
            U_temp = [row[:] for row in U]
            V_temp = [row[:] for row in V]
            W_temp = [row[:] for row in W]
            for i in range(n * m):
                U_temp[i][c] = 0
            for j in range(m * p):
                V_temp[j][c] = 0
            for k in range(p * n):
                W_temp[k][c] = 0
            
            # Check if the decomposition still works
            if verify(U_temp, V_temp, W_temp, rank):
                # Term c is not needed, remove it
                U_new = []
                V_new = []
                W_new = []
                for i in range(n * m):
                    U_new.append([U_temp[i][c2] for c2 in range(rank) if c2 != c])
                for j in range(m * p):
                    V_new.append([V_temp[j][c2] for c2 in range(rank) if c2 != c])
                for k in range(p * n):
                    W_new.append([W_temp[k][c2] for c2 in range(rank) if c2 != c])
                
                return {"U": U_new, "V": V_new, "W": W_new, "rank": rank - 1}
        
        return None

    def try_improved_decomposition():
        """
        Try an improved decomposition that uses a different partitioning strategy.
        """
        if time.time() > deadline:
            return None
        
        # Try a different partition that might be more efficient
        i_groups = [[0], [1, 2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        strassen_terms = strassen_2x2x2_terms()
        
        # Count total rank
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen for this block
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
                # Schoolbook for this block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        result = {"U": U, "V": V, "W": W, "rank": total_rank}
        
        if total_rank < n * m * p:
            # Verify
            if verify(U, V, W, total_rank):
                return result
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try optimized decomposition
    optimized = build_optimized_decomposition()
    if optimized is not None:
        optimized_rank = len(optimized["U"][0])
        if optimized_rank < best_rank:
            best_result = optimized
            best_rank = optimized_rank

    # Try Strassen decomposition
    strassen = build_strassen_decomposition()
    if strassen is not None:
        strassen_rank = len(strassen["U"][0])
        if strassen_rank < best_rank:
            best_result = strassen
            best_rank = strassen_rank

    # Try tensor rank minimization
    minimized = try_tensor_rank_minimization()
    if minimized is not None:
        minimized_rank = len(minimized["U"][0])
        if minimized_rank < best_rank:
            best_result = minimized
            best_rank = minimized_rank

    # Try improved decomposition
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
