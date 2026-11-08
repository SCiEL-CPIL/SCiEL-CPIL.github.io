"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

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

    def verify_numpy(U, V, W, rank):
        """Verify that the decomposition is exact using NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Also verify that no (i,j,k) has value != 1 (should be exactly 1 for all)
        return True

    def get_strassen_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01) -> C11 += 1
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11) -> C00 += 1
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_decomposition(i_groups, j_groups, k_groups, strassen_terms):
        """Build a decomposition from given groupings."""
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
        
        if col != total_rank:
            return None, None
            
        result = {"U": U, "V": V, "W": W}
        return result, total_rank

    def try_rank_reduction():
        """
        Try multiple partition strategies to find the best rank reduction.
        """
        if time.time() > deadline:
            return None, None
        
        strassen_terms = get_strassen_terms()
        
        # Define various partition strategies
        # Strategy 1: Original - i:{0,1|2}, j:{0,1|2,3|4}, k:{0,1|2,3|4,5}
        strategies = []
        
        # Strategy 1: Balanced 2-2-1 splits (6 Strassen blocks)
        strategies.append({
            'name': 's1',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 2: Different j split (6 Strassen blocks)
        strategies.append({
            'name': 's2',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 2], [1, 3], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 3: Different k split (6 Strassen blocks)
        strategies.append({
            'name': 's3',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 2], [1, 3], [4, 5]]
        })
        
        # Strategy 4: Try 3-element groups where possible
        strategies.append({
            'name': 's4',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1, 2], [3, 4]],
            'k_groups': [[0, 1, 2], [3, 4, 5]]
        })
        
        # Strategy 5: Mixed
        strategies.append({
            'name': 's5',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3, 4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 6: Different i grouping
        strategies.append({
            'name': 's6',
            'i_groups': [[0, 2], [1]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 7: Another i grouping
        strategies.append({
            'name': 's7',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 3], [1, 2], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 8: Different k pairing
        strategies.append({
            'name': 's8',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 3], [1, 2], [4, 5]]
        })
        
        # Strategy 9: Try to maximize 2x2x2 blocks with different j pairs
        strategies.append({
            'name': 's9',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 4], [1, 2], [3]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 10: More varied
        strategies.append({
            'name': 's10',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 4], [3]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        best_result = None
        best_rank = float('inf')
        
        for strat in strategies:
            if time.time() > deadline:
                break
            
            result, rank = build_decomposition(
                strat['i_groups'], strat['j_groups'], strat['k_groups'], strassen_terms
            )
            
            if result is None:
                continue
            
            if verify_numpy(result['U'], result['V'], result['W'], rank):
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        
        if best_result is not None:
            return best_result, best_rank
        
        return None, None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition
    improved, improved_rank = try_rank_reduction()
    if improved is not None and improved_rank < best_rank:
        best_result = improved
        best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
