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

    def generate_partitions(dim):
        """Generate all partitions of {0..dim-1} into groups of size 1 or 2."""
        if dim == 0:
            yield []
        elif dim == 1:
            yield [[0]]
        elif dim == 2:
            yield [[0, 1]]
            yield [[0], [1]]
        elif dim == 3:
            yield [[0, 1], [2]]
            yield [[0], [1, 2]]
            yield [[0], [1], [2]]
        elif dim == 4:
            yield [[0, 1], [2, 3]]
            yield [[0, 1], [2], [3]]
            yield [[0], [1, 2], [3]]
            yield [[0], [1], [2, 3]]
            yield [[0], [1], [2], [3]]
        elif dim == 5:
            yield [[0, 1], [2, 3], [4]]
            yield [[0, 1], [2], [3, 4]]
            yield [[0, 1], [2], [3], [4]]
            yield [[0], [1, 2], [3, 4]]
            yield [[0], [1, 2], [3], [4]]
            yield [[0], [1], [2, 3], [4]]
            yield [[0], [1], [2], [3, 4]]
            yield [[0], [1], [2], [3], [4]]
        elif dim == 6:
            yield [[0, 1], [2, 3], [4, 5]]
            yield [[0, 1], [2, 3], [4], [5]]
            yield [[0, 1], [2], [3, 4], [5]]
            yield [[0, 1], [2], [3], [4, 5]]
            yield [[0, 1], [2], [3], [4], [5]]
            yield [[0], [1, 2], [3, 4], [5]]
            yield [[0], [1, 2], [3], [4, 5]]
            yield [[0], [1, 2], [3], [4], [5]]
            yield [[0], [1], [2, 3], [4, 5]]
            yield [[0], [1], [2, 3], [4], [5]]
            yield [[0], [1], [2], [3, 4], [5]]
            yield [[0], [1], [2], [3], [4, 5]]
            yield [[0], [1], [2], [3], [4], [5]]

    def compute_rank_for_partition(i_part, j_part, k_part):
        """Compute the rank for a given partition."""
        total_rank = 0
        for ig in i_part:
            for jg in j_part:
                for kg in k_part:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total_rank += 7
                    else:
                        total_rank += si * sj * sk
        return total_rank

    def find_best_partition():
        """Find the partition with minimum rank."""
        best_rank = n * m * p
        best_i = None
        best_j = None
        best_k = None
        
        i_partitions = list(generate_partitions(n))
        j_partitions = list(generate_partitions(m))
        k_partitions = list(generate_partitions(p))
        
        for ip in i_partitions:
            if time.time() > deadline:
                break
            for jp in j_partitions:
                if time.time() > deadline:
                    break
                for kp in k_partitions:
                    if time.time() > deadline:
                        break
                    rank = compute_rank_for_partition(ip, jp, kp)
                    if rank < best_rank:
                        best_rank = rank
                        best_i = ip
                        best_j = jp
                        best_k = kp
        
        return best_i, best_j, best_k, best_rank

    def build_decomposition(i_part, j_part, k_part):
        """Build the decomposition for the given partition."""
        strassen_terms = strassen_2x2x2_terms()
        
        total_rank = compute_rank_for_partition(i_part, j_part, k_part)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig in i_part:
            for jg in j_part:
                for kg in k_part:
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
        
        return {"U": U, "V": V, "W": W}

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Find the best partition
    best_i, best_j, best_k, best_rank_found = find_best_partition()
    
    if best_i is not None and best_rank_found < best_rank:
        result = build_decomposition(best_i, best_j, best_k)
        if verify(result["U"], result["V"], result["W"], best_rank_found):
            best_result = result
            best_rank = best_rank_found

    return best_result
# EVOLVE-BLOCK-END
