"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
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

    def generate_partitions(size, max_block=6):
        """Generate all partitions of size into blocks of size 1..max_block."""
        if size == 0:
            yield []
            return
        for first in range(1, min(size, max_block) + 1):
            for rest in generate_partitions(size - first, max_block):
                yield [first] + rest

    def compute_block_rank(si, sj, sk):
        """Compute the rank for a block of given dimensions."""
        if si == 2 and sj == 2 and sk == 2:
            return 7  # Strassen
        else:
            return si * sj * sk

    def total_rank_for_partition(i_part, j_part, k_part):
        """Compute total rank for given partitions."""
        total = 0
        for si in i_part:
            for sj in j_part:
                for sk in k_part:
                    total += compute_block_rank(si, sj, sk)
        return total

    def find_best_partition():
        """Find the partition that minimizes total rank."""
        best_rank = n * m * p  # schoolbook upper bound
        best_partition = None
        
        i_partitions = list(generate_partitions(n, max_block=3))
        j_partitions = list(generate_partitions(m, max_block=5))
        k_partitions = list(generate_partitions(p, max_block=6))
        
        for ip in i_partitions:
            for jp in j_partitions:
                for kp in k_partitions:
                    r = total_rank_for_partition(ip, jp, kp)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (ip, jp, kp)
        
        return best_partition, best_rank

    def strassen_2x3x2_terms():
        """
        Return rank-11 decomposition for 2x3x2 matrix multiplication.
        C = A*B where A is 2x3, B is 3x2.
        """
        terms = []
        # We can view this as combining two 2x2x2 problems with sharing
        # A = [A00 A01 A02], B = [[B00 B01],[B10 B11],[B20 B21]]
        # C00 = A00*B00 + A01*B10 + A02*B20
        # C01 = A00*B01 + A01*B11 + A02*B21
        # C10 = A10*B00 + A11*B10 + A12*B20
        # C11 = A10*B01 + A11*B11 + A12*B21
        
        # Use 11 terms based on known result
        # Term 1: (A00+A10) * (B00+B10) -> contributes to C00, C10, C01, C11
        u1 = {(0,0): 1, (1,0): 1}
        v1 = {(0,0): 1, (1,0): 1}
        w1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: (A00-A10) * (B00-B10)
        u2 = {(0,0): 1, (1,0): -1}
        v2 = {(0,0): 1, (1,0): -1}
        w2 = {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: A01 * B10
        u3 = {(0,1): 1}
        v3 = {(1,0): 1}
        w3 = {(0,0): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: A11 * B10
        u4 = {(1,1): 1}
        v4 = {(1,0): 1}
        w4 = {(1,0): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: A02 * B20
        u5 = {(0,2): 1}
        v5 = {(2,0): 1}
        w5 = {(0,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: A12 * B20
        u6 = {(1,2): 1}
        v6 = {(2,0): 1}
        w6 = {(1,0): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: A01 * B11
        u7 = {(0,1): 1}
        v7 = {(1,1): 1}
        w7 = {(0,1): 1}
        terms.append((u7, v7, w7))
        
        # Term 8: A11 * B11
        u8 = {(1,1): 1}
        v8 = {(1,1): 1}
        w8 = {(1,1): 1}
        terms.append((u8, v8, w8))
        
        # Term 9: A02 * B21
        u9 = {(0,2): 1}
        v9 = {(2,1): 1}
        w9 = {(0,1): 1}
        terms.append((u9, v9, w9))
        
        # Term 10: A12 * B21
        u10 = {(1,2): 1}
        v10 = {(2,1): 1}
        w10 = {(1,1): 1}
        terms.append((u10, v10, w10))
        
        # Term 11: (A00+A10) * (B01-B11)
        u11 = {(0,0): 1, (1,0): 1}
        v11 = {(0,1): 1, (1,1): -1}
        w11 = {(0,1): 1, (1,1): 1}
        terms.append((u11, v11, w11))
        
        return terms

    def strassen_3x2x2_terms():
        """
        Return rank-11 decomposition for 3x2x2 matrix multiplication.
        C = A*B where A is 3x2, B is 2x2.
        """
        terms = []
        # A = [[A00 A01],[A10 A11],[A20 A21]], B = [[B00 B01],[B10 B11]]
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # C20 = A20*B00 + A21*B10
        # C21 = A20*B01 + A21*B11
        
        # Term 1: (A00+A10) * (B00+B10)
        u1 = {(0,0): 1, (1,0): 1}
        v1 = {(0,0): 1, (1,0): 1}
        w1 = {(0,0): 1, (1,0): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: (A00-A10) * (B00-B10)
        u2 = {(0,0): 1, (1,0): -1}
        v2 = {(0,0): 1, (1,0): -1}
        w2 = {(0,0): 1, (1,0): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: A20 * (B00+B10)
        u3 = {(2,0): 1}
        v3 = {(0,0): 1, (1,0): 1}
        w3 = {(2,0): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: (A00+A10) * (B01+B11)
        u4 = {(0,0): 1, (1,0): 1}
        v4 = {(0,1): 1, (1,1): 1}
        w4 = {(0,1): 1, (1,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: (A00-A10) * (B01-B11)
        u5 = {(0,0): 1, (1,0): -1}
        v5 = {(0,1): 1, (1,1): -1}
        w5 = {(0,1): 1, (1,1): -1}
        terms.append((u5, v5, w5))
        
        # Term 6: A20 * (B01+B11)
        u6 = {(2,0): 1}
        v6 = {(0,1): 1, (1,1): 1}
        w6 = {(2,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: A01 * (B00+B10)
        u7 = {(0,1): 1}
        v7 = {(0,0): 1, (1,0): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        # Term 8: A11 * (B00+B10)
        u8 = {(1,1): 1}
        v8 = {(0,0): 1, (1,0): 1}
        w8 = {(1,0): 1}
        terms.append((u8, v8, w8))
        
        # Term 9: A01 * (B01+B11)
        u9 = {(0,1): 1}
        v9 = {(0,1): 1, (1,1): 1}
        w9 = {(0,1): 1}
        terms.append((u9, v9, w9))
        
        # Term 10: A11 * (B01+B11)
        u10 = {(1,1): 1}
        v10 = {(0,1): 1, (1,1): 1}
        w10 = {(1,1): 1}
        terms.append((u10, v10, w10))
        
        # Term 11: A21 * (B00+B10+B01+B11) -- this won't work, need different approach
        # Actually, let me reconsider. The rank of 3x2x2 is 11.
        # Let me use a different construction.
        # For now, return 12 terms (schoolbook) as fallback
        terms = []
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    terms.append((u, v, w))
        return terms

    def strassen_2x2x3_terms():
        """
        Return rank-11 decomposition for 2x2x3 matrix multiplication.
        C = A*B where A is 2x2, B is 2x3.
        """
        terms = []
        # A = [[A00 A01],[A10 A11]], B = [[B00 B01 B02],[B10 B11 B12]]
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C02 = A00*B02 + A01*B12
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # C12 = A10*B02 + A11*B12
        
        # For now, use schoolbook (12 terms) as fallback
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    terms.append((u, v, w))
        return terms

    def build_decomposition(i_part, j_part, k_part):
        """Build the actual decomposition matrices for the given partition."""
        # Convert partition sizes to actual index groups
        i_groups = []
        idx = 0
        for s in i_part:
            i_groups.append(list(range(idx, idx + s)))
            idx += s
        
        j_groups = []
        idx = 0
        for s in j_part:
            j_groups.append(list(range(idx, idx + s)))
            idx += s
        
        k_groups = []
        idx = 0
        for s in k_part:
            k_groups.append(list(range(idx, idx + s)))
            idx += s
        
        # Compute total rank
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    total_rank += compute_block_rank(si, sj, sk)
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_222 = strassen_2x2x2_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen
                        for (u_pat, v_pat, w_pat) in strassen_222:
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

    def verify_fast(U, V, W, rank):
        """Verify using numpy for speed."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(actual - 1.0) > 1e-9:
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

    # Find the best partition
    best_partition, best_rank = find_best_partition()
    
    # Build the decomposition
    if best_partition is not None and best_rank < n * m * p:
        result = build_decomposition(*best_partition)
        rank = result.pop("rank")
        
        # Verify
        if verify(result["U"], result["V"], result["W"], rank):
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
