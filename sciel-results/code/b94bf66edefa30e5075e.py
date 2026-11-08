"""Matrix multiplication 3x5x6: rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        terms = []
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}))
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}))
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    STRASSEN = strassen_2x2x2_terms()

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_decomposition(i_groups, j_groups, k_groups):
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
                for (u_pat, v_pat, w_pat) in STRASSEN:
                    for (li, lj), val_u in u_pat.items():
                        U[ig[li] * m + jg[lj]][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        V[jg[lj] * p + kg[lk]][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        W[kg[lk] * n + ig[li]][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            U[ig[li] * m + jg[lj]][col] = 1
                            V[jg[lj] * p + kg[lk]][col] = 1
                            W[kg[lk] * n + ig[li]][col] = 1
                            col += 1

        return U, V, W, total_rank

    def gen_set_partitions(size):
        if size == 0:
            yield []
            return
        if size == 1:
            yield [frozenset([0])]
            return
        for part in gen_set_partitions(size - 1):
            yield part + [frozenset([size - 1])]
            for i in range(len(part)):
                new_part = list(part)
                new_part[i] = new_part[i] | frozenset([size - 1])
                yield new_part

    def find_best_partition():
        i_parts = [tuple(sorted(g) for g in part) for part in gen_set_partitions(n)]
        j_parts = [tuple(sorted(g) for g in part) for part in gen_set_partitions(m)]
        k_parts = [tuple(sorted(g) for g in part) for part in gen_set_partitions(p)]

        best_rank = n * m * p
        best_result = None

        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    est_rank = 0
                    for ig in ip:
                        for jg in jp:
                            for kg in kp:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    est_rank += 7
                                else:
                                    est_rank += si * sj * sk
                    if est_rank >= best_rank:
                        continue
                    ip_list = [list(ig) for ig in ip]
                    jp_list = [list(jg) for jg in jp]
                    kp_list = [list(kg) for kg in kp]
                    U, V, W, rank = build_decomposition(ip_list, jp_list, kp_list)
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)

        return best_result, best_rank

    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    if time.time() < deadline - 20:
        found, found_rank = find_best_partition()
        if found is not None and found_rank < best_rank:
            best_result = {"U": found[0], "V": found[1], "W": found[2]}
            best_rank = found_rank

    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result
# EVOLVE-BLOCK-END
