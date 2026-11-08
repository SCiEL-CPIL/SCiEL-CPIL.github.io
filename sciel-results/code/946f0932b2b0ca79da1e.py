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
        Each term is (u_dict, v_dict, w_dict) where:
        - u_dict maps (local_i, local_j) -> coefficient for U
        - v_dict maps (local_j, local_k) -> coefficient for V  
        - w_dict maps (local_k, local_i) -> coefficient for W
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # Term 2: M2 = (A10+A11)B00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        # Term 3: M3 = A00(B01-B11)
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        # Term 4: M4 = A11(B10-B00)
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        # Term 5: M5 = (A00+A01)B11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ))
        # Term 6: M6 = (A10-A00)(B00+B01)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # Term 7: M7 = (A01-A11)(B10+B11)
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        return terms

    def build_block_decomposition(i_groups, j_groups, k_groups):
        """
        Build a block decomposition using Strassen for 2x2x2 blocks
        and schoolbook for all other blocks.
        """
        strassen_terms = strassen_2x2x2_terms()
        
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
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_all_partitions():
        """
        Try all possible partitions of i, j, k into groups of size 1 or 2
        and find the one with minimum rank.
        """
        if time.time() > deadline:
            return None
        
        # Generate all partitions of range(N) into groups of size 1 or 2
        def gen_partitions(N):
            """Generate all ways to partition [0..N-1] into consecutive groups of size 1 or 2."""
            if N == 0:
                return [[]]
            results = []
            # Start with group of size 1
            for rest in gen_partitions(N - 1):
                results.append([[0]] + rest)
            # Start with group of size 2
            if N >= 2:
                for rest in gen_partitions(N - 2):
                    results.append([[0, 1]] + rest)
            return results
        
        i_parts = gen_partitions(n)
        j_parts = gen_partitions(m)
        k_parts = gen_partitions(p)
        
        best_rank = n * m * p
        best_result = None
        
        for ip in i_parts:
            if time.time() > deadline:
                break
            for jp in j_parts:
                if time.time() > deadline:
                    break
                for kp in k_parts:
                    if time.time() > deadline:
                        break
                    result = build_block_decomposition(ip, jp, kp)
                    rank = result["rank"]
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
        
        if best_result is not None and best_rank < n * m * p:
            if verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
                best_result.pop("rank")
                return best_result
        
        return None

    # Try all partitions to find the best block decomposition
    best_result = None
    best_rank = n * m * p
    
    partitioned = try_all_partitions()
    if partitioned is not None:
        best_result = partitioned
        best_rank = len(best_result["U"][0])
    
    # Fallback to schoolbook
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
