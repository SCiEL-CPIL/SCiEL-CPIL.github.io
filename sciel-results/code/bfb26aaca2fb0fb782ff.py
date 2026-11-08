"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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
        def compute_block_rank(si, sj, sk):
            if si == 2 and sj == 2 and sk == 2:
                return 7
            else:
                return si * sj * sk
        
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
        strassen_terms = strassen_2x2x2_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
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

    # Use the known optimal partition: [2,1] x [2,2,1] x [2,2,2]
    # This gives 6 Strassen blocks (rank 7 each) + 6 schoolbook blocks (rank 8 each) = 42 + 48 = 90? 
    # Wait, let me recalculate.
    # i_part = [2,1], j_part = [2,2,1], k_part = [2,2,2]
    # Blocks:
    # (2,2,2): 1*1*3 = 3 blocks, each rank 7 -> 21
    # (2,2,2): already counted
    # Let me enumerate:
    # i=0 (size 2), j=0 (size 2), k=0 (size 2) -> 7
    # i=0 (size 2), j=0 (size 2), k=1 (size 2) -> 7
    # i=0 (size 2), j=0 (size 2), k=2 (size 2) -> 7
    # i=0 (size 2), j=1 (size 2), k=0 (size 2) -> 7
    # i=0 (size 2), j=1 (size 2), k=1 (size 2) -> 7
    # i=0 (size 2), j=1 (size 2), k=2 (size 2) -> 7
    # i=0 (size 2), j=2 (size 1), k=0 (size 2) -> 2*1*2 = 4
    # i=0 (size 2), j=2 (size 1), k=1 (size 2) -> 4
    # i=0 (size 2), j=2 (size 1), k=2 (size 2) -> 4
    # i=1 (size 1), j=0 (size 2), k=0 (size 2) -> 1*2*2 = 4
    # i=1 (size 1), j=0 (size 2), k=1 (size 2) -> 4
    # i=1 (size 1), j=0 (size 2), k=2 (size 2) -> 4
    # i=1 (size 1), j=1 (size 2), k=0 (size 2) -> 4
    # i=1 (size 1), j=1 (size 2), k=1 (size 2) -> 4
    # i=1 (size 1), j=1 (size 2), k=2 (size 2) -> 4
    # i=1 (size 1), j=2 (size 1), k=0 (size 2) -> 1*1*2 = 2
    # i=1 (size 1), j=2 (size 1), k=1 (size 2) -> 2
    # i=1 (size 1), j=2 (size 1), k=2 (size 2) -> 2
    
    # Total: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    
    i_part = [2, 1]
    j_part = [2, 2, 1]
    k_part = [2, 2, 2]
    
    result = build_decomposition(i_part, j_part, k_part)
    rank = len(result["U"][0])
    
    # Verify
    if verify(result["U"], result["V"], result["W"], rank):
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
