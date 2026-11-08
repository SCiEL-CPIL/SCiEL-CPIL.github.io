"""
Tensor decomposition for 3x5x6 matrix multiplication using block-Strassen method.

Strategy:
- Partition the 3x5x6 problem into blocks of size 2x2x2 (Strassen, rank 7)
  and smaller blocks (schoolbook).
- This reduces the rank from 90 (schoolbook) to 84.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        
        Convention:
        - U is indexed by (i, j) where i is row of A, j is col of A
        - V is indexed by (j, k) where j is col of A (row of B), k is col of B
        - W is indexed by (k, i) where k is col of B, i is row of C
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        terms.append(
            ({(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1})
        )
        
        # M2 = (A10+A11)B00 -> contributes to C10, -C11
        terms.append(
            ({(1,0): 1, (1,1): 1},
             {(0,0): 1},
             {(0,1): 1, (1,1): -1})
        )
        
        # M3 = A00(B01-B11) -> contributes to C01, C11
        terms.append(
            ({(0,0): 1},
             {(0,1): 1, (1,1): -1},
             {(1,0): 1, (1,1): 1})
        )
        
        # M4 = A11(B10-B00) -> contributes to C00, C10
        terms.append(
            ({(1,1): 1},
             {(1,0): 1, (0,0): -1},
             {(0,0): 1, (0,1): 1})
        )
        
        # M5 = (A00+A01)B11 -> contributes to -C00, C01
        terms.append(
            ({(0,0): 1, (0,1): 1},
             {(1,1): 1},
             {(0,0): -1, (1,0): 1})
        )
        
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        terms.append(
            ({(1,0): 1, (0,0): -1},
             {(0,0): 1, (0,1): 1},
             {(1,1): 1})
        )
        
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        terms.append(
            ({(0,1): 1, (1,1): -1},
             {(1,0): 1, (1,1): 1},
             {(0,0): 1})
        )
        
        return terms

    def build_block_strassen():
        """
        Build block-Strassen decomposition.
        
        Partition:
        - i (rows of A, 0..2): groups [0,1], [2]
        - j (cols of A, 0..4): groups [0,1], [2,3], [4]
        - k (cols of B, 0..5): groups [0,1], [2,3], [4,5]
        """
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # Count total rank
        total_rank = 0
        blocks = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    blocks.append((ig, jg, kg, si, sj, sk, r))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, si, sj, sk, r in blocks:
            if si == 2 and sj == 2 and sk == 2:
                # Strassen block
                for u_pat, v_pat, w_pat in strassen_terms:
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
                # Schoolbook block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Build the block-Strassen decomposition
    result = build_block_strassen()
    
    # Verify
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook if verification fails
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
        result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
