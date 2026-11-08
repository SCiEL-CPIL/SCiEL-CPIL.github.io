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

    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to decompose T as sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # with rank as small as possible.

    # We use a structured approach:
    # 1. Try to find a decomposition using Strassen on overlapping 2x2x2 sub-blocks
    # 2. Fall back to a known good decomposition

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact using numpy."""
        # Convert to numpy arrays for speed
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the reconstruction
                    reconstructed = 0.0
                    for r in range(rank):
                        reconstructed += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(reconstructed - 1.0) > 1e-9:
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
        return {"U": U, "V": V, "W": W, "rank": rank}

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

    def try_strassen_based():
        """
        Try to build a Strassen-based decomposition.
        
        The idea: partition the indices into blocks and apply Strassen
        where possible, schoolbook otherwise.
        
        For 3x5x6:
        - i: {0,1}, {2}  (2+1)
        - j: {0,1,2,3}, {4}  (4+1)  
        - k: {0,1,2}, {3,4,5}  (3+3)
        
        Block (2,4,3): split j into {0,1},{2,3} and k into {0,1},{2}
          -> (2,2,2) + (2,2,1) + (2,2,2) + (2,2,1) = 7+4+7+4 = 22
        
        Block (2,1,3): schoolbook = 6
        Block (1,4,3): schoolbook = 12
        Block (1,1,3): schoolbook = 3
        
        Total: 22 + 6 + 12 + 3 = 43
        
        But we can do better by being more clever about overlaps.
        """
        if time.time() > deadline:
            return None
            
        strassen_terms = strassen_2x2x2_terms()
        
        # Define the block structure
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1, 2, 3], [4]]
        k_groups = [[0, 1, 2], [3, 4, 5]]
        
        total_rank = 0
        block_decomps = []
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    
                    if si == 2 and sj >= 2 and sk >= 2:
                        # Try to decompose using Strassen sub-blocks
                        # Split j into pairs, k into groups of up to 2
                        j_pairs = [jg[i:i+2] for i in range(0, sj, 2)]
                        k_groups_local = [kg[i:i+2] for i in range(0, sk, 2)]
                        
                        sub_blocks = []
                        for jp in j_pairs:
                            for kp in k_groups_local:
                                sj_p, sk_p = len(jp), len(kp)
                                if sj_p == 2 and sk_p == 2:
                                    r = 7
                                elif sj_p == 1 and sk_p == 2:
                                    r = 3  # A single i-index: need 2 multiplications for (1,1,2) and 1 for (1,1,1)... actually schoolbook is 2
                                    # For (2,1,2): schoolbook gives 4, but maybe we can do better?
                                    # For now, use schoolbook
                                    r = 2 * 1 * 2
                                elif sj_p == 2 and sk_p == 1:
                                    r = 4
                                else:
                                    r = si * sj_p * sk_p
                                sub_blocks.append((ig, jp, kp, r, si, sj_p, sk_p))
                                total_rank += r
                        block_decomps.append(('complex', sub_blocks))
                    else:
                        r = si * sj * sk
                        total_rank += r
                        block_decomps.append(('simple', (ig, jg, kg, r, si, sj, sk)))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        for block_type, data in block_decomps:
            if block_type == 'complex':
                for (ig, jg, kg, r, si, sj, sk) in data:
                    if si == 2 and sj == 2 and sk == 2:
                        # Strassen
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
                    elif si == 2 and sj == 2 and sk == 1:
                        # Schoolbook for 2x2x1
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
                    else:
                        # General schoolbook
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
            else:  # simple
                ig, jg, kg, r, si, sj, sk = data
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        # Verify
        if verify(U, V, W, total_rank):
            return {"U": U, "V": V, "W": W, "rank": total_rank}
        
        return None

    # First try the Strassen-based approach
    result = try_strassen_based()
    
    if result is not None and result["rank"] < n * m * p:
        rank = result.pop("rank")
        return result
    
    # Fall back to schoolbook
    school_result = schoolbook()
    rank = school_result.pop("rank")
    return school_result

# EVOLVE-BLOCK-END
