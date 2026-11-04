import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    terms = []

    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))

    # --- Part 1: 4x4 * 4x4 (k=0,1,2,3) using Recursive Strassen (Rank 49) ---
    
    # Base Strassen coefficients for 2x2 * 2x2
    # u: coefficients for X entries (index 0:00, 1:01, 2:10, 3:11)
    # v: coefficients for Y entries
    # w: coefficients for Z entries
    base_strassen = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C21 += M2, C22 -= M2 -> w[2]=1, w[3]=-1
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # C12 += M3, C22 += M3 -> w[1]=1, w[3]=1
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # C11 += M4, C21 += M4 -> w[0]=1, w[2]=1
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]), # C11 -= M5, C12 += M5 -> w[0]=-1, w[1]=1
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # C22 += M6 -> w[3]=1
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # C11 += M7 -> w[0]=1
    ]
    
    # Correction on base Strassen w-vectors:
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Let's re-verify w-vectors (indices 0:Z11, 1:Z12, 2:Z21, 3:Z22)
    # M1: w = [1, 0, 0, 1]
    # M2: w = [0, 0, 1, -1]
    # M3: w = [0, 1, 0, 1]
    # M4: w = [1, 0, 1, 0]
    # M5: w = [-1, 1, 0, 0]
    # M6: w = [0, 0, 0, 1]
    # M7: w = [1, 0, 0, 0]
    
    base_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Strassen sums for 4x4 * 4x4
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in block coords
    # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1) in block coords
    # Block size is 2x2.
    # Global A index for block (bi, bj) and entry (pi, pj) is (2*bi+pi)*m + (2*bj+pj)
    # Global B index for block (bi, bj) and entry (pi, pj) is (2*bi+pi)*p + (2*bj+pj) -- WAIT, p is 7 for global, but here we only care about k=0..3.
    # In the 4x4*4x4 part, the "B" matrix is 4x4. Let's call it B4.
    # B4 index for block (bi, bj) and entry (pi, pj) is (2*bi+pi)*4 + (2*bj+pj) in the local 4x4 sense.
    # But our global V vector has size m*p = 4*7 = 28.
    # The columns k=0,1,2,3 correspond to the first 4 columns.
    # So global V index is j_global * 7 + k_global.
    # j_global = 2*bi + pi.
    # k_global = 2*bj + pj.
    
    # Global W index for block (bi, bj) and entry (pi, pj) of C (which is 4x4 here)
    # C index is k_global * n + i_global.
    # k_global = 2*bi + pi (row of C, which is col of B? No. C[i,k]. i is row, k is col).
    # In Strassen, Z is the result. Z rows correspond to i, cols to k.
    # So Z block (bi, bj) corresponds to i in [2bi, 2bi+1] and k in [2bj, 2bj+1].
    # Entry (pi, pj) in Z block corresponds to i_global = 2*bi+pi, k_global = 2*bj+pj.
    # W index = k_global * n + i_global = (2*bj+pj)*4 + (2*bi+pi).

    # Define the 7 Strassen combinations for A and B blocks
    # A_combos[m][block] = coefficient
    # B_combos[m][block] = coefficient
    # Blocks: 0:11, 1:12, 2:21, 3:22
    
    A_combos = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [1, 0, -1, 0],# M6: A21 - A11 -> Wait. M6 = (A21-A11)(B11+B12). So A21 coeff 1, A11 coeff -1.
                      # My list above: [0,0,1,-1] for u? No, u is for X.
                      # Let's re-list A coefficients for each M
    ]
    
    # Let's be precise.
    # M1: (A11+A22) -> [1, 0, 0, 1]
    # M2: (A21+A22) -> [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    # M5: (A11+A12) -> [1, 1, 0, 0]
    # M6: (A21-A11) -> [ -1, 0, 1, 0]
    # M7: (A12-A22) -> [ 0, 1, 0, -1]
    
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # B coefficients
    # M1: (B11+B22) -> [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: (B12-B22) -> [0, 1, 0, -1]
    # M4: (B21-B11) -> [0, 0, 1, -1] -- Wait. B21 is block (1,0), B11 is (0,0).
    # My block indexing: 0:11, 1:12, 2:21, 3:22.
    # So B21 is index 2, B11 is index 0.
    # M4 = A22 * (B21 - B11). Coeffs: [ -1, 0, 1, 0]
    # M5: B22 -> [0, 0, 0, 1]
    # M6: (B11+B12) -> [1, 1, 0, 0]
    # M7: (B21+B22) -> [0, 0, 1, 1]
    
    B_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Now generate 49 terms
    for m_idx in range(7):
        a_c = A_coeffs[m_idx]
        b_c = B_coeffs[m_idx]
        for s_idx in range(7):
            u_base, v_base, w_base = base_strassen[s_idx]
            
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Fill U
            # u_base indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) within the 2x2 block X
            # X is the product of A blocks.
            # For each block b in 0..3:
            #   coeff = a_c[b]
            #   if coeff != 0:
            #     for each entry pi, pj in 2x2:
            #       base_idx = pi*2 + pj
            #       base_val = u_base[base_idx]
            #       if base_val != 0:
            #         global_i = 2*(b//2) + pi  # b//2 gives row block index (0 or 1)
            #         global_j = 2*(b%2) + pj   # b%2 gives col block index (0 or 1)
            #         u_idx = global_i * m + global_j
            #         u[u_idx] += coeff * base_val
            
            for b in range(4):
                coeff_a = a_c[b]
                if coeff_a == 0: continue
                bi = b // 2
                bj = b % 2
                for pi in range(2):
                    for pj in range(2):
                        base_idx = pi * 2 + pj
                        val = u_base[base_idx]
                        if val == 0: continue
                        gi = 2 * bi + pi
                        gj = 2 * bj + pj
                        u[gi * m + gj] += coeff_a * val
            
            # Fill V
            # v_base indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) within the 2x2 block Y
            # Y is the product of B blocks.
            # For each block b in 0..3:
            #   coeff = b_c[b]
            #   if coeff != 0:
            #     for each entry pi, pj in 2x2:
            #       base_idx = pi*2 + pj
            #       base_val = v_base[base_idx]
            #       if base_val != 0:
            #         global_j = 2*(b//2) + pi
            #         global_k = 2*(b%2) + pj
            #         v_idx = global_j * p + global_k
            #         v[v_idx] += coeff * base_val
            
            for b in range(4):
                coeff_b = b_c[b]
                if coeff_b == 0: continue
                bi = b // 2
                bj = b % 2
                for pi in range(2):
                    for pj in range(2):
                        base_idx = pi * 2 + pj
                        val = v_base[base_idx]
                        if val == 0: continue
                        gj = 2 * bi + pi
                        gk = 2 * bj + pj
                        v[gj * p + gk] += coeff_b * val
            
            # Fill W
            # w_base indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) within the 2x2 block Z
            # Z is the result C block.
            # For each block b in 0..3:
            #   coeff = 1 (since Z = X*Y, and we are summing these rank-1 terms to form Z)
            #   Wait, the w_base already contains the coefficients for the reconstruction of Z from M_s.
            #   So we just map the base w values to the global C positions.
            #   For each entry pi, pj in 2x2:
            #     base_idx = pi*2 + pj
            #     base_val = w_base[base_idx]
            #     if base_val != 0:
            #       global_i = 2*(b//2) + pi
            #       global_k = 2*(b%2) + pj
            #       w_idx = global_k * n + global_i
            #       w[w_idx] += base_val
            
            for b in range(4):
                bi = b // 2
                bj = b % 2
                for pi in range(2):
                    for pj in range(2):
                        base_idx = pi * 2 + pj
                        val = w_base[base_idx]
                        if val == 0: continue
                        gi = 2 * bi + pi
                        gk = 2 * bj + pj
                        w[gk * n + gi] += val
            
            add_term(u, v, w)

    # --- Part 2: 4x4 * 4x3 (k=4,5,6) using Schoolbook (Rank 48) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                add_term(u, v, w)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
