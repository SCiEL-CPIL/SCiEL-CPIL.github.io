import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 115.0

    # Strategy: Use a hierarchical decomposition.
    # Level 1: Strassen 2x2 of 3x3 blocks (7 multiplications of 3x3 matrices).
    # Level 2: Decompose 3x3 matrix multiplication using the BCLT (Bini-Capovani-Lotti-Turletti) algorithm,
    # which has rank 23.
    # Total rank = 7 * 23 = 161.
    # This is an improvement over the naive 7 * 27 = 189 (Strassen + schoolbook 3x3).
    # 1.1429 is the score for rank 189? 
    # Score = seed / value.
    # If seed is 213 (e.g. Strassen 3x3x3 rank 23 * 9? No. 6x6x6 naive is 216. Strassen 6x6 is 189. 
    # 216/189 = 1.142857... Yes, the seed is likely 216 (naive) and current best is 189.
    # So if we achieve rank 161, score = 216 / 161 = 1.3416...
    # This would be a significant improvement.

    rank3 = 23
    rank = 7 * rank3

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen definitions for 2x2 blocks of 3x3 matrices
    # Blocks: 0=00, 1=01, 2=10, 3=11
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    # Reconstruction coefficients for Strassen
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    # BCLT algorithm for 3x3 matrix multiplication
    # We need to define the 23 terms for 3x3.
    # The BCLT algorithm is a specific rank-23 decomposition.
    # Finding the exact coefficients for BCLT 3x3 is non-trivial.
    # However, we can use a known rank-23 algorithm.
    # One such algorithm is by Bini et al. (1976).
    # Let's try to construct it or use a known implementation.
    # If I cannot find the exact BCLT coefficients, I might have to fall back to rank 27 (schoolbook) for the inner 3x3,
    # which gives rank 189.
    # But the task asks for rank-minimal. 161 is better than 189.
    # Is there a simpler rank-23?
    # Actually, the rank of 3x3 matrix multiplication is known to be 19 (lower bound) and 23 (upper bound by BCLT).
    # It is not known if 19, 20, 21, or 22 is possible. 23 is the best known upper bound.
    # So 161 is the best known upper bound for 6x6 using this hierarchy.
    
    # I need the explicit coefficients for BCLT 3x3.
    # Since I don't have them memorized, I will attempt to generate a valid rank-23 decomposition if possible,
    # or stick to rank 27 if I can't verify rank 23 quickly.
    # However, writing a rank-23 decomposition from scratch is risky.
    # Let's look at the inspirations. Inspiration 2 mentions BCLT but the code is cut off.
    # If I use rank 27, I get 189. Score 1.1429.
    # If I can get rank 23, I get 161. Score 1.3416.
    # Is there a way to get rank < 189 without BCLT?
    # Strassen 2x2 of 3x3 is 189.
    # What about Strassen 2x2 of 3x3 but with some optimization?
    # No, the inner 3x3 must be exact.
    
    # Let's check if there is a simpler improvement.
    # Maybe use a different block size?
    # 6x6 can be viewed as 3x3 of 2x2.
    # Rank of 2x2 is 7 (Strassen).
    # Rank of 3x3 is 23 (BCLT) or 27 (Schoolbook).
    # If we use 3x3 of 2x2:
    # Outer: 3x3 matrix multiplication of 2x2 blocks.
    # If we use BCLT (23) for the outer 3x3, and Strassen (7) for the inner 2x2:
    # Total rank = 23 * 7 = 161.
    # This is symmetric and also yields 161.
    
    # Which is easier to implement?
    # 2x2 Strassen coefficients are well-known and simple.
    # 3x3 BCLT coefficients are complex.
    # So implementing "3x3 of 2x2" with BCLT outer and Strassen inner is hard because of BCLT.
    # Implementing "2x2 of 3x3" with Strassen outer and BCLT inner is hard because of BCLT.
    
    # Is there a rank-23 algorithm for 3x3 that is simpler to code?
    # Or can I find the coefficients?
    # The BCLT algorithm is often described in terms of a specific set of 23 products.
    # Without the explicit coefficients, I cannot implement it.
    
    # Alternative: Use the known rank-19 lower bound? No, that's a lower bound, not an algorithm.
    # The best known algorithm for 3x3 is rank 23.
    
    # If I cannot implement rank 23, I should stick to rank 27 (189 total).
    # But wait, is there any other trick?
    # What if I use a different decomposition for 6x6 directly?
    # The rank of 6x6 matrix multiplication is at most 161.
    # Is it possible to get a rank between 161 and 189?
    # Maybe 162? 163?
    # Unlikely to be easy.
    
    # Let's try to find the BCLT coefficients online or in memory.
    # Bini, Capovani, Lotti, Turletti (1976) "A 23-multiplication algorithm for 3x3 matrix multiplication".
    # The algorithm is not widely published in a simple list format like Strassen.
    # However, there are implementations in libraries.
    # Since I cannot access the internet, I must rely on my knowledge.
    # I do not have the 23 terms memorized.
    
    # Therefore, I will stick to the rank 189 solution (Strassen 2x2 of 3x3 schoolbook).
    # But I can try to optimize the code to be faster and cleaner.
    # The current program already does this.
    # Is there any other improvement?
    # Maybe using integer arithmetic to avoid floating point issues?
    # The problem asks for integer/half-integer entries.
    # Strassen uses 1, -1, 0. So it's integer.
    
    # Let's double check if there's a rank-16 algorithm for 4x4 that can be combined?
    # 6x6 is not a multiple of 4.
    
    # What if I use a 2x2 Strassen where the blocks are not square?
    # No, standard Strassen requires square blocks for the standard 7-term formula.
    # Rectangular Strassen exists but is more complex.
    
    # Let's assume the rank 189 is the best I can do without BCLT coefficients.
    # Wait, the inspiration 2 code was cut off but it claimed to use BCLT.
    # If I could just copy the BCLT part from inspiration 2... but it's omitted.
    # I will have to implement the rank 189 version robustly.
    
    # One potential improvement:
    # The current code uses `np.einsum` for verification. This is slow.
    # I can skip verification in the final version to save time, or do it very quickly.
    # But correctness is paramount.
    
    # Let's refine the rank 189 implementation.
    
    bs = 3
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen definitions
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # U update
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    
                    # V update
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
                    
                    # W update
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                            
                    r_idx += 1
    
    # Verification
    # Construct the tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0
                
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback to naive
        U_std = np.zeros((n * m, n * m * p), dtype=float)
        V_std = np.zeros((m * p, n * m * p), dtype=float)
        W_std = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_std[i * m + k, idx] = 1.0
                    V_std[k * p + j, idx] = 1.0
                    W_std[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
