"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 10
    rank = 50
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    terms = []
    
    # Rank 5 decomposition for 2x3 * 3x1 sub-problem, replicated for each column
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
        a00, a01, a02 = 0, 1, 2
        a10, a11, a12 = 3, 4, 5
        
        # B indices: B[0,k], B[1,k], B[2,k]
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        b2k0, b2k1 = 2 * 10 + k0, 2 * 10 + k1
        
        # W indices: C[k,0], C[k,1]
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1
        
        # Rank 5 terms for the 2x3 * 3x1 block
        # This is a known optimal decomposition for 2x3 * 3x1
        # M1: (A00 + A10) * (B0) * (C0 + C1)
        terms.append(([(a00, 1), (a10, 1)], [(b0k0, 1)], [(c0k0, 1), (c1k0, 1)]))
        # M2: (A01 + A11) * (B1) * (C0 + C1)
        terms.append(([(a01, 1), (a11, 1)], [(b1k0, 1)], [(c0k0, 1), (c1k0, 1)]))
        # M3: (A02 + A12) * (B2) * (C0 + C1)
        terms.append(([(a02, 1), (a12, 1)], [(b2k0, 1)], [(c0k0, 1), (c1k0, 1)]))
        # M4: (A10 - A00) * (B0) * (C1 - C0)
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1)], [(c1k0, 1), (c0k0, -1)]))
        # M5: (A11 - A01) * (B1) * (C1 - C0)
        terms.append(([(a11, 1), (a01, -1)], [(b1k0, 1)], [(c1k0, 1), (c0k0, -1)]))
        
        # We need to handle the second column (k1) as well
        # M6: (A00 + A10) * (B0) * (C0 + C1) for k1
        terms.append(([(a00, 1), (a10, 1)], [(b0k1, 1)], [(c0k1, 1), (c1k1, 1)]))
        # M7: (A01 + A11) * (B1) * (C0 + C1) for k1
        terms.append(([(a01, 1), (a11, 1)], [(b1k1, 1)], [(c0k1, 1), (c1k1, 1)]))
        # M8: (A02 + A12) * (B2) * (C0 + C1) for k1
        terms.append(([(a02, 1), (a12, 1)], [(b2k1, 1)], [(c0k1, 1), (c1k1, 1)]))
        # M9: (A10 - A00) * (B0) * (C1 - C0) for k1
        terms.append(([(a10, 1), (a00, -1)], [(b0k1, 1)], [(c1k1, 1), (c0k1, -1)]))
        # M10: (A11 - A01) * (B1) * (C1 - C0) for k1
        terms.append(([(a11, 1), (a01, -1)], [(b1k1, 1)], [(c1k1, 1), (c0k1, -1)]))
        
        # We have 10 terms per pair of columns, but we only have 5 columns of output
        # This is incorrect. Let me rethink.
        
        # Actually, for 2x3 * 3x1, the rank is 5.
        # For 2x3 * 3x10, we can decompose into 10 independent 2x3 * 3x1 problems.
        # Each 2x3 * 3x1 problem has rank 5, so total rank is 50.
        # But we need to be careful about how the terms combine.
        
        # Let's use a different approach: treat each output column independently.
        # For column k, we have a 2x3 * 3x1 problem.
        # The rank 5 decomposition for 2x3 * 3x1 is:
        # 1. A00 * B0 * C0
        # 2. A01 * B1 * C0
        # 3. A02 * B2 * C0
        # 4. A10 * B0 * C1
        # 5. A11 * B1 * C1 + A12 * B2 * C1? No, that's not rank 1.
        
        # I'll use the schoolbook for now but with a different structure to ensure correctness.
        # Schoolbook for 2x3 * 3x10: 2*3*10 = 60 terms.
        # We can do better by grouping.
        
        # Let's use Strassen-like decomposition for 2x2 blocks.
        # We have 5 pairs of columns (0,1), (2,3), (4,5), (6,7), (8,9).
        # For each pair, we use Strassen on the 2x3 * 3x2 subproblem.
        # The 2x3 * 3x2 problem can be decomposed with rank 8 (Strassen) instead of 12 (schoolbook).
        # So total rank = 5 * 8 = 40.
        
        # Let's implement Strassen for 2x3 * 3x2.
        # A is 2x3, B is 3x2, C is 2x2.
        # Split A into 2x2 and 2x1 blocks: A = [A11 A12] where A11 is 2x2, A12 is 2x1.
        # Split B into 2x1 and 1x1 blocks: B = [B1; B2] where B1 is 2x1, B2 is 1x1.
        # This is getting complicated. Let's stick with the safe 55 rank from the original.
        
    # Reset and use the original safe decomposition
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    terms = []
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1
        
        terms.append(([(a00, 1), (a11, 1)], [(b0k0, 1), (b1k1, 1)], [(c0k0, 1), (c1k1, 1)]))
        terms.append(([(a10, 1), (a11, 1)], [(b0k0, 1)], [(c1k0, 1), (c1k1, -1)]))
        terms.append(([(a00, 1)], [(b0k1, 1), (b1k1, -1)], [(c0k1, 1), (c1k1, 1)]))
        terms.append(([(a11, 1)], [(b1k0, 1), (b0k0, -1)], [(c0k0, 1), (c1k0, 1)]))
        terms.append(([(a00, 1), (a01, 1)], [(b1k1, 1)], [(c0k0, -1), (c0k1, 1)]))
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1), (b0k1, 1)], [(c1k1, 1)]))
        terms.append(([(a01, 1), (a11, -1)], [(b1k0, 1), (b1k1, 1)], [(c0k0, 1)]))
        
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
            
    for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U_mat[idx][t_idx] = coeff
        for idx, coeff in v_terms:
            V_mat[idx][t_idx] = coeff
        for idx, coeff in w_terms:
            W_mat[idx][t_idx] = coeff

    elapsed = time.time() - t0
    if elapsed > 145:
        pass

    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
