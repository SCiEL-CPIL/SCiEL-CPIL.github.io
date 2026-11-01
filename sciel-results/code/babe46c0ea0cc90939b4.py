import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,2,13} into M_{2,2,8} + M_{2,2,4} + M_{2,2,1}
    # Rank = 28 + 14 + 4 = 46
    
    # We will build the U, V, W matrices by concatenating the decompositions of the blocks.
    
    # Helper to get Strassen decomposition for 2x2x2
    def get_strassen_2x2():
        # Returns list of (u, v, w) where u is len 4, v is len 4, w is len 4
        # Indices for B: 0:1,1:2, 2:1,2:2 -> 0,1,2,3
        # Indices for C: 0:1,1:2, 2:1,2:2 -> 0,1,2,3
        
        terms = []
        
        # P1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1] # B11 + B22
        w = [1, 0, 0, 1] # C11 + C22
        terms.append((u, v, w))
        
        # P2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0] # B11
        w = [0, 0, 1, 0] # C21 part? No, P2 contributes to C21 and C22.
        # Let's just store the linear combination coefficients for U, V, W.
        # The final C is sum of w_i * P_i.
        # So w_i should be the vector of coefficients for C.
        
        # Let's re-evaluate.
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P7
        # C21 = P2 + P4
        # C22 = P1 - P3 + P2 - P6
        
        # So for each P_i, we define u_i, v_i, and w_i (vector of coeffs for C).
        
        t1 = ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]) # P1
        t2 = ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]) # P2 (C21, C22)
        t3 = ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0]) # P3 (C12, C22)
        t4 = ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]) # P4 (C11, C21)
        t5 = ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, -1]) # P5 (C11, C22)
        t6 = ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]) # P6 (C22)
        t7 = ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]) # P7 (C11, C12)
        
        return [t1, t2, t3, t4, t5, t6, t7]

    def get_trivial_1():
        # 2x2x1
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        terms = []
        t1 = ([1, 0, 0, 0], [1, 0], [1, 0])
        t2 = ([0, 1, 0, 0], [0, 1], [1, 0])
        t3 = ([0, 0, 1, 0], [1, 0], [0, 1])
        t4 = ([0, 0, 0, 1], [0, 1], [0, 1])
        return [t1, t2, t3, t4]

    def decompose(N):
        # Returns list of (u, v, w) for 2x2xN
        if N == 1:
            return get_trivial_1()
        if N == 2:
            return get_strassen_2x2()
        
        # Split N into N1 and N2
        # Prefer powers of 2
        if N % 2 == 0:
            N1 = N // 2
            N2 = N // 2
        else:
            N1 = N // 2
            N2 = N - N1
            
        terms1 = decompose(N1)
        terms2 = decompose(N2)
        
        all_terms = []
        
        # Shift indices for terms2
        offset_v = 2 * N1
        offset_w = 2 * N1
        
        for u, v, w in terms1:
            all_terms.append((u, v, w))
            
        for u, v, w in terms2:
            # Pad v and w
            v_new = [0] * (2 * N)
            w_new = [0] * (2 * N)
            for i in range(len(v)):
                v_new[offset_v + i] = v[i]
            for i in range(len(w)):
                w_new[offset_w + i] = w[i]
            all_terms.append((u, v_new, w_new))
            
        return all_terms

    terms = decompose(13)
    rank = len(terms)
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(26)]
    W = [[0] * rank for _ in range(26)]
    
    for col, (u, v, w) in enumerate(terms):
        for i in range(4):
            U[i][col] = u[i]
        for i in range(26):
            V[i][col] = v[i]
            W[i][col] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
