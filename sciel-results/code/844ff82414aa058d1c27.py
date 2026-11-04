"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Helper to get 2x2x2 Strassen decomposition
    def get_strassen_222():
        # Rank 7
        # U, V, W are 4x7
        # Indices: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # a=0,0; d=1,1; e=0,0; h=1,1
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (c+d)e
        # c=1,0; d=1,1; e=0,0
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = 1 # c+d is row 1 of A. e is col 0 of B.
        # Wait, M2 contributes to C10 and C11.
        # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So W for M2 should be placed in C10 and C11 with appropriate signs.
        # But here we are just defining the base 2x2x2 decomposition for a single multiplication.
        # The "sign" and "placement" is handled when embedding into 4x4x4.
        # So W2 for M2 should just be the identity for the output of M2.
        # M2 is a 2x2 matrix.
        # So W2 should be 1s for the output indices of M2.
        # But M2 is a scalar in the Strassen context? No, M2 is a 2x2 matrix product.
        # So the "rank 7" decomposition is for 2x2x2 matrix multiplication.
        # So W2[k*2+i, t] should be 1 for the output indices.
        # For M2, the output is a 2x2 matrix.
        # So W2 should have 1s for all 4 output indices?
        # No, the rank-7 decomposition of 2x2x2 is a sum of 7 rank-1 tensors.
        # Each rank-1 tensor corresponds to a scalar multiplication.
        # So W2[k*2+i, t] is 1 if the scalar multiplication contributes to C[k,i].
        # For M1 = (a+d)(e+h), the result is a scalar.
        # But in 2x2x2 matrix mult, A and B are 2x2.
        # So M1 is (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]).
        # This is a scalar.
        # But the output C is 2x2.
        # C[0,0] = M1 + M4 - M5 + M7.
        # C[0,1] = M3 + M5.
        # C[1,0] = M2 + M4.
        # C[1,1] = M1 - M2 + M3 + M6.
        
        # So for the base 2x2x2 decomposition, we define U2, V2, W2 such that
        # sum_t U2[i,j,t] V2[j,k,t] W2[k,i,t] = C[i,k]
        
        # Let's re-verify the Strassen 2x2x2 decomposition.
        # U2[i*2+j, t], V2[j*2+k, t], W2[k*2+i, t]
        
        # t=0: M1
        U2[0, 0] = 1; U2[3, 0] = 1 # a, d
        V2[0, 0] = 1; V2[3, 0] = 1 # e, h
        W2[0, 0] = 1; W2[3, 0] = 1 # C[0,0], C[1,1] get M1
        
        # t=1: M2
        U2[2, 1] = 1; U2[3, 1] = 1 # c, d
        V2[0, 1] = 1 # e
        W2[2, 1] = 1; W2[3, 1] = 1 # C[1,0], C[1,1] get M2 (with sign in 4x4)
        # Wait, W2 for the base decomposition should just be 1 for the output indices?
        # No, the base decomposition is for the tensor M_{2,2,2}.
        # So W2[k*2+i, t] should be 1 if the term t contributes to C[k,i].
        # For M2, it contributes to C[1,0] and C[1,1].
        # So W2[2, 1] = 1, W2[3, 1] = 1.
        
        # t=2: M3
        U2[0, 2] = 1 # a
        V2[1, 2] = 1; V2[3, 2] = -1 # f, -h
        W2[1, 2] = 1; W2[3, 2] = 1 # C[0,1], C[1,1] get M3
        
        # t=3: M4
        U2[3, 3] = 1 # d
        V2[2, 3] = 1; V2[0, 3] = -1 # g, -e
        W2[0, 3] = 1; W2[2, 3] = 1 # C[0,0], C[1,0] get M4
        
        # t=4: M5
        U2[0, 4] = 1; U2[1, 4] = 1 # a, b
        V2[3, 4] = 1 # h
        W2[0, 4] = -1; W2[1, 4] = 1 # C[0,0] gets -M5, C[0,1] gets M5
        
        # t=5: M6
        U2[2, 5] = 1; U2[0, 5] = -1 # c, -a
        V2[0, 5] = 1; V2[1, 5] = 1 # e, f
        W2[3, 5] = 1 # C[1,1] gets M6
        
        # t=6: M7
        U2[1, 6] = 1; U2[3, 6] = -1 # b, -d
        V2[2, 6] = 1; V2[3, 6] = 1 # g, h
        W2[0, 6] = 1 # C[0,0] gets M7
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_222()
    
    # Define the 7 Strassen products for 4x4x4
    # Each product is a 2x2x2 multiplication.
    # We need to map the 2x2 indices to 4x4 indices.
    
    # Product 1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    # A blocks: 00, 11. B blocks: 00, 11. C blocks: 00, 11.
    
    # We will create a list of tuples: (a_blocks, b_blocks, c_blocks, sign)
    # a_blocks: list of (row_block, col_block, sign)
    # b_blocks: list of (row_block, col_block, sign)
    # c_blocks: list of (row_block, col_block, sign)
    
    products = [
        # P1
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # P2
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (1,1,-1))]),
        # P3
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1), (1,1,1))]),
        # P4
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),
        # P5
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (0,1,1))]),
        # P6
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
        # P7
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))])
    ]
    
    # Note: The structure of "a_blocks" in the tuple above is a list of lists?
    # Let's re-format for clarity.
    # Each product has:
    # - A factors: list of (br, bc, sign)
    # - B factors: list of (br, bc, sign)
    # - C factors: list of (br, bc, sign)
    
    prod_defs = [
        # P1
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # P2
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1), (1,1,-1))]),
        # P3
        ([((0,0,1))], [((0,1,1), (1,1,-1))], [((0,1,1), (1,1,1))]),
        # P4
        ([((1,1,1))], [((1,0,1), (0,0,-1))], [((0,0,1), (1,0,1))]),
        # P5
        ([((0,0,1), (0,1,1))], [((1,1,1))], [((0,0,-1), (0,1,1))]),
        # P6
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [((1,1,1))]),
        # P7
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [((0,0,1))])
    ]
    
    col = 0
    for p_idx in range(7):
        a_factors, b_factors, c_factors = prod_defs[p_idx]
        
        # For each of the 7 terms in the 2x2x2 decomposition
        for t_local in range(7):
            t_global = col + t_local
            
            # Update U
            for (ar, ac, sign) in a_factors:
                # Map 2x2 index (i,j) to 4x4 index (I,J)
                # I = ar*2 + i, J = ac*2 + j
                for i in range(2):
                    for j in range(2):
                        idx_u = (ar*2 + i) * 4 + (ac*2 + j)
                        idx_u2 = i * 2 + j
                        U[idx_u, t_global] += sign * U2[idx_u2, t_local]
            
            # Update V
            for (br, bc, sign) in b_factors:
                for j in range(2):
                    for k in range(2):
                        idx_v = (br*2 + j) * 4 + (bc*2 + k)
                        idx_v2 = j * 2 + k
                        V[idx_v, t_global] += sign * V2[idx_v2, t_local]
            
            # Update W
            for (cr, cc, sign) in c_factors:
                for k in range(2):
                    for i in range(2):
                        idx_w = (cr*2 + k) * 4 + (cc*2 + i)
                        idx_w2 = k * 2 + i
                        W[idx_w, t_global] += sign * W2[idx_w2, t_local]
                        
            col += 1
            
    # Part 2: 4x4x1 Schoolbook
    # A is 4x4, B is 4x1 (col 4 of B), C is 4x1 (col 4 of C)
    # C[i, 4] = sum_j A[i,j] B[j,4]
    # U[i*4+j, t], V[j*5+4, t], W[4*4+i, t]
    
    for i in range(4):
        for j in range(4):
            t_global = col
            U[i*4+j, t_global] = 1
            V[j*5+4, t_global] = 1
            W[4*4+i, t_global] = 1
            col += 1

    # Verification (optional, but good for debugging)
    # The rank is 65.
    
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
