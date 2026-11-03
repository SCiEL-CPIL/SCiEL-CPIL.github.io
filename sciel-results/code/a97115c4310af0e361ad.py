import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a 2x2 block structure with 3x3 sub-blocks.
    # Strassen for 2x2 blocks: Rank 7
    # For each 3x3 block multiplication, we use a rank-19 decomposition (Laderman).
    # Total rank = 7 * 19 = 133
    
    rank = 133
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen combinations for 2x2 blocks
    # A_comb[t] is a list of (row, col, coeff) for A
    # B_comb[t] is a list of (row, col, coeff) for B
    # C_comb[t] is a list of (row, col, coeff) for C
    
    A_comb_0 = [(0,0,1), (1,1,1)]
    B_comb_0 = [(0,0,1), (1,1,1)]
    C_comb_0 = [(0,0,1), (1,1,1)]
    
    A_comb_1 = [(1,0,1), (1,1,1)]
    B_comb_1 = [(0,0,1)]
    C_comb_1 = [(1,0,1), (1,1,-1)]
    
    A_comb_2 = [(0,0,1)]
    B_comb_2 = [(0,1,1), (1,1,-1)]
    C_comb_2 = [(0,1,1), (1,1,1)]
    
    A_comb_3 = [(1,1,1)]
    B_comb_3 = [(1,0,1), (0,0,-1)]
    C_comb_3 = [(0,0,1), (1,0,1)]
    
    A_comb_4 = [(0,0,1), (0,1,1)]
    B_comb_4 = [(1,1,1)]
    C_comb_4 = [(0,0,-1), (0,1,1)]
    
    A_comb_5 = [(1,0,1), (0,0,-1)]
    B_comb_5 = [(0,0,1), (0,1,1)]
    C_comb_5 = [(1,1,1)]
    
    A_comb_6 = [(0,1,1), (1,1,-1)]
    B_comb_6 = [(1,0,1), (1,1,1)]
    C_comb_6 = [(0,0,1)]
    
    A_combs = [A_comb_0, A_comb_1, A_comb_2, A_comb_3, A_comb_4, A_comb_5, A_comb_6]
    B_combs = [B_comb_0, B_comb_1, B_comb_2, B_comb_3, B_comb_4, B_comb_5, B_comb_6]
    C_combs = [C_comb_0, C_comb_1, C_comb_2, C_comb_3, C_comb_4, C_comb_5, C_comb_6]
    
    block_size = 3
    
    # Laderman's rank-19 decomposition of 3x3 matrix multiplication
    # We need 19 sets of coefficients for U_3, V_3, W_3
    # Each set is a 3x3 matrix (flattened to 9 elements)
    # U_3[i,j,t], V_3[j,k,t], W_3[k,i,t]
    
    # The 19 tensors for 3x3 (Laderman)
    # Format: (U_flat, V_flat, W_flat) where flat is 9 elements (row-major)
    # 1. a11 b11
    T0 = (([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]))
    # 2. a22 b22
    T1 = (([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]))
    # 3. a33 b33
    T2 = (([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]))
    # 4. (a12+a21) b12
    T3 = (([0,1,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]))
    # 5. (a13+a31) b13
    T4 = (([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]))
    # 6. (a23+a32) b23
    T5 = (([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]))
    # 7. (a11+a22) b21
    T6 = (([1,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]))
    # 8. (a11+a33) b31
    T7 = (([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]))
    # 9. (a22+a33) b32
    T8 = (([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]))
    # 10. (a12+a22+a32) b11
    T9 = (([0,1,0, 0,1,0, 0,1,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]))
    # 11. (a13+a23+a33) b12
    T10 = (([0,0,1, 0,0,1, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]))
    # 12. (a11+a12+a13) b21
    T11 = (([1,1,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]))
    # 13. (a11+a21+a31) b13
    T12 = (([1,0,0, 1,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]))
    # 14. (a22+a23+a33) b21 -- Wait, this is not correct.
    # Let's use a verified set.
    
    # I will use the coefficients from the "Laderman 19" paper.
    # The 19 terms are:
    # 1. a11 b11
    # 2. a22 b22
    # 3. a33 b33
    # 4. (a12+a21) b12
    # 5. (a13+a31) b13
    # 6. (a23+a32) b23
    # 7. (a11+a22) b21
    # 8. (a11+a33) b31
    # 9. (a22+a33) b32
    # 10. (a12+a22+a32) b11
    # 11. (a13+a23+a33) b12
    # 12. (a11+a12+a13) b21
    # 13. (a11+a21+a31) b13
    # 14. (a22+a23+a33) b21 -- No, this is wrong.
    
    # I will fall back to schoolbook for 3x3 if I cannot verify Laderman.
    # Schoolbook 3x3 rank is 27.
    # Total rank 7 * 27 = 189.
    # This is the current score.
    
    # To improve, I MUST use Laderman.
    # I will use the coefficients from a trusted source.
    
    laderman_tensors = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([1,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,1,0, 0,1,0, 0,1,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,1, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([1,1,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([1,0,0, 1,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 1,1,1, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]), # Guess
        ([0,0,0, 0,0,0, 1,1,1], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]), # Guess
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0])  # Placeholder
    ]
    
    # Since I cannot guarantee the correctness of the guessed Laderman coefficients,
    # I will return the schoolbook 189 rank solution to ensure validity.
    # However, this does not improve the score.
    
    # Is there a valid improvement?
    # What if I use Strassen for 4x4 blocks?
    # 6x6 cannot be split into 4x4.
    
    # I will return the 189 rank solution.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = t * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    for rb, cb, coeff in A_combs[t]:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob][col] += coeff
                        
                    for rb, cb, coeff in B_combs[t]:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob][col] += coeff
                        
                    for rb, cb, coeff in C_combs[t]:
                        k_glob = cb * block_size + k_l
                        i_glob = rb * block_size + i_l
                        W[k_glob * n + i_glob][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
