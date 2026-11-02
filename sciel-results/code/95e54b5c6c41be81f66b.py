"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # Indices: A[i,j], B[j,k], C[i,k]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {(i,j): val}
        # v_coeffs: dict {(j,k): val}
        # w_coeffs: dict {(i,k): val}
        pass # We need to assign to specific columns
        
    # The 23 products M_r.
    # We define U[:, r], V[:, r], W[:, r] for each r.
    
    # r=0
    U[0*3+0][0] = 1; U[1*3+1][0] = 1; U[2*3+2][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1; V[2*3+2][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1; W[2*3+2][0] = 1
    # This term is (A00+A11+A22)(B00+B11+B22). 
    # Contribution to C:
    # C00 += 1*1*1 = 1 (from A00 B00)
    # C00 also gets A00 B11? No, W[0,0] is 1, but V[0,1] is 0.
    # Wait, the product is sum over r of U[i,j,r] V[j,k,r] W[i,k,r].
    # For r=0:
    # U[0,0]=1, V[0,0]=1, W[0,0]=1 -> C00 += 1
    # U[0,0]=1, V[0,1]=0 ...
    # U[0,1]=0 ...
    # U[1,1]=1, V[1,1]=1, W[1,1]=1 -> C11 += 1
    # U[2,2]=1, V[2,2]=1, W[2,2]=1 -> C22 += 1
    # Cross terms?
    # U[0,0]=1, V[0,0]=1. W[0,0]=1.
    # Does it contribute to C01?
    # Term for C01: sum_r U[0,0,r] V[0,1,r] W[0,1,r].
    # For r=0: U[0,0]=1, V[0,1]=0. So 0.
    # So r=0 only contributes to diagonal C00, C11, C22?
    # Let's check C01.
    # We need C01 = A00 B01 + A01 B11 + A02 B21.
    
    # This hardcoded approach is error-prone without the full table.
    # I will use the known explicit lists for Laderman.
    
    # Let's reconstruct from a verified source.
    # The 23 terms are:
    
    terms = [
        # 1
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 2
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 3
        ({(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}),
        # 4
        ({(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}),
        # 5
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 6
        ({(1,1):1, (0,0):1}, {(1,1):1, (0,0):1}, {(1,1):1, (0,0):1}),
        # 7
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,0):1, (1,1):1, (2,2):1}), # W needs to be checked
        # Actually, the W matrix coefficients are determined by the requirement that the sum equals the identity tensor.
        # It is safer to define the U and V matrices, and then solve for W?
        # No, W is also part of the decomposition.
        
        # I will use the specific coefficients from "A 23-multiplication algorithm for 3x3 matrix multiplication"
        # I found a reliable list online in my training data.
        
        # Let's use the standard Laderman decomposition coefficients.
        # I will write them out carefully.
    ]
    
    # Since I cannot browse, I will rely on the structure of Strassen-like algorithms or known Laderman tables.
    # However, a simpler improvement is to use rank 23 if I can be sure.
    # If I am unsure, I should stick to a valid decomposition.
    # The schoolbook is rank 27.
    # Is there a rank 24 or 25 decomposition that is easier to verify?
    # Laderman is 23.
    
    # Let's try to construct the 23 terms properly.
    # I will use the following set of (U, V, W) tuples.
    # Each tuple is (u_dict, v_dict, w_dict).
    # u_dict maps (i,j) to coeff.
    # v_dict maps (j,k) to coeff.
    # w_dict maps (i,k) to coeff.
    
    decomp = [
        # 1
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 2
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 3
        ({(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}),
        # 4
        ({(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}),
        # 5
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 6
        ({(1,1):1, (0,0):1}, {(1,1):1, (0,0):1}, {(1,1):1, (0,0):1}),
        # 7
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,0):1, (1,1):1, (2,2):1}), # W is identity here?
        # Let's check term 7 contribution to C00.
        # U[0,1]=1, V[1,0]=1. W[0,0]=1. C00 += 1.
        # But C00 should be A00 B00 + A01 B10 + A02 B20.
        # Term 7 gives A01 B10. Correct.
        # But it also gives A01 B21? V[1,1]=0.
        # And A12 B21? U[1,2]=1, V[2,1]=1. W[1,1]=1. C11 += 1.
        # C11 should have A12 B21? No, C11 = A10 B01 + A11 B11 + A12 B21.
        # So A12 B21 is correct for C11.
        # And A01 B10 is correct for C00.
        # And A12 B21 is correct for C11.
        # What about C22?
        # U[2,2]=0.
        # So term 7 contributes to C00 and C11.
        
        # This suggests the W matrix for term 7 is indeed Identity?
        # Let's check if this set of 23 terms is valid.
        # I need the full list.
        
        # I will output the Schoolbook method but with a comment that it is suboptimal, 
        # OR I will try to implement a known rank 23 decomposition if I can recall it perfectly.
        # Given the risk of error in hardcoding 23 complex terms, and the directive to "Keep the program's entry-point",
        # I will stick to the Schoolbook method if I cannot guarantee the Laderman implementation.
        # However, the task asks for "stronger algorithm".
        # Rank 23 is significantly better than 27.
        
        # I will use a verified Laderman decomposition from memory/logic.
        # The 23 terms are well-documented.
        
        # Let's try to find a simpler improvement.
        # Maybe rank 24?
        # No, Laderman is 23.
        
        # I will implement the 23 terms.
    ]
    
    # I will use the following 23 terms (U, V, W)
    # Reference: "Fast Matrix Multiplication" by C. Strassen (for 2x2), extended by Laderman.
    
    # I will hardcode the 23 terms below.
    # If I make a mistake, the verification will fail.
    # To be safe, I will include a verification step in the construct function?
    # No, construct must return the matrices.
    # I will verify in my head or use a known correct set.
    
    # Here is a known correct set of 23 terms for Laderman:
    
    terms = [
        # 1
        ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 2
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 3
        ({(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}, {(1,1):1, (2,2):1}),
        # 4
        ({(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}),
        # 5
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # 6
        ({(1,1):1, (0,0):1}, {(1,1):1, (0,0):1}, {(1,1):1, (0,0):1}),
        # 7
        ({(0,1):1, (1,2):1}, {(1,0):1, (2,1):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 8
        ({(1,2):1, (2,0):1}, {(2,1):1, (0,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 9
        ({(2,0):1, (0,1):1}, {(0,2):1, (1,0):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 10
        ({(0,1):1, (1,2):1, (2,0):1}, {(1,0):1, (2,1):1, (0,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 11
        ({(0,2):1, (1,0):1}, {(2,0):1, (0,1):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 12
        ({(1,0):1, (2,1):1}, {(0,1):1, (1,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 13
        ({(2,1):1, (0,2):1}, {(1,2):1, (2,0):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 14
        ({(0,2):1, (1,0):1, (2,1):1}, {(2,0):1, (0,1):1, (1,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 15
        ({(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 16
        ({(1,1):1, (1,2):1}, {(2,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 17
        ({(2,2):1, (2,0):1}, {(0,2):1, (0,0):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 18
        ({(0,0):1, (0,1):1, (0,2):1}, {(1,0):1, (1,1):1, (1,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 19
        ({(1,0):1, (1,1):1, (1,2):1}, {(2,0):1, (2,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 20
        ({(2,0):1, (2,1):1, (2,2):1}, {(0,0):1, (0,1):1, (0,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 21
        ({(0,0):1, (1,0):1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 22
        ({(1,1):1, (2,1):1}, {(1,2):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}),
        # 23
        ({(2,2):1, (0,2):1}, {(2,0):1, (0,0):1}, {(0,0):1, (1,1):1, (2,2):1}),
    ]
    
    # This set above is a guess based on symmetry. 
    # The W matrix being Identity for all terms is unlikely to be correct for all.
    # In Laderman's algorithm, the W coefficients vary.
    
    # Given the high risk of providing an incorrect decomposition, 
    # and the fact that the current program (rank 27) is valid,
    # I will return the Schoolbook method but optimized for speed?
    # No, the score is rank.
    
    # I will stick to the Schoolbook method to ensure validity.
    # Rank 27 is valid.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
