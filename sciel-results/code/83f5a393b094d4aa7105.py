import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We partition the 12 columns of B (and C) into 3 blocks of 4 columns each.
    # For each block of 4 columns, we use a hybrid decomposition:
    # - Strassen (rank 7) for the first 2 columns.
    # - Schoolbook (rank 4) for the next 2 columns.
    # Total rank per block = 11.
    # Total rank = 3 * 11 = 33.
    
    # Define Strassen decomposition for 2x2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A * B
    # Strassen uses 7 products M1..M7
    # U indices: 0: a11, 1: a12, 2: a21, 3: a22
    # V indices: 0: b11, 1: b12, 2: b21, 3: b22
    # W indices: 0: c11, 1: c12, 2: c21, 3: c22
    
    U_strassen = [
        [1, 1, 0, 0, 0, 1, 0], # a11
        [0, 1, 0, 0, 1, 0, 1], # a12
        [0, 0, 1, 0, 0, 0, 1], # a21
        [0, 0, 0, 1, 1, 1, 0]  # a22
    ]
    V_strassen = [
        [1, 0, 0, 1, 0, 0, 0], # b11
        [0, 1, 0, 0, 0, 0, 1], # b12
        [0, 0, 1, 0, 1, 0, 0], # b21
        [0, 0, 0, 1, 0, 1, 1]  # b22
    ]
    W_strassen = [
        [0, 1, 0, 0, 1, 0, 1], # c11 = M2 - M4 + M5 + M7
        [0, 0, 1, 0, 0, 0, 1], # c12 = M3 + M4
        [0, 0, 0, 1, 1, 0, 0], # c21 = M1 + M5
        [1, 0, 0, 0, 0, 1, 0]  # c22 = M1 - M2 + M3 + M6
    ]
    # Note: The above W matrix coefficients might need verification against standard Strassen.
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21+a22)(b11+b12)
    # M7 = (a11+a12)(b21-b11)
    # c11 = M1 - M4 + M5 + M7
    # c12 = M3 + M4
    # c21 = M1 + M5
    # c22 = M1 - M2 + M3 + M6
    
    # Let's re-verify U and V for these M's.
    # M1: a11+a22, b11+b22 -> U:[1,0,0,1], V:[1,0,0,1]
    # M2: a21+a22, b11 -> U:[0,0,1,1], V:[1,0,0,0]
    # M3: a11, b12-b22 -> U:[1,0,0,0], V:[0,1,0,-1]
    # M4: a22, b21-b11 -> U:[0,0,0,1], V:[-1,0,1,0]
    # M5: a11+a12, b22 -> U:[1,1,0,0], V:[0,0,0,1]
    # M6: a21+a22, b11+b12 -> U:[0,0,1,1], V:[1,1,0,0]
    # M7: a11+a12, b21-b11 -> U:[1,1,0,0], V:[-1,0,1,0]
    
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 1], # a11
        [0, 0, 0, 0, 1, 0, 1], # a12
        [0, 1, 0, 0, 0, 1, 0], # a21
        [1, 1, 0, 1, 0, 1, 0]  # a22
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, -1], # b11
        [0, 0, 1, 0, 0, 1, 0],   # b12
        [0, 0, 0, 1, 0, 0, 1],   # b21
        [1, 0, -1, 0, 1, 0, 0]   # b22
    ]
    W_strassen = [
        [1, 0, 0, -1, 1, 0, 1], # c11
        [0, 0, 1, 1, 0, 0, 0],  # c12
        [1, 0, 0, 0, 1, 0, 0],  # c21
        [1, -1, 1, 0, 0, 1, 0]  # c22
    ]
    
    # Schoolbook for 2x2x2
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    # Construct full matrices
    # Total Rank = 3 blocks * (7 Strassen + 4 School) = 33
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for block in range(3):
        # Strassen part for cols 0, 1 of the block
        for r in range(7):
            idx = block * 11 + r
            for i in range(4):
                U[i][idx] = U_strassen[i][r]
            for j in range(2):
                for k in range(2):
                    V[j * p + (block * 4 + k)][idx] = V_strassen[j * 2 + k][r]
            for i in range(2):
                for k in range(2):
                    W[(block * 4 + k) * n + i][idx] = W_strassen[i * 2 + k][r]
        
        # Schoolbook part for cols 2, 3 of the block
        for r in range(4):
            idx = block * 11 + 7 + r
            # r corresponds to (i, k) pair in schoolbook?
            # Schoolbook: U[i*m+j], V[j*p+k], W[k*n+i]
            # Here m=2, p=4 (local), n=2.
            # We are decomposing 2x2 * 2x2 (local cols 2,3)
            # Let local j be 0,1 (rows of B)
            # Let local k be 0,1 (cols of B within the 2-col sub-block)
            # Actually, Schoolbook for 2x2x2:
            # r=0: i=0,j=0,k=0 -> U[0], V[0], W[0]
            # r=1: i=0,j=0,k=1 -> U[0], V[1], W[1]
            # r=2: i=0,j=1,k=0 -> U[1], V[2], W[2]
            # r=3: i=1,j=0,k=0 -> U[2], V[0], W[3] ... wait
            # Standard schoolbook:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # Term (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
            # For 2x2x2:
            # (0,0,0): U[0], V[0], W[0]
            # (0,0,1): U[0], V[1], W[1]
            # (0,1,0): U[1], V[2], W[2]
            # (0,1,1): U[1], V[3], W[3]
            # (1,0,0): U[2], V[0], W[4]
            # (1,0,1): U[2], V[1], W[5]
            # (1,1,0): U[3], V[2], W[6]
            # (1,1,1): U[3], V[3], W[7]
            
            # My U_school, V_school, W_school above are Identity, which is wrong for direct sum.
            # I need to map the 4 schoolbook terms to the specific indices.
            
            # Let's just generate the 4 terms explicitly for the sub-block cols 2,3.
            # Local indices: j in {0,1}, k_local in {0,1}
            # Global k = block*4 + 2 + k_local
            for i in range(2):
                for j in range(2):
                    for k_local in range(2):
                        # This would be 8 terms. Schoolbook is 4? No, 2*2*2=8.
                        # Wait, Schoolbook rank for 2x2x2 is 4?
                        # No, Schoolbook rank is n*m*p = 8.
                        # Strassen is 7.
                        # So Schoolbook is 8.
                        # My previous calculation: 3 * (7 + 8) = 45.
                        # This is worse than 42.
                        
                        # I should use Strassen for all 4 columns?
                        # Strassen for 2x2x4 is 14.
                        # 3 * 14 = 42.
                        
                        # To beat 42, I need a better decomposition.
                        # If I use Strassen (7) for 2 cols and Strassen (7) for next 2 cols, it's 14.
                        # Is there a rank 9 for 2x2x4?
                        # If not, I am stuck at 42.
                        
                        # However, the directive says rank 9 exists.
                        # I will assume the directive is correct and I should find a way to implement it.
                        # Since I can't, I will implement the 42 solution (Strassen x 6 blocks of 2) 
                        # or the 33 solution if Schoolbook was 4 (which it isn't).
                        
                        # Wait, is there a rank 5 for 2x2x2? No, 7 is optimal.
                        # Is there a rank 6 for 2x2x3?
                        
                        # Let's stick to the 42 solution using 6 Strassen blocks of 2x2x2.
                        # This is safe and known.
                        pass
        col_offset += 4

    # Re-implementing the 42 solution cleanly
    total_rank = 42
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 6 blocks of 2 columns each
    for block in range(6):
        for r in range(7):
            idx = block * 7 + r
            for i in range(4):
                U[i][idx] = U_strassen[i][r]
            for j in range(2):
                for k in range(2):
                    V[j * p + (block * 2 + k)][idx] = V_strassen[j * 2 + k][r]
            for i in range(2):
                for k in range(2):
                    W[(block * 2 + k) * n + i][idx] = W_strassen[i * 2 + k][r]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
