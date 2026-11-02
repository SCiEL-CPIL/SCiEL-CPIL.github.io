import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen (rank 7) on 2x2 blocks of 3x3 submatrices.
    # Each 3x3 multiplication uses schoolbook (rank 9).
    # Total rank = 7 * 9 = 63.
    
    # Strassen indices: s in 0..6, (i, k) block positions where C[i,k] gets contribution from P_s
    # Strassen formula:
    # P0 = (A00+A11)*(B00+B11)
    # P1 = (A10+A11)*B00
    # P2 = A00*(B01-B11)
    # P3 = A11*(B10-B00)
    # P4 = (A00+A01)*B11
    # P5 = (A10-A00)*(B00+B10)
    # P6 = (A01-A11)*(B01+B11)
    #
    # C00 = P0 + P3 - P5 - P6
    # C01 = P2 + P4
    # C10 = P1 + P5
    # C11 = P0 - P1 + P2 + P4
    #
    # So for each s, the (i,k) blocks that get contribution and with what coefficient:
    strassen_coeffs = {
        0: [(0,0,1), (1,1,1)],      # P0
        1: [(1,0,1), (1,1,-1)],     # P1
        2: [(0,1,1), (1,1,1)],      # P2
        3: [(0,0,1)],               # P3
        4: [(0,1,1), (1,1,1)],      # P4
        5: [(0,0,-1), (1,0,1)],     # P5
        6: [(0,0,-1), (0,1,1)],     # P6
    }
    
    # Schoolbook for 3x3: rank 9, terms indexed by (a,b,c) in 0..2
    # U3[a*3+b, t] = 1 if t corresponds to (a,b,c)
    # V3[b*3+c, t] = 1 if t corresponds to (a,b,c)
    # W3[c*3+a, t] = 1 if t corresponds to (a,b,c)
    # t = a*9 + b*3 + c
    
    rank = 7 * 9
    U = [[0]*rank for _ in range(n*m)]  # 36 x 63
    V = [[0]*rank for _ in range(m*p)]  # 36 x 63
    W = [[0]*rank for _ in range(p*n)]  # 36 x 63
    
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    t = a*9 + b*3 + c
                    col = s*9 + t
                    
                    # Determine which (I,J) blocks of A and (J,K) blocks of B are involved in P_s
                    # We need to figure out the linear combination for each P_s.
                    # Let's define the A and B block coefficients for each s.
                    #
                    # For Strassen, each P_s is a product of a linear combination of A-blocks 
                    # and a linear combination of B-blocks.
                    # Since we're using schoolbook within each 3x3, we need to handle the 
                    # linear combinations at the block level.
                    #
                    # Actually, the standard Strassen decomposition works on matrices, and 
                    # the linear combinations are at the entry level. When we partition into
                    # 3x3 blocks, the linear combinations apply to entire blocks.
                    #
                    # So for P_s = (linear_comb_of_A_blocks) * (linear_comb_of_B_blocks),
                    # the U matrix gets the A-block coefficients, V gets B-block coefficients,
                    # and W gets the C-block assembly coefficients.
                    #
                    # But wait - in the tensor decomposition, U, V, W are such that:
                    # C[i,k] = sum_s sum_{a,b,c} U[i*6+j, s*9+t] * V[j*6+k, s*9+t] * W[k*6+i, s*9+t]
                    # where i,j are row/col of A (global), j,k are row/col of B (global),
                    # k,i are row/col of C (global).
                    #
                    # With 3x3 blocks:
                    # Global A has blocks A[i,j] for i,j in {0,1} (each 3x3)
                    # Global B has blocks B[j,k] for j,k in {0,1} (each 3x3)
                    # Global C has blocks C[i,k] for i,k in {0,1} (each 3x3)
                    #
                    # For a given Strassen term s and schoolbook term (a,b,c):
                    # U row (I,J) where I=i*3+a, J=j*3+b:
                    #   Gets coefficient alpha_{s,i,j} if (i,j) block of A is used in P_s's A-part
                    # V row (J,K) where J=j*3+b, K=k*3+c:
                    #   Gets coefficient beta_{s,j,k} if (j,k) block of B is used in P_s's B-part
                    # W row (K,I) where K=k*3+c, I=i*3+a:
                    #   Gets coefficient gamma_{s,i,k} if C[i,k] gets P_s with that coefficient
                    
                    # A-block coefficients for each s:
                    A_coeffs = {
                        0: {(0,0):1, (1,1):1},
                        1: {(1,0):1, (1,1):1},
                        2: {(0,0):1},
                        3: {(1,1):1},
                        4: {(0,0):1, (0,1):1},
                        5: {(1,0):1, (0,0):-1},
                        6: {(0,1):1, (1,1):-1},
                    }
                    # B-block coefficients for each s:
                    B_coeffs = {
                        0: {(0,0):1, (1,1):1},
                        1: {(0,0):1},
                        2: {(0,1):1, (1,1):-1},
                        3: {(1,0):1, (0,0):-1},
                        4: {(1,1):1},
                        5: {(0,0):1, (1,0):1},
                        6: {(0,1):1, (1,1):1},
                    }
                    
                    a_coeff = A_coeffs[s]
                    b_coeff = B_coeffs[s]
                    c_coeff = strassen_coeffs[s]
                    
                    # Fill U: for each A block (i,j) with nonzero coeff
                    for (bi, bj), alpha in a_coeff.items():
                        for ia in range(3):
                            for jb in range(3):
                                if ia == a and jb == b:
                                    I = bi*3 + ia
                                    J = bj*3 + jb
                                    U[I*m + J][col] += alpha
                    
                    # Fill V: for each B block (j,k) with nonzero coeff
                    for (bj, bk), beta in b_coeff.items():
                        for jv in range(3):
                            for kw in range(3):
                                if jv == b and kw == c:
                                    J = bj*3 + jv
                                    K = bk*3 + kw
                                    V[J*p + K][col] += beta
                    
                    # Fill W: for each C block (i,k) with nonzero coeff
                    for (ci, ck), gamma in c_coeff:
                        for ka in range(3):
                            for ia2 in range(3):
                                if ka == c and ia2 == a:
                                    K = ck*3 + ka
                                    I = ci*3 + ia2
                                    W[K*n + I][col] += gamma
    
    return {"U": U, "V": V, "W": W}
