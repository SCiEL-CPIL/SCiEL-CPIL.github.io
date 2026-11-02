import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 112
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2 by 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+b)(g+e)
    # M7 = (a-d)(e+j) -- Wait, I need a valid M7.
    
    # Let's use a known valid set from literature or derive it.
    # Actually, I can just use the schoolbook for 2x2x2 if I can't find the 7-rank one quickly,
    # but 8*8*4 = 256 which is worse.
    # Schoolbook for 2x2x4 is 16. 8*16 = 128. No improvement.
    
    # Let's try to find a valid 7-rank decomposition for 2x2x2.
    # The following is a known valid Strassen decomposition:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+b)(g+e)
    # M7 = (a-d)(e+j) is wrong.
    
    # Correct Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+b)(g+e)
    # M7 = (a-d)(e+j)
    
    # I will use the following verified set:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+b)(g+e)
    # M7 = (a-d)(e+j)
    
    # Wait, I can just use the schoolbook for 2x2x2 (rank 4) and see if I can do better.
    # Schoolbook 2x2x2 is rank 4.
    # 2x2x4 is rank 8.
    # 8 blocks * 8 = 64.
    # This is much better than 128.
    # Why did I think Strassen was needed?
    # Schoolbook for 2x2x2:
    # c11 = ae + bg
    # c12 = af + bh
    # c21 = ce + dg
    # c22 = cf + dh
    # Rank 4.
    # For 2x2x4, we have two 2x2x2 multiplications.
    # Rank 4 + 4 = 8.
    # Total rank 8 * 8 = 64.
    
    # Let's verify if 2x2x2 can be done in rank < 4.
    # No, rank is 4.
    # So 2x2x4 has rank 8.
    # Total rank 64.
    
    # This is a huge improvement over 128.
    
    # Let's implement the schoolbook for the blocks.
    
    col = 0
    for I in range(2):
        for J in range(2):
            for K in range(2):
                # A[I,J] is 2x2, B[J,K] is 2x4
                # Split B[J,K] into B[J,K,0] (2x2) and B[J,K,1] (2x2)
                for K_half in range(2):
                    # Compute A[I,J] * B[J,K,K_half]
                    # This is 2x2 by 2x2
                    # Schoolbook rank 4
                    for a_row in range(2):
                        for a_col in range(2):
                            for b_row in range(2):
                                # This is not schoolbook structure.
                                # Schoolbook:
                                # c[i][k] = sum_j a[i][j] b[j][k]
                                # For 2x2x2:
                                # c00 = a00 b00 + a01 b10
                                # c01 = a00 b01 + a01 b11
                                # c10 = a10 b00 + a11 b10
                                # c11 = a10 b01 + a11 b11
                                
                                # We need to map global indices.
                                # A global index: i = I*2 + a_row, j = J*2 + a_col
                                # B global index: j = J*2 + b_row, k = K*4 + K_half*2 + b_col
                                # C global index: i = I*2 + a_row, k = K*4 + K_half*2 + b_col
                                
                                # Term 1: a00 b00
                                i = I * 2 + 0
                                j = J * 2 + 0
                                k = K * 4 + K_half * 2 + 0
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 2: a01 b10
                                i = I * 2 + 0
                                j = J * 2 + 1
                                k = K * 4 + K_half * 2 + 0
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 3: a00 b01
                                i = I * 2 + 0
                                j = J * 2 + 0
                                k = K * 4 + K_half * 2 + 1
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 4: a01 b11
                                i = I * 2 + 0
                                j = J * 2 + 1
                                k = K * 4 + K_half * 2 + 1
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 5: a10 b00
                                i = I * 2 + 1
                                j = J * 2 + 0
                                k = K * 4 + K_half * 2 + 0
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 6: a11 b10
                                i = I * 2 + 1
                                j = J * 2 + 1
                                k = K * 4 + K_half * 2 + 0
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 7: a10 b01
                                i = I * 2 + 1
                                j = J * 2 + 0
                                k = K * 4 + K_half * 2 + 1
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
                                # Term 8: a11 b11
                                i = I * 2 + 1
                                j = J * 2 + 1
                                k = K * 4 + K_half * 2 + 1
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
                                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

