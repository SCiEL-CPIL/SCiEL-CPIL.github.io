import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    
    # Try to implement Laderman 4x4x4 rank-17 correctly
    # Then use it for both halves of the 4x8 matrix B
    
    # Laderman 4x4x4: C = A*B where A,B,C are 4x4
    # The 17 products are:
    # P0 = (A00+A11+A22+A33)*(B00+B11+B22+B33)
    # P1 = (A01+A10+A23+A32)*(B01+B10+B23+B32)  
    # P2 = (A02+A13+A20+A31)*(B02+B13+B20+B31)
    # P3 = (A03+A12+A21+A30)*(B03+B12+B21+B30)
    # P4 = (A00+A11)*(B00+B11)
    # P5 = (A00+A22)*(B00+B22)
    # P6 = (A00+A33)*(B00+B33)
    # P7 = (A11+A22)*(B11+B22)
    # P8 = (A11+A33)*(B11+B33)
    # P9 = (A22+A33)*(B22+B33)
    # P10 = (A01+A23)*(B01+B23)
    # P11 = (A02+A31)*(B02+B31)
    # P12 = (A03+A12)*(B03+B12)
    # P13 = (A10+A32)*(B10+B32)
    # P14 = (A13+A20)*(B13+B20)
    # P15 = (A21+A30)*(B21+B30)
    # P16 = (A01+A10+A23+A32)*(B10+B01+B32+B23)
    
    # Let me verify this numerically
    def laderman_4x4x4(A, B):
        """Compute C = A*B using Laderman's 17-term decomposition"""
        C = np.zeros((4, 4))
        
        # P0
        P0 = (A[0,0]+A[1,1]+A[2,2]+A[3,3]) * (B[0,0]+B[1,1]+B[2,2]+B[3,3])
        C[0,0] += P0; C[1,1] += P0; C[2,2] += P0; C[3,3] += P0
        
        # P1
        P1 = (A[0,1]+A[1,0]+A[2,3]+A[3,2]) * (B[0,1]+B[1,0]+B[2,3]+B[3,2])
        C[0,1] += P1; C[1,0] += P1; C[2,3] += P1; C[3,2] += P1
        
        # P2
        P2 = (A[0,2]+A[1,3]+A[2,0]+A[3,1]) * (B[0,2]+B[1,3]+B[2,0]+B[3,1])
        C[0,2] += P2; C[1,3] += P2; C[2,0] += P2; C[3,1] += P2
        
        # P3
        P3 = (A[0,3]+A[1,2]+A[2,1]+A[3,0]) * (B[0,3]+B[1,2]+B[2,1]+B[3,0])
        C[0,3] += P3; C[1,2] += P3; C[2,1] += P3; C[3,0] += P3
        
        # P4
        P4 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        C[0,0] -= P4; C[1,1] -= P4
        
        # P5
        P5 = (A[0,0]+A[2,2]) * (B[0,0]+B[2,2])
        C[0,0] -= P5; C[2,2] -= P5
        
        # P6
        P6 = (A[0,0]+A[3,3]) * (B[0,0]+B[3,3])
        C[0,0] -= P6; C[3,3] -= P6
        
        # P7
        P7 = (A[1,1]+A[2,2]) * (B[1,1]+B[2,2])
        C[1,1] -= P7; C[2,2] -= P7
        
        # P8
        P8 = (A[1,1]+A[3,3]) * (B[1,1]+B[3,3])
        C[1,1] -= P8; C[3,3] -= P8
        
        # P9
        P9 = (A[2,2]+A[3,3]) * (B[2,2]+B[3,3])
        C[2,2] -= P9; C[3,3] -= P9
        
        # P10
        P10 = (A[0,1]+A[2,3]) * (B[0,1]+B[2,3])
        C[0,1] -= P10; C[2,3] -= P10
        
        # P11
        P11 = (A[0,2]+A[3,1]) * (B[0,2]+B[3,1])
        C[0,2] -= P11; C[3,1] -= P11
        
        # P12
        P12 = (A[0,3]+A[1,2]) * (B[0,3]+B[1,2])
        C[0,3] -= P12; C[1,2] -= P12
        
        # P13
        P13 = (A[1,0]+A[3,2]) * (B[1,0]+B[3,2])
        C[1,0] -= P13; C[3,2] -= P13
        
        # P14
        P14 = (A[1,3]+A[2,0]) * (B[1,3]+B[2,0])
        C[1,3] -= P14; C[2,0] -= P14
        
        # P15
        P15 = (A[2,1]+A[3,0]) * (B[2,1]+B[3,0])
        C[2,1] -= P15; C[3,0] -= P15
        
        # P16
        P16 = (A[0,1]+A[1,0]+A[2,3]+A[3,2]) * (B[1,0]+B[0,1]+B[3,2]+B[2,3])
        C[0,1] -= P16; C[1,0] -= P16; C[2,3] -= P16; C[3,2] -= P16
        
        return C
    
    # Verify
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_direct = A_test @ B_test
    C_laderman = laderman_4x4x4(A_test, B_test)
    
    if not np.allclose(C_direct, C_laderman, atol=1e-10):
        # Laderman decomposition is incorrect, fall back to Strassen
        return _strassen_fallback(n, m, p, start_time)
    
    # Laderman is correct! Use it for rank 34 (2 halves × 17 terms)
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each half (0 or 1), and each Laderman term (0-16), fill in U, V, W
    # A is 4x4, B_half is 4x4 (columns half*4 to half*4+3 of B)
    # C_half is 4x4 (columns half*4 to half*4+3 of C)
    
    for half in range(2):
        for k in range(17):
            col = half * 17 + k
            
            # Define U, V, W for this term
            U_k = np.zeros((4, 4))
            V_k = np.zeros((4, 4))
            W_k = np.zeros((4, 4))
            
            if k == 0:
                # P0: (A00+A11+A22+A33)(B00+B11+B22+B33)
                U_k[0,0]=1; U_k[1,1]=1; U_k[2,2]=1; U_k[3,3]=1
                V_k[0,0]=1; V_k[1,1]=1; V_k[2,2]=1; V_k[3,3]=1
                W_k[0,0]=1; W_k[1,1]=1; W_k[2,2]=1; W_k[3,3]=1
            elif k == 1:
                # P1: (A01+A10+A23+A32)(B01+B10+B23+B32)
                U_k[0,1]=1; U_k[1,0]=1; U_k[2,3]=1; U_k[3,2]=1
                V_k[0,1]=1; V_k[1,0]=1; V_k[2,3]=1; V_k[3,2]=1
                W_k[0,1]=1; W_k[1,0]=1; W_k[2,3]=1; W_k[3,2]=1
            elif k == 2:
                # P2: (A02+A13+A20+A31)(B02+B13+B20+B31)
                U_k[0,2]=1; U_k[1,3]=1; U_k[2,0]=1; U_k[3,1]=1
                V_k[0,2]=1; V_k[1,3]=1; V_k[2,0]=1; V_k[3,1]=1
                W_k[0,2]=1; W_k[1,3]=1; W_k[2,0]=1; W_k[3,1]=1
            elif k == 3:
                # P3: (A03+A12+A21+A30)(B03+B12+B21+B30)
                U_k[0,3]=1; U_k[1,2]=1; U_k[2,1]=1; U_k[3,0]=1
                V_k[0,3]=1; V_k[1,2]=1; V_k[2,1]=1; V_k[3,0]=1
                W_k[0,3]=1; W_k[1,2]=1; W_k[2,1]=1; W_k[3,0]=1
            elif k == 4:
                # P4: (A00+A11)(B00+B11), subtracted from C[0,0] and C[1,1]
                U_k[0,0]=1; U_k[1,1]=1
                V_k[0,0]=1; V_k[1,1]=1
                W_k[0,0]=-1; W_k[1,1]=-1
            elif k == 5:
                # P5: (A00+A22)(B00+B22), subtracted from C[0,0] and C[2,2]
                U_k[0,0]=1; U_k[2,2]=1
                V_k[0,0]=1; V_k[2,2]=1
                W_k[0,0]=-1; W_k[2,2]=-1
            elif k == 6:
                # P6: (A00+A33)(B00+B33), subtracted from C[0,0] and C[3,3]
                U_k[0,0]=1; U_k[3,3]=1
                V_k[0,0]=1; V_k[3,3]=1
                W_k[0,0]=-1; W_k[3,3]=-1
            elif k == 7:
                # P7: (A11+A22)(B11+B22), subtracted from C[1,1] and C[2,2]
                U_k[1,1]=1; U_k[2,2]=1
                V_k[1,1]=1; V_k[2,2]=1
                W_k[1,1]=-1; W_k[2,2]=-1
            elif k == 8:
                # P8: (A11+A33)(B11+B33), subtracted from C[1,1] and C[3,3]
                U_k[1,1]=1; U_k[3,3]=1
                V_k[1,1]=1; V_k[3,3]=1
                W_k[1,1]=-1; W_k[3,3]=-1
            elif k == 9:
                # P9: (A22+A33)(B22+B33), subtracted from C[2,2] and C[3,3]
                U_k[2,2]=1; U_k[3,3]=1
                V_k[2,2]=1; V_k[3,3]=1
                W_k[2,2]=-1; W_k[3,3]=-1
            elif k == 10:
                # P10: (A01+A23)(B01+B23), subtracted from C[0,1] and C[2,3]
                U_k[0,1]=1; U_k[2,3]=1
                V_k[0,1]=1; V_k[2,3]=1
                W_k[0,1]=-1; W_k[2,3]=-1
            elif k == 11:
                # P11: (A02+A31)(B02+B31), subtracted from C[0,2] and C[3,1]
                U_k[0,2]=1; U_k[3,1]=1
                V_k[0,2]=1; V_k[3,1]=1
                W_k[0,2]=-1; W_k[3,1]=-1
            elif k == 12:
                # P12: (A03+A12)(B03+B12), subtracted from C[0,3] and C[1,2]
                U_k[0,3]=1; U_k[1,2]=1
                V_k[0,3]=1; V_k[1,2]=1
                W_k[0,3]=-1; W_k[1,2]=-1
            elif k == 13:
                # P13: (A10+A32)(B10+B32), subtracted from C[1,0] and C[3,2]
                U_k[1,0]=1; U_k[3,2]=1
                V_k[1,0]=1; V_k[3,2]=1
                W_k[1,0]=-1; W_k[3,2]=-1
            elif k == 14:
                # P14: (A13+A20)(B13+B20), subtracted from C[1,3] and C[2,0]
                U_k[1,3]=1; U_k[2,0]=1
                V_k[1,3]=1; V_k[2,0]=1
                W_k[1,3]=-1; W_k[2,0]=-1
            elif k == 15:
                # P15: (A21+A30)(B21+B30), subtracted from C[2,1] and C[3,0]
                U_k[2,1]=1; U_k[3,0]=1
                V_k[2,1]=1; V_k[3,0]=1
                W_k[2,1]=-1; W_k[3,0]=-1
            elif k == 16:
                # P16: (A01+A10+A23+A32)(B10+B01+B32+B23), subtracted from C[0,1],C[1,0],C[2,3],C[3,2]
                U_k[0,1]=1; U_k[1,0]=1; U_k[2,3]=1; U_k[3,2]=1
                V_k[1,0]=1; V_k[0,1]=1; V_k[3,2]=1; V_k[2,3]=1
                W_k[0,1]=-1; W_k[1,0]=-1; W_k[2,3]=-1; W_k[3,2]=-1
            
            # Fill U: A is 4x4, so U[i*4+j, col] = U_k[i,j]
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j, col] = U_k[i, j]
            
            # Fill V: B_half is 4x4 (rows 0-3 of B, columns half*4 to half*4+3)
            # V is (m*p, rank) = (4*8, rank), indexed as V[row*8+col, rank]
            # For half h, B_half[i,j] corresponds to B[i, j + h*4]
            for i in range(4):
                for j in range(4):
                    V[i * 8 + (j + half * 4), col] = V_k[i, j]
            
            # Fill W: C_half is 4x4 (rows 0-3 of C, columns half*4 to half*4+3)
            # W is (p*n, rank) = (8*4, rank), indexed as W[k*n+i, rank] = W[col*4+row, rank]
            # For half h, C_half[i,j] corresponds to C[i, j + h*4]
            # W index: (j + half*4) * 4 + i
            for i in range(4):
                for j in range(4):
                    W[(j + half * 4) * 4 + i, col] = W_k[i, j]
    
    # Verify the decomposition
    A_rand = np.random.randn(16).reshape(4, 4)
    B_rand = np.random.randn(32).reshape(4, 8)
    
    # Compute C using the decomposition
    C_decomp = np.zeros((4, 8))
    for i in range(4):
        for j in range(8):
            total = 0.0
            for k in range(rank):
                total += U[i * 4 + (j % 4), k] * V[i * 8 + j, k] * W[j * 4 + i, k]
            C_decomp[i, j] = total
    
    C_direct = A_rand @ B_rand
    
    if not np.allclose(C_decomp, C_direct, atol=1e-10):
        # Decomposition is incorrect, fall back
        return _strassen_fallback(n, m, p, start_time)
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_fallback(n, m, p, start_time):
    """Fallback to Strassen-based rank-98 decomposition"""
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
