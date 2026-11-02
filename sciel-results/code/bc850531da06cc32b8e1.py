"""Program for matrix multiplication tensor decomposition 4x4 x 4x6."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def strassen_2x2_2x2():
        """
        Returns U, V, W for 2x2 x 2x2 -> 2x2 multiplication with rank 7.
        U: (4, 7) where row index is i*2+j
        V: (4, 7) where row index is j*2+k
        W: (4, 7) where row index is k*2+i
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Indices:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00 -> C10, C11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B01-B11) -> C01, C11
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00) -> C00, C10
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11 -> C00, C01
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def rank5_2x2x3():
        """
        Returns U, V, W for 2x2 x 2x3 -> 2x3 multiplication with rank 5.
        This is a known optimal decomposition.
        U: (4, 5) where row index is i*2+j
        V: (6, 5) where row index is j*3+k
        W: (6, 5) where row index is k*2+i
        
        Let's verify a standard rank 5 construction for 2x2 x 2x3.
        A = [[a, b], [c, d]]
        B = [[e, f, g], [h, i, j]]
        C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
        
        We need 5 terms.
        T1 = (a+b)(e+h) -> contributes to C00, C01, C02? No, (a+b)(e+h) = ae+ah+be+bh.
        This mixes C00 (ae+bh) and cross terms.
        
        Let's use a known rank 5 scheme for 2x2x3.
        Actually, the rank of M_{2,2,3} is 5.
        
        Construction:
        Let A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
        
        Product 1: (a+b)(e+h) = ae + ah + be + bh
        Product 2: (a-b)(e-h) = ae - ah - be + bh
        Sum 1+2 = 2(ae + bh) -> C00
        Diff 1-2 = 2(ah + be) -> Cross terms
        
        This doesn't isolate C00 easily without affecting others.
        
        Let's look for a standard explicit decomposition.
        From literature, M_{2,2,3} has rank 5.
        
        One valid decomposition:
        U (4x5), V (6x5), W (6x5)
        
        Indices:
        A: 00, 01, 10, 11
        B: 00, 01, 02, 10, 11, 12
        C: 00, 01, 02, 10, 11, 12
        
        Let's try to construct it via linear algebra if needed, but I will hardcode a known valid one.
        If I can't recall the exact coefficients, I will use the schoolbook rank 6 for the 2x2x3 part,
        which gives total rank 6*7=42. Still better than 96.
        But let's try to find rank 5.
        
        Actually, I can just use the schoolbook for 2x2x3 (rank 6) and Strassen for 2x2x2 (rank 7).
        Total rank = 6 * 7 = 42.
        This is a safe, guaranteed valid construction.
        
        Let's check if rank 5 is easy to hardcode.
        Reference: "The rank of the 2x2x3 matrix multiplication tensor is 5."
        
        Let's use the rank 6 schoolbook for 2x2x3 to be safe and simple, ensuring correctness first.
        Rank 6 * Rank 7 = 42.
        """
        rank = 6
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2+j, col] = 1.0
                    V[j*3+k, col] = 1.0
                    W[k*2+i, col] = 1.0
                    col += 1
        return U, V, W

    def kron_decomposition(U1, V1, W1, U2, V2, W2):
        """
        Combine two tensor decompositions using Kronecker product.
        T1 = (n1, m1, p1) with rank r1
        T2 = (n2, m2, p2) with rank r2
        
        Result T = (n1*n2, m1*m2, p1*p2) with rank r1*r2
        
        U_total: (n1*n2*m1*m2, r1*r2)
        V_total: (m1*m2*p1*p2, r1*r2)
        W_total: (p1*p2*n1*n2, r1*r2)
        
        Index mapping:
        U row index: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
        V row index: (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
        W row index: (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
        
        Wait, the standard Kronecker product of tensors T1 and T2:
        T[i1,i2, j1,j2, k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
        
        Decomposition:
        T = sum_{c1,c2} (U1[i1,j1,c1] U2[i2,j2,c2]) (V1[j1,k1,c1] V2[j2,k2,c2]) (W1[k1,i1,c1] W2[k2,i2,c2])
        
        So U_total row (i1,i2,j1,j2) col (c1,c2) = U1[i1,j1,c1] * U2[i2,j2,c2]
        V_total row (j1,j2,k1,k2) col (c1,c2) = V1[j1,k1,c1] * V2[j2,k2,c2]
        W_total row (k1,k2,i1,i2) col (c1,c2) = W1[k1,i1,c1] * W2[k2,i2,c2]
        
        We need to map these to 1D indices.
        """
        n1, m1, p1 = U1.shape[0]//m1, U1.shape[0], V1.shape[0]//p1 # This is getting messy with shapes
        # Let's infer dimensions from shapes
        # U1 shape: (n1*m1, r1)
        # V1 shape: (m1*p1, r1)
        # W1 shape: (p1*n1, r1)
        
        # We know n1=2, m1=2, p1=3 for the first part
        # And n2=2, m2=2, p2=2 for the second part
        
        # Let's hardcode the dimensions for clarity
        n1, m1, p1 = 2, 2, 3
        n2, m2, p2 = 2, 2, 2
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r_total = r1 * r2
        
        n_total = n1 * n2
        m_total = m1 * m2
        p_total = p1 * p2
        
        U_total = np.zeros((n_total * m_total, r_total))
        V_total = np.zeros((m_total * p_total, r_total))
        W_total = np.zeros((p_total * n_total, r_total))
        
        for i1 in range(n1):
            for j1 in range(m1):
                for k1 in range(p1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                # Calculate indices
                                idx_U_row = (i1 * n2 + i2) * m_total + (j1 * m2 + j2)
                                idx_V_row = (j1 * m2 + j2) * p_total + (k1 * p2 + k2)
                                idx_W_row = (k1 * p2 + k2) * n_total + (i1 * n2 + i2)
                                
                                for c1 in range(r1):
                                    u1_val = U1[i1 * m1 + j1, c1]
                                    v1_val = V1[j1 * p1 + k1, c1]
                                    w1_val = W1[k1 * n1 + i1, c1]
                                    
                                    for c2 in range(r2):
                                        u2_val = U2[i2 * m2 + j2, c2]
                                        v2_val = V2[j2 * p2 + k2, c2]
                                        w2_val = W2[k2 * n2 + i2, c2]
                                        
                                        u_val = u1_val * u2_val
                                        v_val = v1_val * v2_val
                                        w_val = w1_val * w2_val
                                        
                                        idx_col = c1 * r2 + c2
                                        
                                        U_total[idx_U_row, idx_col] += u_val
                                        V_total[idx_V_row, idx_col] += v_val
                                        W_total[idx_W_row, idx_col] += w_val
                                        
        return U_total, V_total, W_total

    def validate(U, V, W, rank):
        """
        Check if U, V, W form a valid tensor decomposition.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=1e-10)

    # Get decompositions
    U1, V1, W1 = rank5_2x2x3() # Actually rank 6 schoolbook
    U2, V2, W2 = strassen_2x2_2x2()
    
    # Combine
    U_total, V_total, W_total = kron_decomposition(U1, V1, W1, U2, V2, W2)
    
    rank = U_total.shape[1]
    
    # Validate
    if validate(U_total, V_total, W_total, rank):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0.0] * best_rank for _ in range(n * m)]
        V = [[0.0] * best_rank for _ in range(m * p)]
        W = [[0.0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
