import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition: 6x6 as 2x2 blocks of 3x3.
    # Outer level: 2x2 matrix multiplication using Strassen (rank 7).
    # Inner level: 3x3 matrix multiplication using Schoolbook (rank 27).
    # Total rank: 7 * 27 = 189.
    
    # Strassen coefficients for 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Indices for 2x2 blocks: 0,1
    # U_outer[r, I, J] where I is row block, J is col block of A
    # V_outer[r, J, K] where J is row block, K is col block of B
    # W_outer[r, I, K] where I is row block, K is col block of C
    
    U_outer = np.zeros((7, 2, 2))
    V_outer = np.zeros((7, 2, 2))
    W_outer = np.zeros((7, 2, 2))
    
    # r=0 (M1)
    U_outer[0, 0, 0] = 1; U_outer[0, 1, 1] = 1
    V_outer[0, 0, 0] = 1; V_outer[0, 1, 1] = 1
    W_outer[0, 0, 0] = 1; W_outer[0, 0, 1] = 1; W_outer[0, 1, 0] = 1; W_outer[0, 1, 1] = 1
    
    # r=1 (M2)
    U_outer[1, 1, 0] = 1; U_outer[1, 1, 1] = 1
    V_outer[1, 0, 0] = 1
    W_outer[1, 1, 0] = 1; W_outer[1, 1, 1] = 1
    
    # r=2 (M3)
    U_outer[2, 0, 0] = 1
    V_outer[2, 0, 1] = 1; V_outer[2, 1, 1] = -1
    W_outer[2, 0, 1] = 1; W_outer[2, 1, 1] = 1
    
    # r=3 (M4)
    U_outer[3, 1, 1] = 1
    V_outer[3, 1, 0] = 1; V_outer[3, 0, 0] = -1
    W_outer[3, 0, 0] = 1; W_outer[3, 1, 0] = 1
    
    # r=4 (M5)
    U_outer[4, 0, 0] = 1; U_outer[4, 0, 1] = 1
    V_outer[4, 1, 1] = 1
    W_outer[4, 0, 0] = 1; W_outer[4, 0, 1] = 1
    
    # r=5 (M6)
    U_outer[5, 1, 0] = 1; U_outer[5, 0, 0] = -1
    V_outer[5, 0, 0] = 1; V_outer[5, 0, 1] = 1
    W_outer[5, 0, 1] = 1; W_outer[5, 1, 1] = 1
    
    # r=6 (M7)
    U_outer[6, 0, 1] = 1; U_outer[6, 1, 1] = -1
    V_outer[6, 1, 0] = 1; V_outer[6, 1, 1] = 1
    W_outer[6, 0, 1] = 1; W_outer[6, 1, 1] = 1

    # Schoolbook coefficients for 3x3
    # C_ij = sum_k A_ik B_kj
    # Rank 27
    U_inner = np.zeros((27, 3, 3))
    V_inner = np.zeros((27, 3, 3))
    W_inner = np.zeros((27, 3, 3))
    
    s = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_inner[s, i, j] = 1
                V_inner[s, k, j] = 1 # Wait, B index is k,j? B is 3x3. B[k,j]
                # In matrix mult C = A B, C[i,j] = sum_k A[i,k] B[k,j].
                # My V_inner definition: V_inner[s, row, col] corresponds to B[row, col].
                # So V_inner[s, k, j] = 1.
                W_inner[s, i, j] = 1
                s += 1

    # Combine to form U, V, W for 6x6
    # Total rank R = 7 * 27 = 189
    R = 189
    U = np.zeros((36, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    
    # Map global indices
    # A global index (r, c) -> Block (I, J) where I=r//3, J=c//3. Inner (i, j) where i=r%3, j=c%3.
    # Same for B and C.
    
    for r_o in range(7):
        for s_i in range(27):
            idx = r_o * 27 + s_i
            
            # U contributions
            for I in range(2):
                for J in range(2):
                    u_coeff = U_outer[r_o, I, J]
                    if u_coeff != 0:
                        for i in range(3):
                            for j in range(3):
                                u_inner_coeff = U_inner[s_i, I, J] # Wait, U_inner is 3x3.
                                # The inner index corresponds to the position within the 3x3 block.
                                # So we need to sum over the inner indices?
                                # No, the tensor product is element-wise.
                                # U_total[(I,i), (J,j), idx] = U_outer[r_o, I, J] * U_inner[s_i, i, j]
                                # My U_inner was defined as U_inner[s, i, j] = 1 for specific i,j,k.
                                # Actually, U_inner[s, i, j] is 1 if the term uses A[i,j]?
                                # In schoolbook, term s corresponds to (i,j,k).
                                # U_inner[s, i, k] = 1.
                                # Let's redefine U_inner, V_inner, W_inner properly.
                                
                                # U_inner[s, i, k] = 1
                                # V_inner[s, k, j] = 1
                                # W_inner[s, i, j] = 1
                                
                                pass # Logic will be fixed in final code

            # V contributions
            for J in range(2):
                for K in range(2):
                    v_coeff = V_outer[r_o, J, K]
                    if v_coeff != 0:
                        for k in range(3):
                            for j in range(3):
                                v_inner_coeff = V_inner[s_i, k, j]
                                if v_inner_coeff != 0:
                                    # Global B index: (J, k) row, (K, j) col
                                    # Row = J*3 + k, Col = K*3 + j
                                    # V[row, col] += v_coeff * v_inner_coeff
                                    V[(J*3+k)*6 + (K*3+j), idx] += v_coeff * v_inner_coeff

            # W contributions
            for I in range(2):
                for K in range(2):
                    w_coeff = W_outer[r_o, I, K]
                    if w_coeff != 0:
                        for i in range(3):
                            for j in range(3):
                                w_inner_coeff = W_inner[s_i, i, j]
                                if w_inner_coeff != 0:
                                    # Global C index: (I, i) row, (K, j) col
                                    # Row = I*3 + i, Col = K*3 + j
                                    # W[row, col] += w_coeff * w_inner_coeff
                                    W[(I*3+i)*6 + (K*3+j), idx] += w_coeff * w_inner_coeff

            # U contributions (Fixed logic)
            for I in range(2):
                for J in range(2):
                    u_coeff = U_outer[r_o, I, J]
                    if u_coeff != 0:
                        for i in range(3):
                            for k in range(3):
                                u_inner_coeff = U_inner[s_i, i, k]
                                if u_inner_coeff != 0:
                                    # Global A index: (I, i) row, (J, k) col
                                    # Row = I*3 + i, Col = J*3 + k
                                    U[(I*3+i)*6 + (J*3+k), idx] += u_coeff * u_inner_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
