"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strassen decomposition for 2x2x2 multiplication
    # 7 factors
    # Local indices: i,j in {0,1} for X (A), j,k in {0,1} for Y (B), i,k in {0,1} for Z (C)
    
    # Coefficients for U (X), V (Y), W (Z) for each of the 7 Strassen factors
    # Format: list of 7 dicts or tuples.
    # Let's use a list of (coeff_X, coeff_Y, coeff_Z) where each coeff is a 2x2 matrix.
    
    def make_mat(coeffs):
        """coeffs is a list of (i, j, val)"""
        mat = np.zeros((2, 2))
        for i, j, val in coeffs:
            mat[i, j] = val
        return mat

    # M1: X00+X11, Y00+Y11 -> Z00+Z11
    U_s1 = make_mat([(0,0,1), (1,1,1)])
    V_s1 = make_mat([(0,0,1), (1,1,1)])
    W_s1 = make_mat([(0,0,1), (1,1,1)])
    
    # M2: X10+X11, Y00 -> Z10-Z11
    U_s2 = make_mat([(1,0,1), (1,1,1)])
    V_s2 = make_mat([(0,0,1)])
    W_s2 = make_mat([(1,0,1), (1,1,-1)])
    
    # M3: X00, Y01-Y11 -> Z01+Z11
    U_s3 = make_mat([(0,0,1)])
    V_s3 = make_mat([(0,1,1), (1,1,-1)])
    W_s3 = make_mat([(0,1,1), (1,1,1)])
    
    # M4: X11, Y10-Y00 -> Z00+Z10
    U_s4 = make_mat([(1,1,1)])
    V_s4 = make_mat([(1,0,1), (0,0,-1)])
    W_s4 = make_mat([(0,0,1), (1,0,1)])
    
    # M5: X00+X01, Y11 -> -Z00+Z01
    U_s5 = make_mat([(0,0,1), (0,1,1)])
    V_s5 = make_mat([(1,1,1)])
    W_s5 = make_mat([(0,0,-1), (0,1,1)])
    
    # M6: X10-X00, Y00+Y01 -> Z11
    U_s6 = make_mat([(1,0,1), (0,0,-1)])
    V_s6 = make_mat([(0,0,1), (0,1,1)])
    W_s6 = make_mat([(1,1,1)])
    
    # M7: X01-X11, Y10+Y11 -> Z00+Z11
    U_s7 = make_mat([(0,1,1), (1,1,-1)])
    V_s7 = make_mat([(1,0,1), (1,1,1)])
    W_s7 = make_mat([(0,0,1), (1,1,1)])
    
    strassen_factors = [
        (U_s1, V_s1, W_s1),
        (U_s2, V_s2, W_s2),
        (U_s3, V_s3, W_s3),
        (U_s4, V_s4, W_s4),
        (U_s5, V_s5, W_s5),
        (U_s6, V_s6, W_s6),
        (U_s7, V_s7, W_s7)
    ]
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # r: block row of A (0,1) and B (0,1)
    # c: block col of B (0,1,2,3)
    for r in range(2):
        for c in range(4):
            for s in range(7):
                U_mat, V_mat, W_mat = strassen_factors[s]
                
                # Fill U
                # Global i = i_local
                # Global j = 2*r + j_local
                for i_l in range(2):
                    for j_l in range(2):
                        val = U_mat[i_l, j_l]
                        if val != 0:
                            u_idx = i_l * m + (2*r + j_l)
                            U[u_idx, col] = val
                
                # Fill V
                # Global j = 2*r + j_local
                # Global k = 2*c + k_local
                for j_l in range(2):
                    for k_l in range(2):
                        val = V_mat[j_l, k_l]
                        if val != 0:
                            v_idx = (2*r + j_l) * p + (2*c + k_l)
                            V[v_idx, col] = val
                
                # Fill W
                # Global k = 2*c + k_local
                # Global i = i_local
                for k_l in range(2):
                    for i_l in range(2):
                        val = W_mat[i_l, k_l]
                        if val != 0:
                            w_idx = (2*c + k_l) * n + i_l
                            W[w_idx, col] = val
                
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
