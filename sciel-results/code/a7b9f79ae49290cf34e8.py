"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen algorithm (2 levels).
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 5.0
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Define helper to flatten 2x2 block index to global index
    # A is 4x4. Block (br, bc) contains A[br*2..br*2+1, bc*2..bc*2+1]
    # Global index for A[r, c] is r*4 + c
    
    # 1. Define CoeffA[k][i][j] and CoeffB[k][i][j]
    # k in 0..6 (outer Strassen terms)
    # i, j in 0..1 (local 2x2 block indices)
    # Each is a 4x4 matrix of coefficients for the 16 global entries
    
    # Initialize as zeros
    CoeffA = np.zeros((7, 2, 2, 4, 4), dtype=int)
    CoeffB = np.zeros((7, 2, 2, 4, 4), dtype=int)
    
    def add_block_to_coeff(coeff, br, bc, val):
        """Add val to the 2x2 sub-block (br, bc) of the 4x4 coeff matrix"""
        coeff[br*2:br*2+2, bc*2:bc*2+2] += val

    # Outer Strassen Terms
    # k=0: M1 = (A00+A11)(B00+B11)
    for i in range(2):
        for j in range(2):
            # X1[i][j] = A00[i][j] + A11[i][j]
            add_block_to_coeff(CoeffA[0, i, j], 0, 0, 1)
            add_block_to_coeff(CoeffA[0, i, j], 1, 1, 1)
            
            # Y1[i][j] = B00[i][j] + B11[i][j]
            add_block_to_coeff(CoeffB[0, i, j], 0, 0, 1)
            add_block_to_coeff(CoeffB[0, i, j], 1, 1, 1)

    # k=1: M2 = (A10+A11)B00
    for i in range(2):
        for j in range(2):
            # X2[i][j] = A10[i][j] + A11[i][j]
            add_block_to_coeff(CoeffA[1, i, j], 1, 0, 1)
            add_block_to_coeff(CoeffA[1, i, j], 1, 1, 1)
            
            # Y2[i][j] = B00[i][j]
            add_block_to_coeff(CoeffB[1, i, j], 0, 0, 1)

    # k=2: M3 = A00(B01-B11)
    for i in range(2):
        for j in range(2):
            # X3[i][j] = A00[i][j]
            add_block_to_coeff(CoeffA[2, i, j], 0, 0, 1)
            
            # Y3[i][j] = B01[i][j] - B11[i][j]
            add_block_to_coeff(CoeffB[2, i, j], 0, 1, 1)
            add_block_to_coeff(CoeffB[2, i, j], 1, 1, -1)

    # k=3: M4 = A11(B10-B00)
    for i in range(2):
        for j in range(2):
            # X4[i][j] = A11[i][j]
            add_block_to_coeff(CoeffA[3, i, j], 1, 1, 1)
            
            # Y4[i][j] = B10[i][j] - B00[i][j]
            add_block_to_coeff(CoeffB[3, i, j], 1, 0, 1)
            add_block_to_coeff(CoeffB[3, i, j], 0, 0, -1)

    # k=4: M5 = (A00+A01)B11
    for i in range(2):
        for j in range(2):
            # X5[i][j] = A00[i][j] + A01[i][j]
            add_block_to_coeff(CoeffA[4, i, j], 0, 0, 1)
            add_block_to_coeff(CoeffA[4, i, j], 0, 1, 1)
            
            # Y5[i][j] = B11[i][j]
            add_block_to_coeff(CoeffB[4, i, j], 1, 1, 1)

    # k=5: M6 = (A10-A00)(B00+B01)
    for i in range(2):
        for j in range(2):
            # X6[i][j] = A10[i][j] - A00[i][j]
            add_block_to_coeff(CoeffA[5, i, j], 1, 0, 1)
            add_block_to_coeff(CoeffA[5, i, j], 0, 0, -1)
            
            # Y6[i][j] = B00[i][j] + B01[i][j]
            add_block_to_coeff(CoeffB[5, i, j], 0, 0, 1)
            add_block_to_coeff(CoeffB[5, i, j], 0, 1, 1)

    # k=6: M7 = (A01-A11)(B10+B11)
    for i in range(2):
        for j in range(2):
            # X7[i][j] = A01[i][j] - A11[i][j]
            add_block_to_coeff(CoeffA[6, i, j], 0, 1, 1)
            add_block_to_coeff(CoeffA[6, i, j], 1, 1, -1)
            
            # Y7[i][j] = B10[i][j] + B11[i][j]
            add_block_to_coeff(CoeffB[6, i, j], 1, 0, 1)
            add_block_to_coeff(CoeffB[6, i, j], 1, 1, 1)

    # 2. Define Inner Strassen Coefficients
    # For each inner term l (0..6), we have:
    # Left Factor: sum of X[i][j] with coeff alpha_L[l][i][j]
    # Right Factor: sum of Y[i][j] with coeff alpha_R[l][i][j]
    # Contribution to Z[i][j] with coeff gamma[l][i][j]
    
    # Standard Strassen:
    # m1 = (X00+X11)(Y00+Y11)
    # m2 = (X10+X11)Y00
    # m3 = X00(Y01-Y11)
    # m4 = X11(Y10-Y00)
    # m5 = (X00+X01)Y11
    # m6 = (X10-X00)(Y00+Y01)
    # m7 = (X01-X11)(Y10+Y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    alpha_L = np.zeros((7, 2, 2), dtype=int)
    alpha_R = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    
    # l=0 (m1)
    alpha_L[0, 0, 0] = 1; alpha_L[0, 1, 1] = 1
    alpha_R[0, 0, 0] = 1; alpha_R[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    
    # l=1 (m2)
    alpha_L[1, 1, 0] = 1; alpha_L[1, 1, 1] = 1
    alpha_R[1, 0, 0] = 1
    gamma[1, 1, 0] = 1; gamma[1, 1, 1] = -1
    
    # l=2 (m3)
    alpha_L[2, 0, 0] = 1
    alpha_R[2, 0, 1] = 1; alpha_R[2, 1, 1] = -1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = 1
    
    # l=3 (m4)
    alpha_L[3, 1, 1] = 1
    alpha_R[3, 1, 0] = 1; alpha_R[3, 0, 0] = -1
    gamma[3, 0, 0] = 1; gamma[3, 1, 0] = 1
    
    # l=4 (m5)
    alpha_L[4, 0, 0] = 1; alpha_L[4, 0, 1] = 1
    alpha_R[4, 1, 1] = 1
    gamma[4, 0, 0] = -1; gamma[4, 0, 1] = 1
    
    # l=5 (m6)
    alpha_L[5, 1, 0] = 1; alpha_L[5, 0, 0] = -1
    alpha_R[5, 0, 0] = 1; alpha_R[5, 0, 1] = 1
    gamma[5, 1, 1] = 1
    
    # l=6 (m7)
    alpha_L[6, 0, 1] = 1; alpha_L[6, 1, 1] = -1
    alpha_R[6, 1, 0] = 1; alpha_R[6, 1, 1] = 1
    gamma[6, 0, 0] = 1

    # 3. Define Outer Reconstruction Coefficients
    # Delta[k][br][bc] is the coefficient of block M_k in C_block[br][bc]
    # C00 = M1 + M4 - M5 + M7  => k=0: (0,0)=1, k=3: (0,0)=1, k=4: (0,0)=-1, k=6: (0,0)=1
    # C01 = M3 + M5             => k=2: (0,1)=1, k=4: (0,1)=1
    # C10 = M2 + M4             => k=1: (1,0)=1, k=3: (1,0)=1
    # C11 = M1 - M2 + M3 + M6   => k=0: (1,1)=1, k=1: (1,1)=-1, k=2: (1,1)=1, k=5: (1,1)=1
    
    Delta = np.zeros((7, 2, 2), dtype=int)
    
    # k=0 (M1)
    Delta[0, 0, 0] = 1
    Delta[0, 1, 1] = 1
    
    # k=1 (M2)
    Delta[1, 1, 0] = 1
    Delta[1, 1, 1] = -1
    
    # k=2 (M3)
    Delta[2, 0, 1] = 1
    Delta[2, 1, 1] = 1
    
    # k=3 (M4)
    Delta[3, 0, 0] = 1
    Delta[3, 1, 0] = 1
    
    # k=4 (M5)
    Delta[4, 0, 0] = -1
    Delta[4, 0, 1] = 1
    
    # k=5 (M6)
    Delta[5, 1, 1] = 1
    
    # k=6 (M7)
    Delta[6, 0, 0] = 1

    # 4. Construct U, V, W
    for k in range(7):
        for l in range(7):
            t = k * 7 + l
            
            # Calculate U column
            # U[:, t] = sum_{i,j} alpha_L[l][i][j] * vec(CoeffA[k][i][j])
            u_vec = np.zeros(16, dtype=int)
            for i in range(2):
                for j in range(2):
                    if alpha_L[l][i][j] != 0:
                        # CoeffA[k][i][j] is 4x4
                        # Flatten row-major to 16
                        u_vec += alpha_L[l][i][j] * CoeffA[k][i][j].flatten()
            U[:, t] = u_vec
            
            # Calculate V column
            # V[:, t] = sum_{i,j} alpha_R[l][i][j] * vec(CoeffB[k][i][j])
            v_vec = np.zeros(16, dtype=int)
            for i in range(2):
                for j in range(2):
                    if alpha_R[l][i][j] != 0:
                        v_vec += alpha_R[l][i][j] * CoeffB[k][i][j].flatten()
            V[:, t] = v_vec
            
            # Calculate W column
            # W[idx, t] = Delta[k][br][bc] * gamma[l][lr][lc]
            # where idx corresponds to C[br*2+lr, bc*2+lc]
            # W index convention: W[k*n+i, r] for C[i, k]
            # So for C[r, c], W index is c*4 + r
            
            for br in range(2):
                for bc in range(2):
                    if Delta[k][br][bc] != 0:
                        for lr in range(2):
                            for lc in range(2):
                                if gamma[l][lr][lc] != 0:
                                    global_r = br * 2 + lr
                                    global_c = bc * 2 + lc
                                    w_idx = global_c * 4 + global_r
                                    W[w_idx, t] += Delta[k][br][bc] * gamma[l][lr][lc]
    
    if time.time() > _DEADLINE - _MARGIN:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
