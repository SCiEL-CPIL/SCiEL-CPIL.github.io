import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define Strassen coefficients for 2x2 matrices
    # alpha[s][u][v] is coefficient of A[u][v] in step s
    # beta[s][u][v] is coefficient of B[u][v] in step s
    # zeta[s][u][v] is coefficient of M_s in C[u][v]
    
    alpha = np.zeros((7, 2, 2))
    beta = np.zeros((7, 2, 2))
    zeta = np.zeros((7, 2, 2))
    
    # Step 1: (A00+A11)(B00+B11)
    alpha[0, 0, 0] = 1; alpha[0, 1, 1] = 1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    
    # Step 2: (A10+A11)B00
    alpha[1, 1, 0] = 1; alpha[1, 1, 1] = 1
    beta[1, 0, 0] = 1
    
    # Step 3: A00(B01-B11)
    alpha[2, 0, 0] = 1
    beta[2, 0, 1] = 1; beta[2, 1, 1] = -1
    
    # Step 4: A11(B10-B00)
    alpha[3, 1, 1] = 1
    beta[3, 1, 0] = 1; beta[3, 0, 0] = -1
    
    # Step 5: (A00+A01)B11
    alpha[4, 0, 0] = 1; alpha[4, 0, 1] = 1
    beta[4, 1, 1] = 1
    
    # Step 6: (A10-A00)(B00+B01)
    alpha[5, 1, 0] = 1; alpha[5, 0, 0] = -1
    beta[5, 0, 0] = 1; beta[5, 0, 1] = 1
    
    # Step 7: (A01-A11)(B10+B11)
    alpha[6, 0, 1] = 1; alpha[6, 1, 1] = -1
    beta[6, 1, 0] = 1; beta[6, 1, 1] = 1
    
    # Reconstruction for C
    # C00 = M1 + M4 - M5 + M7
    zeta[0, 0, 0] = 1
    zeta[3, 0, 0] = 1
    zeta[4, 0, 0] = -1
    zeta[6, 0, 0] = 1
    
    # C01 = M3 + M5
    zeta[2, 0, 1] = 1
    zeta[4, 0, 1] = 1
    
    # C10 = M2 + M4
    zeta[1, 1, 0] = 1
    zeta[3, 1, 0] = 1
    
    # C11 = M1 - M2 + M3 + M6
    zeta[0, 1, 1] = 1
    zeta[1, 1, 1] = -1
    zeta[2, 1, 1] = 1
    zeta[5, 1, 1] = 1
    
    # Construct U, V, W
    for s1 in range(7):
        for s2 in range(7):
            r = s1 * 7 + s2
            
            # U: A indices
            # i = 2*u + x, j = 2*v + y
            # U[i, j, r] = alpha[s1][u][v] * alpha[s2][x][y]
            for u in range(2):
                for v in range(2):
                    if alpha[s1, u, v] != 0:
                        for x in range(2):
                            for y in range(2):
                                if alpha[s2, x, y] != 0:
                                    i = 2 * u + x
                                    j = 2 * v + y
                                    U[i * m + j, r] = alpha[s1, u, v] * alpha[s2, x, y]
            
            # V: B indices
            # j = 2*u + x, k = 2*v + y
            # V[j, k, r] = beta[s1][u][v] * beta[s2][x][y]
            for u in range(2):
                for v in range(2):
                    if beta[s1, u, v] != 0:
                        for x in range(2):
                            for y in range(2):
                                if beta[s2, x, y] != 0:
                                    j = 2 * u + x
                                    k = 2 * v + y
                                    V[j * p + k, r] = beta[s1, u, v] * beta[s2, x, y]
            
            # W: C indices
            # i = 2*u + p, k = 2*v + q
            # W[4*k + i, r] = zeta[s1][u][v] * zeta[s2][p][q]
            # Note: zeta[s][u][v] is coeff of M_s in C[u][v]
            # Here s1 is outer step, s2 is inner step.
            # Outer coeff: zeta[s1][u][v]
            # Inner coeff: zeta[s2][p][q]
            for u in range(2):
                for v in range(2):
                    if zeta[s1, u, v] != 0:
                        for p in range(2):
                            for q in range(2):
                                if zeta[s2, p, q] != 0:
                                    i = 2 * u + p
                                    k = 2 * v + q
                                    w_idx = k * n + i
                                    W[w_idx, r] = zeta[s1, u, v] * zeta[s2, p, q]
                                    
    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

