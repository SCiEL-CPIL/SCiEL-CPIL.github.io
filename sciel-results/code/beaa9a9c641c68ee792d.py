"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    # Strassen-based block decomposition
    # Split: i = 2*I + i0, j = 2*J + j0, k = 2*K + k0
    # where I,J in {0,1}, K in {0,1,2}, i0,j0,k0 in {0,1}
    # Each sub-problem is 2x2 * 2x2, solved with Strassen rank-7
    # Total rank = 12 * 7 = 84
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e + b*g, a*f + b*h],
    #      [c*e + d*g, c*f + d*h]]
    # 
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For tensor decomposition, each "multiplication" M_r corresponds to one rank-1 term.
    # For M1 = (a+d)(e+h): 
    #   A part: a has coeff 1, d has coeff 1 -> U contributions
    #   B part: e has coeff 1, h has coeff 1 -> V contributions
    #   C part: C11 gets +1, C22 gets +1 -> W contributions
    #
    # Let's build the full decomposition.
    
    # Index mapping:
    # A[i,j] where i = 2*I + i0, j = 2*J + j0
    # B[j,k] where j = 2*J + j0, k = 2*K + k0
    # C[i,k] where i = 2*I + i0, k = 2*K + k0
    #
    # U index: i*m + j = (2*I+i0)*4 + (2*J+j0) = 8*I + 4*i0 + 4*J + j0
    # V index: j*p + k = (2*J+j0)*6 + (2*K+k0) = 12*J + 6*j0 + 6*K + k0
    # W index: k*n + i = (2*K+k0)*4 + (2*I+i0) = 8*K + 4*k0 + 4*I + i0
    
    rank = 12 * 7  # 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # Strassen coefficients for each of the 7 multiplications:
    # For M_r = (linear combo of A entries) * (linear combo of B entries)
    # contributes to C entries with certain coefficients
    #
    # A entries indexed by (i0, j0) in {0,1}x{0,1}
    # B entries indexed by (j0, k0) in {0,1}x{0,1}
    # C entries indexed by (i0, k0) in {0,1}x{0,1}
    #
    # Strassen multiplications:
    # M1: (A00 + A11) * (B00 + B11) -> C00 += 1, C11 += 1
    # M2: (A10 + A11) * B00 -> C10 += 1
    # M3: A00 * (B01 - B11) -> C01 += 1
    # M4: A11 * (B10 - B00) -> C00 += 1, C10 += 1
    # M5: (A00 + A01) * B11 -> C00 -= 1, C01 += 1
    # M6: (A10 - A00) * (B00 + B01) -> C11 += 1
    # M7: (A01 - A11) * (B10 + B11) -> C00 += 1, C11 -= 1
    
    strassen = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict from (i0,j0) to coeff
        # B_coeffs: dict from (j0,k0) to coeff
        # C_coeffs: dict from (i0,k0) to coeff
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    col = 0
    for I in range(2):  # row block of A
        for J in range(2):  # col block of A / row block of B
            for K in range(3):  # col block of B
                for r in range(7):  # Strassen multiplication index
                    A_coeffs, B_coeffs, C_coeffs = strassen[r]
                    
                    # Fill U: for each A entry (i0, j0) with nonzero coeff
                    for (i0, j0), coeff in A_coeffs.items():
                        u_idx = (2*I + i0) * m + (2*J + j0)
                        U[u_idx][col] += coeff
                    
                    # Fill V: for each B entry (j0, k0) with nonzero coeff
                    for (j0, k0), coeff in B_coeffs.items():
                        v_idx = (2*J + j0) * p + (2*K + k0)
                        V[v_idx][col] += coeff
                    
                    # Fill W: for each C entry (i0, k0) with nonzero coeff
                    for (i0, k0), coeff in C_coeffs.items():
                        w_idx = (2*K + k0) * n + (2*I + i0)
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
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
