import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 on 3x3 blocks
    # Block size 3, 2x2 grid of blocks
    bs = 3
    nb = 2
    
    # Strassen coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # X_s (A-side) coefficients for blocks (a, b)
    # s, a, b -> coeff
    x_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): 1}, # M2
        {(0,0): 1},           # M3
        {(1,1): 1},           # M4
        {(0,0): 1, (0,1): 1}, # M5
        {(1,0): 1, (0,0): -1},# M6
        {(0,1): 1, (1,1): -1} # M7
    ]
    
    # Y_s (B-side) coefficients for blocks (c, d)
    # s, c, d -> coeff
    y_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(0,0): 1},           # M2
        {(0,1): 1, (1,1): -1},# M3
        {(1,0): 1, (0,0): -1},# M4
        {(1,1): 1},           # M5
        {(0,0): 1, (0,1): 1}, # M6
        {(1,0): 1, (1,1): 1}  # M7
    ]
    
    # C-side coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # s, e, f -> coeff where e is block row of C, f is block col of C
    w_coeffs = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, # M1
        {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1},# M2
        {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}, # M3
        {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}, # M4
        {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0},# M5
        {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}, # M6
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}  # M7
    ]
    
    rank = 7 * 27
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    r_idx = 0
    for s in range(7):
        # Iterate over local indices for the 3x3 schoolbook multiplication
        for il in range(bs):
            for kl in range(bs):
                for jl in range(bs):
                    # U part: A[i, k]
                    # U[i*m+k, r] = coeff if i%3==il, k%3==kl, and block matches
                    for a in range(nb):
                        for b in range(nb):
                            coeff = x_coeffs[s].get((a,b), 0)
                            if coeff != 0:
                                i = a * bs + il
                                k = b * bs + kl
                                U[i*m + k, r_idx] = coeff
                    
                    # V part: B[k, j]
                    # V[k*p+j, r] = coeff if k%3==kl, j%3==jl, and block matches
                    for c in range(nb):
                        for d in range(nb):
                            coeff = y_coeffs[s].get((c,d), 0)
                            if coeff != 0:
                                k = c * bs + kl
                                j = d * bs + jl
                                V[k*p + j, r_idx] = coeff
                    
                    # W part: C[i, j]
                    # W[i*n+j, r] = coeff if i%3==il, j%3==jl, and block matches
                    for e in range(nb):
                        for f in range(nb):
                            coeff = w_coeffs[s].get((e,f), 0)
                            if coeff != 0:
                                i = e * bs + il
                                j = f * bs + jl
                                W[i*n + j, r_idx] = coeff
                                
                    r_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
