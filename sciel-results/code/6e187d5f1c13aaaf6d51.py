import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0
    
    # Build the 63-rank decomposition: 3x3 schoolbook blocks of 2x2 Strassen
    # Total rank = 9 * 7 = 63
    def build_kronecker_strassen_6x6():
        rank = 63
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        idx = 0
        # Loop over 3x3 block indices
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # Loop over 7 Strassen terms
                    for s in range(7):
                        # Define local 2x2 indices within the blocks
                        # A block (i, k) is at rows 2i, 2i+1 and cols 2k, 2k+1
                        # B block (k, j) is at rows 2k, 2k+1 and cols 2j, 2j+1
                        # C block (j, i) is at rows 2j, 2j+1 and cols 2i, 2i+1
                        
                        a00 = (2*i) * 6 + 2*k
                        a01 = (2*i) * 6 + 2*k + 1
                        a10 = (2*i+1) * 6 + 2*k
                        a11 = (2*i+1) * 6 + 2*k + 1
                        
                        b00 = (2*k) * 6 + 2*j
                        b01 = (2*k) * 6 + 2*j + 1
                        b10 = (2*k+1) * 6 + 2*j
                        b11 = (2*k+1) * 6 + 2*j + 1
                        
                        c00 = (2*j) * 6 + 2*i
                        c01 = (2*j+1) * 6 + 2*i
                        c10 = (2*j) * 6 + (2*i+1)
                        c11 = (2*j+1) * 6 + (2*i+1)
                        
                        if s == 0:
                            U[a00, idx] = 1; U[a11, idx] = 1
                            V[b00, idx] = 1; V[b11, idx] = 1
                            W[c00, idx] = 1; W[c11, idx] = 1
                        elif s == 1:
                            U[a10, idx] = 1; U[a11, idx] = 1
                            V[b00, idx] = 1
                            W[c10, idx] = 1; W[c11, idx] = -1
                        elif s == 2:
                            U[a00, idx] = 1
                            V[b01, idx] = 1; V[b11, idx] = -1
                            W[c01, idx] = 1; W[c11, idx] = 1
                        elif s == 3:
                            U[a11, idx] = 1
                            V[b10, idx] = 1; V[b00, idx] = -1
                            W[c00, idx] = 1; W[c10, idx] = 1
                        elif s == 4:
                            U[a00, idx] = 1; U[a01, idx] = 1
                            V[b11, idx] = 1
                            W[c00, idx] = -1; W[c01, idx] = 1
                        elif s == 5:
                            U[a10, idx] = 1; U[a00, idx] = -1
                            V[b00, idx] = 1; V[b01, idx] = 1
                            W[c11, idx] = 1
                        elif s == 6:
                            U[a01, idx] = 1; U[a11, idx] = -1
                            V[b10, idx] = 1; V[b11, idx] = 1
                            W[c00, idx] = 1
                        idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return build_kronecker_strassen_6x6()
# EVOLVE-BLOCK-END
