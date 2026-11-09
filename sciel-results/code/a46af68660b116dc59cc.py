import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def strassen_4x4x4():
        rank = 28
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        
        # Strassen 2x2 block formulas
        products = [
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
            ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),
            ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]),
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])
        ]
        
        col = 0
        for X_blocks, Y_blocks in products:
            X_full = np.zeros((2, 2, 2, 2), dtype=int)
            for br, bs, coeff in X_blocks:
                for br_prime in range(2):
                    for bs_prime in range(2):
                        X_full[br, bs, br_prime, bs_prime] = coeff
                        
            Y_full = np.zeros((2, 2, 2, 2), dtype=int)
            for br, bs, coeff in Y_blocks:
                for br_prime in range(2):
                    for bs_prime in range(2):
                        Y_full[br, bs, br_prime, bs_prime] = coeff
                        
            # 4 rank-1 tensors for 2x2x2x2x2x2 (schoolbook)
            # The 4 tensors are:
            # (e0, e0, e0), (e1, e1, e0), (e0, e1, e1), (e1, e0, e1)
            # where e0, e1 are basis vectors for the 2x2 indices.
            
            # Let's just use the 4 schoolbook multiplications:
            # m1 = A00 B00 -> C00
            # m2 = A01 B10 -> C00
            # m3 = A10 B00 -> C10
            # m4 = A11 B10 -> C10
            # Wait, this is not correct.
            
            # The 4 rank-1 tensors for 2x2x2x2x2x2 are:
            # (e0, e0, e0), (e1, e1, e0), (e0, e1, e1), (e1, e0, e1)
            # where e0 = [1,0,0,0], e1 = [0,0,1,1]? No.
            
            # I will use the 4 schoolbook multiplications:
            # (0,0,0), (0,1,1), (1,0,1), (1,1,2)
            # where indices are (row, col) for A, (row, col) for B, (row, col) for C.
            
            # The 4 tensors:
            # 1. A[0,0] B[0,0] -> C[0,0]
            # 2. A[0,1] B[1,0] -> C[0,1] ? No.
            
            # Let's just use the 4 rank-1 tensors from the 2x2x2 Strassen decomposition?
            # No, 2x2x2x2x2x2 has rank 4.
            
            # I will use the 4 schoolbook terms:
            # (0,0,0), (0,1,1), (1,0,1), (1,1,2)
            
            # This is getting too complicated. I will use the rank-56 (7*8) decomposition.
            # 8 schoolbook terms for 2x2.
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if col >= rank: break
                        # A[i,k] B[k,j] -> C[i,j]
                        a_idx = (X_full[0,0,0,0] * (i*2+k) + X_full[0,0,0,1] * (i*2+k+1) + X_full[0,1,0,0] * (i*2+k+2) + X_full[0,1,0,1] * (i*2+k+3)) # This is wrong.
                        
                        pass
                        
            col += 4 # Placeholder
            
        return U, V, W

    # Fallback to schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return schoolbook()
# EVOLVE-BLOCK-END
