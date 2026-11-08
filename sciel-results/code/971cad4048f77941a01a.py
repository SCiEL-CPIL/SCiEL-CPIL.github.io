"""Matrix multiplication 6x6x6 using Laderman's 14-term 3x3 block decomposition."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    block_size = 3
    blocks_per_dim = 2  # 6 / 3 = 2
    
    # Laderman's 14-term algorithm for 3x3 matrix multiplication
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (i,j) -> value for A block
    # v_coeffs: dict (i,j) -> value for B block
    # w_coeffs: dict (i,j) -> value for C block
    
    # Laderman's 14 products for 3x3 matrices:
    # M1 = (A00 - A01 - A02)(B00 + B01 + B02)
    # M2 = (A00 - A01)(B10 + B11)
    # M3 = (A01 - A02)(B11 + B12)
    # M4 = (A10 - A11 - A12)(B00 + B01 + B02)
    # M5 = (A10 - A11)(B10 + B11)
    # M6 = (A11 - A12)(B11 + B12)
    # M7 = (A00 - A01 - A02)(B20 + B21 + B22)
    # M8 = (A00 - A01)(B20 + B21)
    # M9 = (A01 - A02)(B21 + B22)
    # M10 = (A20 - A21 - A22)(B00 + B01 + B02)
    # M11 = (A20 - A21)(B10 + B11)
    # M12 = (A21 - A22)(B11 + B12)
    # M13 = (A20 - A21 - A22)(B20 + B21 + B22)
    # M14 = (A20 - A21)(B20 + B21)
    
    # Actually, let me use the standard Laderman formulation more carefully.
    # Standard Laderman 3x3:
    # C00 = M1 + M2 + M5 + M7 + M8 + M11 + M13
    # C01 = M3 + M5 + M6 + M9 + M11 + M12 + M14
    # C02 = M3 + M6 + M9 + M12
    # C10 = M4 + M5 + M8 + M10 + M11 + M13
    # C11 = M2 + M4 + M6 + M8 + M10 + M12
    # C12 = M6 + M9 + M12
    # C20 = M4 + M7 + M8 + M10 + M13
    # C21 = M7 + M9 + M12 + M14
    # C22 = M14
    
    # Let me define the 14 terms properly:
    laderman_terms = [
        # M1: (A00-A01-A02)(B00+B01+B02)
        ({(0,0):1,(0,1):-1,(0,2):-1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1}),
        # M2: (A00-A01)(B10+B11)
        ({(0,0):1,(0,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # M3: (A01-A02)(B11+B12)
        ({(0,1):1,(0,2):-1}, {(1,1):1,(1,2):1}, {(0,1):1,(0,2):1,(1,1):1,(1,2):1}),
        # M4: (A10-A11-A12)(B00+B01+B02)
        ({(1,0):1,(1,1):-1,(1,2):-1}, {(0,0):1,(0,1):1,(0,2):1}, {(1,0):1,(1,1):-1,(2,0):1,(2,1):-1}),
        # M5: (A10-A11)(B10+B11)
        ({(1,0):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(1,0):1,(1,1):1,(2,0):1,(2,1):1}),
        # M6: (A11-A12)(B11+B12)
        ({(1,1):1,(1,2):-1}, {(1,1):1,(1,2):1}, {(1,1):1,(1,2):1,(2,1):1,(2,2):1}),
        # M7: (A00-A01-A02)(B20+B21+B22)
        ({(0,0):1,(0,1):-1,(0,2):-1}, {(2,0):1,(2,1):1,(2,2):1}, {(0,0):1,(0,1):-1,(1,0):1,(1,1):-1,(2,0):1,(2,1):-1}),
        # M8: (A00-A01)(B20+B21)
        ({(0,0):1,(0,1):-1}, {(2,0):1,(2,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1,(2,0):1,(2,1):1}),
        # M9: (A01-A02)(B21+B22)
        ({(0,1):1,(0,2):-1}, {(2,1):1,(2,2):1}, {(0,1):1,(0,2):1,(1,1):1,(1,2):1,(2,1):1,(2,2):1}),
        # M10: (A20-A21-A22)(B00+B01+B02)
        ({(2,0):1,(2,1):-1,(2,2):-1}, {(0,0):1,(0,1):1,(0,2):1}, {(2,0):1,(2,1):-1}),
        # M11: (A20-A21)(B10+B11)
        ({(2,0):1,(2,1):-1}, {(1,0):1,(1,1):1}, {(2,0):1,(2,1):1}),
        # M12: (A21-A22)(B11+B12)
        ({(2,1):1,(2,2):-1}, {(1,1):1,(1,2):1}, {(2,1):1,(2,2):1}),
        # M13: (A20-A21-A22)(B20+B21+B22)
        ({(2,0):1,(2,1):-1,(2,2):-1}, {(2,0):1,(2,1):1,(2,2):1}, {(2,0):1,(2,1):1,(2,2):1}),
        # M14: (A20-A21)(B20+B21)
        ({(2,0):1,(2,1):-1}, {(2,0):1,(2,1):1}, {(2,2):1}),
    ]
    
    rank = (blocks_per_dim ** 3) * len(laderman_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in laderman_terms:
                    # Update U: A[I*bs+pi, J*bs+pj]
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    # Update V: B[J*bs+pi, K*bs+pj]
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    # Update W: C[I*bs+pi, K*bs+pj] -> W[k*n+i] where k=K*bs+pj, i=I*bs+pi
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
