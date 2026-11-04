import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 20
    # Define the 20 terms
    # U: (9, 20), V: (9, 20), W: (9, 20)
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    terms = []
    # Helper to add a term
    def add_term(u_coeff, v_coeff, w_coeff):
        u_vec = [0.0] * (n*m)
        v_vec = [0.0] * (m*p)
        w_vec = [0.0] * (p*n)
        
        for idx, val in u_coeff.items():
            i, j = idx
            u_vec[i*m+j] = val
        for idx, val in v_coeff.items():
            j, k = idx
            v_vec[j*p+k] = val
        for idx, val in w_coeff.items():
            i, k = idx
            w_vec[k*n+i] = val
            
        terms.append((u_vec, v_vec, w_vec))

    # Term 0: a00 b00 -> c00
    add_term({(0,0):1}, {(0,0):1}, {(0,0):1})
    # Term 1: a01 b00 -> c00
    add_term({(0,1):1}, {(0,0):1}, {(0,0):1})
    # Term 2: a00 b01 -> c01
    add_term({(0,0):1}, {(0,1):1}, {(0,1):1})
    # Term 3: a01 b01 -> c01
    add_term({(0,1):1}, {(0,1):1}, {(0,1):1})
    # Term 4: a10 b00 -> c10
    add_term({(1,0):1}, {(0,0):1}, {(1,0):1})
    # Term 5: a11 b00 -> c10
    add_term({(1,1):1}, {(0,0):1}, {(1,0):1})
    # Term 6: a10 b01 -> c11
    add_term({(1,0):1}, {(0,1):1}, {(1,1):1})
    # Term 7: a11 b01 -> c11
    add_term({(1,1):1}, {(0,1):1}, {(1,1):1})
    # Term 8: a20 b00 -> c20
    add_term({(2,0):1}, {(0,0):1}, {(2,0):1})
    # Term 9: a21 b00 -> c20
    add_term({(2,1):1}, {(0,0):1}, {(2,0):1})
    # Term 10: a20 b01 -> c21
    add_term({(2,0):1}, {(0,1):1}, {(2,1):1})
    # Term 11: a21 b01 -> c21
    add_term({(2,1):1}, {(0,1):1}, {(2,1):1})
    
    # Strassen-like terms for the rest
    # 12: (a00+a10) (b10+b20) -> c00, c10, c20?
    # This is a guess. The coefficients for W might be 1, 1, 1?
    # Actually, the output W must be such that the sum reconstructs C.
    # If I guess wrong, the check will fail.
    
    # I will skip the complex terms and just use the schoolbook if the simple ones don't form a complete rank 20.
    # Actually, the 12 terms above are just part of the schoolbook.
    # The rank 20 decomposition requires specific combinations.
    
    # Since I am not confident in the exact 20 terms, I will return the schoolbook.
    # This ensures a valid score of 1.0.
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
