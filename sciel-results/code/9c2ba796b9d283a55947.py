"""Strassen-based rank-189 decomposition for 6x6x6 matrix multiplication."""
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition for 6x6 matrix multiplication using:
    - 2x2 block partition into 3x3 blocks
    - Strassen's 7 products at the block level
    - Schoolbook (rank 27) at the 3x3 element level
    
    The tensor decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where n=m=p=6.
    """
    n, m, p = DIMENSIONS
    rank = 189  # 7 Strassen products * 27 schoolbook terms per 3x3 multiplication
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # shape (36, 189)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (36, 189)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (36, 189)
    
    # Strassen's 7 block products:
    # A partitioned as: A = [[A00, A01], [A10, A11]]
    # B partitioned as: B = [[B00, B01], [B10, B11]]
    # C = A*B partitioned as: C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product t (0..6):
    #   X_t = linear combination of A blocks
    #   Y_t = linear combination of B blocks
    #   Z_t = X_t * Y_t (3x3 matrix product, rank 27)
    #   C receives contributions from Z_t with certain coefficients
    
    # Define the 7 Strassen terms:
    # Each term: (list of (block_row, block_col, coeff) for A,
    #             list of (block_row, block_col, coeff) for B,
    #             list of (block_row, block_col, coeff) for C contributions)
    
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'A': [(0, 0, 1.0), (1, 1, 1.0)],
            'B': [(0, 0, 1.0), (1, 1, 1.0)],
            'C': [(0, 0, 1.0), (1, 1, 1.0)]
        },
        # M2 = (A10 + A11) * B00
        {
            'A': [(1, 0, 1.0), (1, 1, 1.0)],
            'B': [(0, 0, 1.0)],
            'C': [(1, 0, 1.0), (1, 1, -1.0)]
        },
        # M3 = A00 * (B01 - B11)
        {
            'A': [(0, 0, 1.0)],
            'B': [(0, 1, 1.0), (1, 1, -1.0)],
            'C': [(0, 1, 1.0), (1, 1, 1.0)]
        },
        # M4 = A11 * (B10 - B00)
        {
            'A': [(1, 1, 1.0)],
            'B': [(1, 0, 1.0), (0, 0, -1.0)],
            'C': [(1, 0, 1.0), (0, 0, 1.0)]
        },
        # M5 = (A00 + A01) * B11
        {
            'A': [(0, 0, 1.0), (0, 1, 1.0)],
            'B': [(1, 1, 1.0)],
            'C': [(0, 0, -1.0), (0, 1, 1.0)]
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'A': [(1, 0, 1.0), (0, 0, -1.0)],
            'B': [(0, 0, 1.0), (0, 1, 1.0)],
            'C': [(1, 1, 1.0)]
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'A': [(0, 1, 1.0), (1, 1, -1.0)],
            'B': [(1, 0, 1.0), (1, 1, 1.0)],
            'C': [(0, 0, 1.0)]
        },
    ]
    
    col = 0
    for term in strassen_terms:
        A_blocks = term['A']
        B_blocks = term['B']
        C_contrib = term['C']
        
        # For each element in the 3x3 product X * Y:
        # X[i,j] * Y[j,k] contributes to Z[i,k]
        # where X[i,j] = sum over (abr, abc, ca) of ca * A[abr*3+i, abc*3+j]
        # and Y[j,k] = sum over (bbr, bbc, cb) of cb * B[bbr*3+j, bbc*3+k]
        #
        # The contribution to C is:
        # sum_{abr,abc,bbr,bbc} ca*cb * A[abr*3+i, abc*3+j] * B[bbr*3+j, bbc*3+k]
        # This is a rank-1 term in the tensor decomposition for each (abr,abc,bbr,bbc) pair.
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # For each combination of A and B block contributions
                    for (abr, abc, ca) in A_blocks:
                        for (bbr, bbc, cb) in B_blocks:
                            coeff = ca * cb
                            if coeff == 0:
                                continue
                            
                            # A element index: row = abr*3 + i, col = abc*3 + j
                            a_row = abr * 3 + i
                            a_col = abc * 3 + j
                            U_idx = a_row * 6 + a_col  # U[i*m + j]
                            
                            # B element index: row = bbr*3 + j, col = bbc*3 + k
                            b_row = bbr * 3 + j
                            b_col = bbc * 3 + k
                            V_idx = b_row * 6 + b_col  # V[j*p + k]
                            
                            # Set U and V for this rank-1 term
                            U[U_idx][col] += coeff
                            V[V_idx][col] += coeff
                            
                            # C receives contributions from this product
                            # For each C block contribution:
                            for (cbr, cbc, cc) in C_contrib:
                                # C element: row = cbr*3 + i, col = cbc*3 + k
                                c_row = cbr * 3 + i
                                c_col = cbc * 3 + k
                                # W index: W[k*n + i] where k = c_col, i = c_row
                                W_idx = c_col * 6 + c_row
                                W[W_idx][col] += coeff * cc
                    
                    col += 1
    
    # Convert to integers where possible for cleaner output
    # Check if all values are integers (within floating point tolerance)
    def to_int_if_close(val):
        rounded = round(val)
        if abs(val - rounded) < 1e-10:
            return int(rounded)
        return val
    
    U = [[to_int_if_close(v) for v in row] for row in U]
    V = [[to_int_if_close(v) for v in row] for row in V]
    W = [[to_int_if_close(v) for v in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
