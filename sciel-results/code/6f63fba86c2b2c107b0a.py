import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: i*m+j
    # v_idx: j*p+k
    # w_idx: k*n+i
    # coeff: integer
    def add_term(i, j, k, coeff, col):
        U[i * m + j][col] += coeff
        V[j * p + k][col] += coeff
        W[k * n + i][col] += coeff

    col = 0
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, col); add_term(1, 1, 1, 1, col)
    add_term(0, 0, 1, 1, col); add_term(1, 1, 0, 1, col)
    # Wait, Strassen terms are sums of products.
    # M1 involves A00, A11, B00, B11.
    # But U, V, W are for a SINGLE product term in the decomposition.
    # The decomposition is sum of rank-1 tensors.
    # Strassen's algorithm is a decomposition of the 2x2 matrix multiplication tensor into 7 rank-1 tensors.
    # So I need to add the 7 Strassen terms.
    
    # Let's define the 7 Strassen terms for 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Term 1: (a+d)(e+h) = ae + ah + de + dh
    # This is 4 products. But Strassen says it's 1 "product" in the algorithm, but in tensor decomposition, it's 4 rank-1 tensors?
    # No. The tensor for 2x2 matrix multiplication has rank 7.
    # The 7 terms are:
    # T1: (a+d)(e+h) -> contributes to all 4 outputs.
    # This is NOT a rank-1 tensor. A rank-1 tensor is of the form u v w^T.
    # (a+d)(e+h) = ae + ah + de + dh. This is a sum of 4 rank-1 tensors.
    # So the rank of 2x2 matrix multiplication is NOT 7 if we count these as separate?
    # No, the rank is 7.
    # This means the 7 terms in Strassen's algorithm are the 7 rank-1 tensors?
    # No, Strassen's algorithm computes 7 bilinear forms.
    # Each bilinear form is a product of two linear forms.
    # A product of two linear forms is a rank-1 tensor.
    # Yes.
    # So M1 = (a+d)(e+h) is ONE rank-1 tensor.
    # U row for 'a+d' is [1, 0, 1, 0] (for a,b,c,d).
    # V row for 'e+h' is [1, 0, 0, 1] (for e,f,g,h).
    # W row for M1 contribution:
    # M1 contributes to C11, C12, C21, C22.
    # C11 = M1 + ...
    # So W row has 1s for C11, C12, C21, C22.
    
    # So I need to add 7 terms for the 2x2 block.
    
    # Indices for 2x2 block:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C01=1, C10=1, C11=1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1, C11=-1 (Wait, C11 = M1 - M2 + M3 + M6)
    # C10 = M2 + M4
    # So W: C10=1, C11=-1
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # Term 2: M3 = A00(B01-B11)
    # U: A00=1
    # V: B01=1, B11=-1
    # W: C01=1, C11=1
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # Term 3: M4 = A11(B10-B00)
    # U: A11=1
    # V: B10=1, B00=-1
    # W: C00=1, C10=1
    U[3][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # Term 4: M5 = (A00+A01)B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C00=-1, C01=1
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11=1
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00=1
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Term 7-10: A[0:2, 2] B[2, 0:2]
    # A02 B20 -> C00
    U[6][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    # A02 B21 -> C01
    U[6][col] = 1; V[7][col] = 1; W[1][col] = 1; col += 1
    # A12 B20 -> C10
    U[9][col] = 1; V[6][col] = 1; W[3][col] = 1; col += 1
    # A12 B21 -> C11
    U[9][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # Term 11-14: A[0:2, 0:2] B[0:2, 2]
    # A00 B02 -> C02
    U[0][col] = 1; V[2][col] = 1; W[2][col] = 1; col += 1
    # A01 B12 -> C02
    U[1][col] = 1; V[5][col] = 1; W[2][col] = 1; col += 1
    # A10 B02 -> C12
    U[2][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    # A11 B12 -> C12
    U[3][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # Term 15-16: A[0:2, 2] B[2, 2]
    # A02 B22 -> C02
    U[6][col] = 1; V[8][col] = 1; W[2][col] = 1; col += 1
    # A12 B22 -> C12
    U[9][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    
    # Term 17-20: A[2, 0:2] B[0:2, 0:2]
    # A20 B00 -> C20
    U[3][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1
    # A21 B10 -> C20
    U[4][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    # A20 B01 -> C21
    U[3][col] = 1; V[1][col] = 1; W[7][col] = 1; col += 1
    # A21 B11 -> C21
    U[4][col] = 1; V[4][col] = 1; W[7][col] = 1; col += 1
    
    # Term 21-22: A[2, 0:2] B[0:2, 2]
    # A20 B02 -> C22
    U[3][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    # A21 B12 -> C22
    U[4][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    
    # Term 23-24: A[2, 2] B[2, 0:2]
    # A22 B20 -> C20
    U[8][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    # A22 B21 -> C21
    U[8][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1
    
    # Term 25: A[2, 2] B[2, 2]
    # A22 B22 -> C22
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
