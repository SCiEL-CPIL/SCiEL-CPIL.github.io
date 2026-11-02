import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # term: (U_indices, V_indices, W_indices)
    # U_indices: list of (i, j, coeff)
    # V_indices: list of (j, k, coeff)
    # W_indices: list of (k, i, coeff) -> stored as W[k*n+i]
    
    terms = []
    
    # 1. A1 B1 -> C11 (top-left 1x1 block)
    # A1 = A[0,0], B1 = B[0,0], C11 = C[0,0]
    terms.append([
        [(0, 0, 1)],
        [(0, 0, 1)],
        [(0, 0, 1)] # W[0*3+0] = W[0]
    ])
    
    # 2. a d^T -> C_top_right? No, a is row 0, cols 1,2. d is col 0, rows 1,2.
    # a = [A01, A02], d^T = [B10, B20]^T? No.
    # B = [[B1, c], [d^T, B2]].
    # B1 is 0,0. c is 0,1 and 0,2. d^T is 1,0 and 2,0. B2 is 1,1..2,2.
    # a d^T: A01*B10, A01*B20, A02*B10, A02*B20.
    # These contribute to C01, C02?
    # C = [[A1B1 + ad^T, A1c + aB2], [b^TB1 + A2d^T, b^Tc + A2B2]]
    # Top-left is 1x1. Top-right is 1x2. Bottom-left is 2x1. Bottom-right is 2x2.
    # ad^T is 1x2. Elements: (A01*B10, A01*B20) and (A02*B10, A02*B20)?
    # No, a is 1x2, d^T is 2x1. Product is 1x1?
    # Wait. A = [[A1, a], [b^T, A2]]. A1 is 1x1. a is 1x2. b is 2x1. A2 is 2x2.
    # B = [[B1, c], [d^T, B2]]. B1 is 1x1. c is 1x2. d is 2x1. B2 is 2x2.
    # C11 = A1B1 + a d^T. a is 1x2, d^T is 2x1. Product is 1x1.
    # a = [A01, A02]. d^T = [B10, B20]^T.
    # a d^T = A01*B10 + A02*B20.
    # This is a sum of 2 products. So 2 terms.
    terms.append([
        [(0, 1, 1)],
        [(1, 0, 1)],
        [(0, 0, 1)]
    ])
    terms.append([
        [(0, 2, 1)],
        [(2, 0, 1)],
        [(0, 0, 1)]
    ])
    
    # 3. A1 c -> C_top_right (1x2)
    # A1 = A00. c = [B01, B02].
    # C01 += A00*B01. C02 += A00*B02.
    terms.append([
        [(0, 0, 1)],
        [(0, 1, 1)],
        [(1, 0, 1)] # C01 -> k=1, i=0 -> W[1*3+0]=W[3]
    ])
    terms.append([
        [(0, 0, 1)],
        [(0, 2, 1)],
        [(2, 0, 1)] # C02 -> k=2, i=0 -> W[2*3+0]=W[6]
    ])
    
    # 4. a B2 -> C_top_right (1x2)
    # a = [A01, A02]. B2 = [[B11, B12], [B21, B22]].
    # C01 += A01*B11 + A02*B21.
    # C02 += A01*B12 + A02*B22.
    # 4 products.
    terms.append([
        [(0, 1, 1)],
        [(1, 1, 1)],
        [(1, 0, 1)]
    ])
    terms.append([
        [(0, 2, 1)],
        [(2, 1, 1)],
        [(1, 0, 1)]
    ])
    terms.append([
        [(0, 1, 1)],
        [(1, 2, 1)],
        [(2, 0, 1)]
    ])
    terms.append([
        [(0, 2, 1)],
        [(2, 2, 1)],
        [(2, 0, 1)]
    ])
    
    # 5. b^T B1 -> C_bottom_left (2x1)
    # b^T = [A10, A20]. B1 = B00.
    # C10 += A10*B00. C20 += A20*B00.
    terms.append([
        [(1, 0, 1)],
        [(0, 0, 1)],
        [(0, 1, 1)] # C10 -> k=0, i=1 -> W[0*3+1]=W[1]
    ])
    terms.append([
        [(2, 0, 1)],
        [(0, 0, 1)],
        [(0, 2, 1)] # C20 -> k=0, i=2 -> W[0*3+2]=W[2]
    ])
    
    # 6. A2 d^T -> C_bottom_left (2x1)
    # A2 = [[A11, A12], [A21, A22]]. d^T = [B10, B20]^T.
    # C10 += A11*B10 + A12*B20.
    # C20 += A21*B10 + A22*B20.
    terms.append([
        [(1, 1, 1)],
        [(1, 0, 1)],
        [(0, 1, 1)]
    ])
    terms.append([
        [(1, 2, 1)],
        [(2, 0, 1)],
        [(0, 1, 1)]
    ])
    terms.append([
        [(2, 1, 1)],
        [(1, 0, 1)],
        [(0, 2, 1)]
    ])
    terms.append([
        [(2, 2, 1)],
        [(2, 0, 1)],
        [(0, 2, 1)]
    ])
    
    # 7. b^T c -> C_bottom_right? No, b^T c is 1x1?
    # b^T is 1x2, c is 1x2. Dot product?
    # C = [[...], [...], [...], [...]].
    # Bottom-right is 2x2.
    # The term b^T c is part of C11? No.
    # C = A B.
    # C_{2 \times 2} (bottom right) = b^T c + A2 B2.
    # b^T c is a scalar? No, b^T is 1x2, c is 1x2.
    # Wait, dimensions:
    # A: 3x3. Partition 1+2.
    # A1: 1x1. a: 1x2. b: 2x1. A2: 2x2.
    # B: 3x3. Partition 1+2.
    # B1: 1x1. c: 1x2. d: 2x1. B2: 2x2.
    # C11 (1x1) = A1B1 + a d^T.
    # C12 (1x2) = A1 c + a B2.
    # C21 (2x1) = b^T B1 + A2 d^T.
    # C22 (2x2) = b^T c + A2 B2.
    # b^T c: (1x2) * (1x2)? No. b^T is 1x2. c is 1x2.
    # Matrix multiplication requires inner dimensions to match.
    # b is 2x1. b^T is 1x2.
    # c is 1x2.
    # b^T c is (1x2) * (1x2) -> Invalid.
    # Ah, c is the top-right block of B.
    # B = [[B1, c], [d^T, B2]].
    # B1 is 1x1. c is 1x2. d^T is 2x1. B2 is 2x2.
    # b is bottom-left of A. 2x1. b^T is 1x2.
    # C22 = b^T c + A2 B2.
    # b^T c: (1x2) * (1x2) is not defined.
    # It should be b^T (1x2) times c (1x2)? No.
    # The block multiplication is:
    # C22 = (row 2 of A) * (col 2 of B).
    # Row 2 of A is [b^T, A2]. (1x2, 1x2) -> 1x4? No.
    # A is 3x3.
    # Rows 1,2 (0-indexed 1,2) are the bottom part.
    # Cols 1,2 are the right part.
    # C22 is the 2x2 block at bottom right.
    # C22 = A[1:3, 0:1] * B[0:1, 1:3] + A[1:3, 1:3] * B[1:3, 1:3].
    # A[1:3, 0:1] is 2x1 (b). B[0:1, 1:3] is 1x2 (c).
    # b * c is (2x1) * (1x2) = 2x2.
    # So it's b c, not b^T c.
    # b is 2x1. c is 1x2.
    # b c has 4 elements: b1*c1, b1*c2, b2*c1, b2*c2.
    # 4 products.
    terms.append([
        [(1, 0, 1)],
        [(0, 1, 1)],
        [(1, 1, 1)] # C11 of the 2x2 block -> global C11? No.
        # C22 block indices: rows 1,2; cols 1,2.
        # Element (1,1) of block is global (1,1).
        # k=1, i=1 -> W[1*3+1]=W[4]
    ])
    terms.append([
        [(1, 0, 1)],
        [(0, 2, 1)],
        [(2, 1, 1)] # global C12 -> k=2, i=1 -> W[2*3+1]=W[7]
    ])
    terms.append([
        [(2, 0, 1)],
        [(0, 1, 1)],
        [(1, 2, 1)] # global C21 -> k=1, i=2 -> W[1*3+2]=W[5]
    ])
    terms.append([
        [(2, 0, 1)],
        [(0, 2, 1)],
        [(2, 2, 1)] # global C22 -> k=2, i=2 -> W[2*3+2]=W[8]
    ])
    
    # 8. A2 B2 (Strassen)
    # Indices in A2: (1,1), (1,2), (2,1), (2,2).
    # Indices in B2: (1,1), (1,2), (2,1), (2,2).
    # Indices in C22: (1,1), (1,2), (2,1), (2,2).
    
    # M1 = (A11 + A22)(B11 + B22)
    terms.append([
        [(1, 1, 1), (2, 2, 1)],
        [(1, 1, 1), (2, 2, 1)],
        [(1, 1, 1), (2, 2, 1)] # C11, C22 of block
    ])
    
    # M2 = (A21 + A22) B11
    terms.append([
        [(2, 1, 1), (2, 2, 1)],
        [(1, 1, 1)],
        [(2, 1, 1), (2, 2, 1)] # C21, C22 of block?
        # C21 += M2. C22 -= M2.
        # So W[1,2]=1, W[2,2]=-1.
    ])
    # Correct M2 W coeffs:
    terms[-1] = [
        [(2, 1, 1), (2, 2, 1)],
        [(1, 1, 1)],
        [(1, 2, 1), (2, 2, -1)]
    ]
    
    # M3 = A11 (B12 - B22)
    terms.append([
        [(1, 1, 1)],
        [(1, 2, 1), (2, 2, -1)],
        [(1, 1, 1), (1, 2, 1)] # C11 += M3, C12 += M3
    ])
    
    # M4 = A22 (B21 - B11)
    terms.append([
        [(2, 2, 1)],
        [(2, 1, 1), (1, 1, -1)],
        [(1, 1, 1), (2, 1, 1)] # C11 += M4, C21 += M4
    ])
    
    # M5 = (A11 + A12) B22
    terms.append([
        [(1, 1, 1), (1, 2, 1)],
        [(2, 2, 1)],
        [(1, 1, -1), (1, 2, 1)] # C11 -= M5, C12 += M5
    ])
    
    # M6 = (A21 - A11) (B11 + B12)
    terms.append([
        [(2, 1, 1), (1, 1, -1)],
        [(1, 1, 1), (1, 2, 1)],
        [(2, 2, 1)] # C22 += M6
    ])
    
    # M7 = (A12 - A22) (B21 + B22)
    terms.append([
        [(1, 2, 1), (2, 2, -1)],
        [(2, 1, 1), (2, 2, 1)],
        [(1, 1, 1), (2, 2, 1)] # C11 += M7, C22 += M7
    ])
    
    # Fill U, V, W
    for r, term in enumerate(terms):
        for i, j, coeff in term[0]:
            U[i * m + j][r] = coeff
        for j, k, coeff in term[1]:
            V[j * p + k][r] = coeff
        for k, i, coeff in term[2]:
            W[k * n + i][r] = coeff
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
