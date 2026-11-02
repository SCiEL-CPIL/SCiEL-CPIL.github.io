import time
import numpy as np

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    Split B (3x10) into five 3x2 blocks.
    For each block, multiply A (2x3) by B_block (3x2).
    - Use Strassen (rank 7) for the top-left 2x2 submatrices of A and B_block.
    - Use Schoolbook (rank 2) for the remaining row of A (index 2) and corresponding row of B_block.
    Total rank = 5 * (7 + 2) = 45? 
    Wait, previous analysis said 55.
    Let's re-verify the schoolbook rank for 2x1x2.
    A is 2x3. B is 3x2.
    A = [A00 A01 A02; A10 A11 A12]
    B = [B00 B01; B10 B11; B20 B21]
    C = A B
    C00 = A00B00 + A01B10 + A02B20
    C01 = A00B01 + A01B11 + A02B21
    C10 = A10B00 + A11B10 + A12B20
    C11 = A10B01 + A11B11 + A12B21
    
    Strassen on A_top (2x2) and B_top (2x2) gives C_top (2x2) using 7 terms.
    These terms compute A00B00, A01B10, A10B00, A11B10 etc?
    Strassen computes the product of two 2x2 matrices.
    Let A_s = [A00 A01; A10 A11] and B_s = [B00 B01; B10 B11].
    Product C_s = A_s B_s = [A00B00+A01B10, A00B01+A01B11; A10B00+A11B10, A10B01+A11B11].
    This covers all terms except those involving A02, A12 and B20, B21.
    Remaining terms:
    C00 += A02 B20
    C01 += A02 B21
    C10 += A12 B20
    C11 += A12 B21
    
    We need to add these 4 products.
    Can we do it with rank 4?
    Yes, 4 schoolbook terms:
    t1: U[A02]=1, V[B20]=1, W[C00]=1
    t2: U[A02]=1, V[B21]=1, W[C01]=1
    t3: U[A12]=1, V[B20]=1, W[C10]=1
    t4: U[A12]=1, V[B21]=1, W[C11]=1
    
    So rank per block is 7 + 4 = 11.
    Total rank = 5 * 11 = 55.
    """
    start_time = time.time()
    
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # A indices: row * m + col
    # A: 2x3
    # 0:00, 1:01, 2:02
    # 3:10, 4:11, 5:12
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices: row * p + col
        # B: 3x10
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        b20, b21 = 20 + k0, 20 + k1
        
        # W indices: col * n + row (trace convention W[k*n+i, r])
        # C: 2x10
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # 1. Strassen for 2x2 x 2x2
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # 2. Schoolbook for remaining terms (A02*B20, A02*B21, A12*B20, A12*B21)
        # Term A02 * B20 -> C00
        U[a02, col] += 1.0
        V[b20, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # Term A02 * B21 -> C01
        U[a02, col] += 1.0
        V[b21, col] += 1.0
        W[c01, col] += 1.0
        col += 1
        
        # Term A12 * B20 -> C10
        U[a12, col] += 1.0
        V[b20, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # Term A12 * B21 -> C11
        U[a12, col] += 1.0
        V[b21, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
