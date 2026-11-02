"""Tensor decomposition for 2x2 by 2x4 matrix multiplication, rank 14."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    # The rank 14 decomposition is optimal for 2x2 by 2x4 matrix multiplication.
    # This is derived from applying Strassen's algorithm (rank 7) to each 2x2 
    # block of the 2x4 matrix B, resulting in 7 + 7 = 14 products.
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    rank = 14

    # Initialize matrices
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to set values
    def set_u(i, c, val): U[i][c] = val
    def set_v(i, c, val): V[i][c] = val
    def set_w(i, c, val): W[i][c] = val

    # Indices for A (2x2): 
    # A00=0, A01=1, A10=2, A11=3
    # Indices for B (2x4):
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    # Indices for C (2x4, col-major j*2+i):
    # C00=0, C10=1, C01=2, C11=3
    # C02=4, C12=5, C03=6, C13=7

    # Left Half (Columns 0, 1 of B) - Strassen
    # M1_L = (A00+A11)(B00+B11) -> C00+=, C11+=
    c = 0
    set_u(0, c, 1); set_u(3, c, 1)
    set_v(0, c, 1); set_v(5, c, 1)
    set_w(0, c, 1); set_w(3, c, 1)

    # M2_L = (A10+A11)B00 -> C10+=, C11-=
    c = 1
    set_u(2, c, 1); set_u(3, c, 1)
    set_v(0, c, 1)
    set_w(1, c, 1); set_w(3, c, -1)

    # M3_L = A00(B01-B11) -> C01+=, C11+=
    c = 2
    set_u(0, c, 1)
    set_v(1, c, 1); set_v(5, c, -1)
    set_w(2, c, 1); set_w(3, c, 1)

    # M4_L = A11(B10-B00) -> C00+=, C10+=
    c = 3
    set_u(3, c, 1)
    set_v(4, c, 1); set_v(0, c, -1)
    set_w(0, c, 1); set_w(1, c, 1)

    # M5_L = (A00+A01)B11 -> C00-=, C01+=
    c = 4
    set_u(0, c, 1); set_u(1, c, 1)
    set_v(5, c, 1)
    set_w(0, c, -1); set_w(2, c, 1)

    # M6_L = (A10-A00)(B00+B01) -> C11+=
    c = 5
    set_u(2, c, 1); set_u(0, c, -1)
    set_v(0, c, 1); set_v(1, c, 1)
    set_w(3, c, 1)

    # M7_L = (A01-A11)(B10+B11) -> C00+=
    c = 6
    set_u(1, c, 1); set_u(3, c, -1)
    set_v(4, c, 1); set_v(5, c, 1)
    set_w(0, c, 1)

    # Right Half (Columns 2, 3 of B) - Strassen
    # M1_R = (A00+A11)(B02+B13) -> C02+=, C13+=
    c = 7
    set_u(0, c, 1); set_u(3, c, 1)
    set_v(2, c, 1); set_v(7, c, 1)
    set_w(4, c, 1); set_w(7, c, 1)

    # M2_R = (A10+A11)B02 -> C12+=, C13-=
    c = 8
    set_u(2, c, 1); set_u(3, c, 1)
    set_v(2, c, 1)
    set_w(5, c, 1); set_w(7, c, -1)

    # M3_R = A00(B03-B13) -> C03+=, C13+=
    c = 9
    set_u(0, c, 1)
    set_v(3, c, 1); set_v(7, c, -1)
    set_w(6, c, 1); set_w(7, c, 1)

    # M4_R = A11(B12-B02) -> C02+=, C12+=
    c = 10
    set_u(3, c, 1)
    set_v(6, c, 1); set_v(2, c, -1)
    set_w(4, c, 1); set_w(5, c, 1)

    # M5_R = (A00+A01)B13 -> C02-=, C03+=
    c = 11
    set_u(0, c, 1); set_u(1, c, 1)
    set_v(7, c, 1)
    set_w(4, c, -1); set_w(6, c, 1)

    # M6_R = (A10-A00)(B02+B03) -> C13+=
    c = 12
    set_u(2, c, 1); set_u(0, c, -1)
    set_v(2, c, 1); set_v(3, c, 1)
    set_w(7, c, 1)

    # M7_R = (A01-A11)(B12+B13) -> C02+=
    c = 13
    set_u(1, c, 1); set_u(3, c, -1)
    set_v(6, c, 1); set_v(7, c, 1)
    set_w(4, c, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
