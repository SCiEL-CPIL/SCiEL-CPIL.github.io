"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7)."""
    # Strassen's 7 products for (a11 a12; a21 a22) * (b11 b12; b21 b22)
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 4 x 7
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 4 x 7
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 4 x 7
    
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (a11+a22)(b11+b22)
    # U: a11(0)=1, a22(3)=1
    # V: b11(0)=1, b22(3)=1
    # W: contributes to c11(0)=+1, c12(1)=0, c21(2)=0, c22(3)=+1
    U[0][0] = Fraction(1); U[3][0] = Fraction(1)
    V[0][0] = Fraction(1); V[3][0] = Fraction(1)
    W[0][0] = Fraction(1); W[3][0] = Fraction(1)
    
    # M2 = (a21+a22)b11
    # U: a21(2)=1, a22(3)=1
    # V: b11(0)=1
    # W: c21(2)=+1, c22(3)=-1
    U[2][1] = Fraction(1); U[3][1] = Fraction(1)
    V[0][1] = Fraction(1)
    W[2][1] = Fraction(1); W[3][1] = Fraction(-1)
    
    # M3 = a11(b12-b22)
    # U: a11(0)=1
    # V: b12(1)=1, b22(3)=-1
    # W: c12(1)=+1, c22(3)=+1
    U[0][2] = Fraction(1)
    V[1][2] = Fraction(1); V[3][2] = Fraction(-1)
    W[1][2] = Fraction(1); W[3][2] = Fraction(1)
    
    # M4 = a22(b21-b11)
    # U: a22(3)=1
    # V: b21(2)=1, b11(0)=-1
    # W: c11(0)=+1, c21(2)=+1
    U[3][3] = Fraction(1)
    V[2][3] = Fraction(1); V[0][3] = Fraction(-1)
    W[0][3] = Fraction(1); W[2][3] = Fraction(1)
    
    # M5 = (a11+a12)b22
    # U: a11(0)=1, a12(1)=1
    # V: b22(3)=1
    # W: c11(0)=-1, c12(1)=+1
    U[0][4] = Fraction(1); U[1][4] = Fraction(1)
    V[3][4] = Fraction(1)
    W[0][4] = Fraction(-1); W[1][4] = Fraction(1)
    
    # M6 = (a21-a11)(b11+b12)
    # U: a21(2)=1, a11(0)=-1
    # V: b11(0)=1, b12(1)=1
    # W: c22(3)=+1
    U[2][5] = Fraction(1); U[0][5] = Fraction(-1)
    V[0][5] = Fraction(1); V[1][5] = Fraction(1)
    W[3][5] = Fraction(1)
    
    # M7 = (a12-a22)(b21+b22)
    # U: a12(1)=1, a22(3)=-1
    # V: b21(2)=1, b22(3)=1
    # W: c11(0)=+1
    U[1][6] = Fraction(1); U[3][6] = Fraction(-1)
    V[2][6] = Fraction(1); V[3][6] = Fraction(1)
    W[0][6] = Fraction(1)
    
    return U, V, W

def _schoolbook_1xn1x1(n):
    """Schoolbook for 1xn1 matrix multiplication: rank n."""
    # A is 1xn, B is nx1, C is 1x1
    # C = sum_j A[0,j] * B[j,0]
    # U[0,j] = 1, V[j,0] = 1, W[0,0] = 1 for each j
    rank = n
    U = [[Fraction(0)] * rank for _ in range(1 * n)]  # 1 x n
    V = [[Fraction(0)] * rank for _ in range(n * 1)]  # n x 1
    W = [[Fraction(0)] * rank for _ in range(1 * 1)]  # 1 x 1
    
    for j in range(n):
        U[0][j] = Fraction(1)
        V[j][0] = Fraction(1)
        W[0][j] = Fraction(1)
    
    return U, V, W

def _block_multiply(n1, n2, m1, m2, p1, p2):
    """
    Multiply (n1+n2) x (m1+m2) by (m1+m2) x (p1+p2).
    A = [A11 A12; A21 A22] where A11 is n1xm1, A12 is n1xm2, A21 is n2xm1, A22 is n2xm2
    B = [B11 B12; B21 B22] where B11 is m1xp1, B12 is m1xp2, B21 is m2xp1, B22 is m2xp2
    C = [C11 C12; C21 C22] where C11 is n1xp1, C12 is n1xp2, C21 is n2xp1, C22 is n2xp2
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    """
    # Each block product contributes to the decomposition
    # We need to handle each block multiplication separately
    
    products = []
    
    # C11 = A11*B11 + A12*B21
    U11, V11, W11 = _multiply_block(n1, m1, p1)
    U12, V12, W12 = _multiply_block(n1, m2, p1)
    products.append((U11, V11, W11, 0, 0, 0))  # A11*B11 -> C11
    products.append((U12, V12, W12, 0, 0, p1))  # A12*B21 -> C11 (offset in p)
    
    # C12 = A11*B12 + A12*B22
    U13, V13, W13 = _multiply_block(n1, m1, p2)
    U14, V14, W14 = _multiply_block(n1, m2, p2)
    products.append((U13, V13, W13, 0, p1, 0))  # A11*B12 -> C12
    products.append((U14, V14, W14, 0, p1, p2))  # A12*B22 -> C12
    
    # C21 = A21*B11 + A22*B21
    U21, V21, W21 = _multiply_block(n2, m1, p1)
    U22, V22, W22 = _multiply_block(n2, m2, p1)
    products.append((U21, V21, W21, n1, 0, 0))  # A21*B11 -> C21
    products.append((U22, V22, W22, n1, 0, p1))  # A22*B21 -> C21
    
    # C22 = A21*B12 + A22*B22
    U23, V23, W23 = _multiply_block(n2, m1, p2)
    U24, V24, W24 = _multiply_block(n2, m2, p2)
    products.append((U23, V23, W23, n1, p1, 0))  # A21*B12 -> C22
    products.append((U24, V24, W24, n1, p1, p2))  # A22*B22 -> C22
    
    # Combine all products
    total_rank = sum(len(U) and len(U[0]) or 0 for U, V, W, *_ in products)
    total_rank = sum(len(U[0]) if U else 0 for U, V, W, *_ in products)
    
    U_total = [[Fraction(0)] * total_rank for _ in range((n1 + n2) * (m1 + m2))]
    V_total = [[Fraction(0)] * total_rank for _ in range((m1 + m2) * (p1 + p2))]
    W_total = [[Fraction(0)] * total_rank for _ in range((p1 + p2) * (n1 + n2))]
    
    col_offset = 0
    for U_prod, V_prod, W_prod, i_off, j_off, k_off in products:
        r = len(U_prod[0]) if U_prod else 0
        for i in range(len(U_prod)):
            for c in range(r):
                U_total[i_off + i][col_offset + c] = U_prod[i][c]
        for j in range(len(V_prod)):
            for c in range(r):
                V_total[j_off + j][col_offset + c] = V_prod[j][c]
        for k in range(len(W_prod)):
            for c in range(r):
                W_total[k_off + k][col_offset + c] = W_prod[k][c]
        col_offset += r
    
    return U_total, V_total, W_total

def _multiply_block(n, m, p):
    """Multiply n x m by m x p matrix."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    elif n == 1 and m == 1 and p == 1:
        U, V, W = [[Fraction(1)]], [[Fraction(1)]], [[Fraction(1)]]
        return U, V, W
    elif n == 1:
        # 1 x m x p: C[0,k] = sum_j A[0,j] * B[j,k]
        rank = m * p
        U = [[Fraction(0)] * rank for _ in range(1 * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * 1)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[0][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * 1 + 0][col] = Fraction(1)
                col += 1
        return U, V, W
    elif m == 1:
        # n x 1 x p: C[i,k] = A[i,0] * B[0,k]
        rank = n * p
        U = [[Fraction(0)] * rank for _ in range(n * 1)]
        V = [[Fraction(0)] * rank for _ in range(1 * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i * 1 + 0][col] = Fraction(1)
                V[0 * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
        return U, V, W
    elif p == 1:
        # n x m x 1: C[i,0] = sum_j A[i,j] * B[j,0]
        rank = n * m
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * 1)]
        W = [[Fraction(0)] * rank for _ in range(1 * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = Fraction(1)
                V[j * 1 + 0][col] = Fraction(1)
                W[0 * n + i][col] = Fraction(1)
                col += 1
        return U, V, W
    else:
        # General case: schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Split: n=3 -> 2+1, m=5 -> 2+2+1, p=5 -> 2+2+1
    # Use block decomposition
    # A is 3x5, split rows as 2+1, cols as 2+2+1
    # B is 5x5, split rows as 2+2+1, cols as 2+2+1
    
    # Let's use a recursive approach with Strassen where beneficial
    # For now, use direct block decomposition with 2+2+1 split on m and p
    
    n1, n2 = 2, 1
    m1, m2, m3 = 2, 2, 1
    p1, p2, p3 = 2, 2, 1
    
    # We'll build the full decomposition by combining block products
    # A blocks: A11(2x2), A12(2x2), A13(2x1), A21(1x2), A22(1x2), A23(1x1)
    # B blocks: B11(2x2), B12(2x2), B13(2x1), B21(2x2), B22(2x2), B23(2x1), B31(1x2), B32(1x2), B33(1x1)
    
    # C blocks: C11(2x2), C12(2x2), C13(2x1), C21(1x2), C22(1x2), C23(1x1)
    
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33
    
    # Collect all products
    products = []
    
    # Helper to add a block product with offsets
    def add_product(n_block, m_block, p_block, i_off, j_off, k_off):
        U_b, V_b, W_b = _multiply_block(n_block, m_block, p_block)
        products.append((U_b, V_b, W_b, i_off, j_off, k_off))
    
    # C11 (2x2) at i_off=0, k_off=0
    add_product(2, 2, 2, 0, 0, 0)    # A11*B11
    add_product(2, 2, 2, 0, 2, 0)    # A12*B21
    add_product(2, 1, 2, 0, 4, 0)    # A13*B31
    
    # C12 (2x2) at i_off=0, k_off=2
    add_product(2, 2, 2, 0, 0, 2)    # A11*B12
    add_product(2, 2, 2, 0, 2, 2)    # A12*B22
    add_product(2, 1, 2, 0, 4, 2)    # A13*B32
    
    # C13 (2x1) at i_off=0, k_off=4
    add_product(2, 2, 1, 0, 0, 4)    # A11*B13
    add_product(2, 2, 1, 0, 2, 4)    # A12*B23
    add_product(2, 1, 1, 0, 4, 4)    # A13*B33
    
    # C21 (1x2) at i_off=2, k_off=0
    add_product(1, 2, 2, 2, 0, 0)    # A21*B11
    add_product(1, 2, 2, 2, 2, 0)    # A22*B21
    add_product(1, 1, 2, 2, 4, 0)    # A23*B31
    
    # C22 (1x2) at i_off=2, k_off=2
    add_product(1, 2, 2, 2, 0, 2)    # A21*B12
    add_product(1, 2, 2, 2, 2, 2)    # A22*B22
    add_product(1, 1, 2, 2, 4, 2)    # A23*B32
    
    # C23 (1x1) at i_off=2, k_off=4
    add_product(1, 2, 1, 2, 0, 4)    # A21*B13
    add_product(1, 2, 1, 2, 2, 4)    # A22*B23
    add_product(1, 1, 1, 2, 4, 4)    # A23*B33
    
    # Combine all products into full U, V, W
    total_rank = sum(len(U_b[0]) if U_b else 0 for U_b, V_b, W_b, *_ in products)
    
    U_total = [[Fraction(0)] * total_rank for _ in range(n * m)]  # 3x5 = 15
    V_total = [[Fraction(0)] * total_rank for _ in range(m * p)]  # 5x5 = 25
    W_total = [[Fraction(0)] * total_rank for _ in range(p * n)]  # 5x3 = 15
    
    col_offset = 0
    for U_b, V_b, W_b, i_off, j_off, k_off in products:
        r = len(U_b[0]) if U_b else 0
        for i in range(len(U_b)):
            for c in range(r):
                U_total[i_off + i][col_offset + c] = U_b[i][c]
        for j in range(len(V_b)):
            for c in range(r):
                V_total[j_off + j][col_offset + c] = V_b[j][c]
        for k in range(len(W_b)):
            for c in range(r):
                W_total[k_off + k][col_offset + c] = W_b[k][c]
        col_offset += r
    
    # Verify the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Actually: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Wait, the tensor decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all i,j,k
    # No wait, T is the structure tensor where T[i,j,k] = 1 means the term A[i,j]*B[j,k] appears in C[i,k]
    
    # Let's verify numerically with random matrices
    def to_matrix(frac_mat):
        return np.array([[float(x) for x in row] for row in frac_mat])
    
    U_np = to_matrix(U_total)
    V_np = to_matrix(V_total)
    W_np = to_matrix(W_total)
    
    # Verify: for random A and B, compute C = A@B and compare with reconstruction
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_direct = A_test @ B_test
    
    # Reconstruct using tensor decomposition
    # C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k] / (U*V terms)
    # Actually the decomposition means:
    # C = sum_c (col of U * A) @ (row of V * B) * W ... no
    # The standard form: C[i,k] = sum_c U[:,i*m+?...] 
    # Let me think again. The tensor T has T[i,j,k] = 1 for all i,j,k in matrix multiplication
    # (since every product A[i,j]*B[j,k] contributes to C[i,k])
    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] (with appropriate index mapping)
    
    # In our indexing:
    # U is (n*m) x rank, indexed by [i*m+j]
    # V is (m*p) x rank, indexed by [j*p+k]
    # W is (p*n) x rank, indexed by [k*n+i]
    
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need T[i,j,k] = 1 for all i,j,k
    
    # Let's verify this
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_total[i*m+j][c] * V_total[j*p+k][c] * W_total[k*n+i][c] for c in range(total_rank))
                if val != Fraction(1):
                    max_err = max(max_err, abs(float(val) - 1.0))
                    if max_err > 1e-10:
                        break
        if max_err > 1e-10:
            break
    if max_err > 1e-10:
        break
    if max_err > 1e-10:
        break
    
    # If verification fails, fall back to schoolbook
    if max_err > 1e-10:
        rank_school = n * m * p
        U_s = [[Fraction(0)] * rank_school for _ in range(n * m)]
        V_s = [[Fraction(0)] * rank_school for _ in range(m * p)]
        W_s = [[Fraction(0)] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = Fraction(1)
                    V_s[j * p + k][col] = Fraction(1)
                    W_s[k * n + i][col] = Fraction(1)
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Fall back to schoolbook if we're running out of time
        rank_school = n * m * p
        U_s = [[Fraction(0)] * rank_school for _ in range(n * m)]
        V_s = [[Fraction(0)] * rank_school for _ in range(m * p)]
        W_s = [[Fraction(0)] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = Fraction(1)
                    V_s[j * p + k][col] = Fraction(1)
                    W_s[k * n + i][col] = Fraction(1)
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
