"""Matrix multiplication 2,4,6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
import random

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term has U (4 coeffs for A), V (4 coeffs for B), W (4 coeffs for C).
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C[i][j] = sum_r W[r][i*2+j] * (U[r] dot A_flat) * (V[r] dot B_flat)
    """
    terms = []
    # P1 = (a+d)(e+h)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # P2 = (c+d)(e)
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # P3 = a(f-h)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # P4 = d(g-e)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]))
    # P5 = (a+b)(h)
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # P6 = (c-a)(e+f)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]))
    # P7 = (b-d)(g+h)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]))
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition numerically."""
    terms = _strassen_2x2x2_terms()
    for trial in range(200):
        a, b, c, d = [random.randint(-10, 10) for _ in range(4)]
        e, f, g, h = [random.randint(-10, 10) for _ in range(4)]
        A_flat = [a, b, c, d]
        B_flat = [e, f, g, h]
        
        C00 = a*e + b*f
        C01 = a*g + b*h
        C10 = c*e + d*f
        C11 = c*g + d*h
        expected = [C00, C01, C10, C11]
        
        computed = [0, 0, 0, 0]
        for U, V, W in terms:
            u_val = sum(U[i] * A_flat[i] for i in range(4))
            v_val = sum(V[i] * B_flat[i] for i in range(4))
            for j in range(4):
                computed[j] += W[j] * u_val * v_val
        
        if computed != expected:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _strassen_2x2x4_terms():
    """
    Find a rank-minimal decomposition for 2x2 times 2x4 matrix multiplication.
    A is 2x2, B is 2x4, C is 2x4.
    
    We can use the fact that 2x2x4 = 2x2x2 + 2x2x2, so rank <= 14.
    But maybe we can do better. Let's try to find a rank-10 or lower decomposition.
    
    Actually, let's just use two copies of Strassen 2x2x2, giving rank 14.
    But we might be able to share some terms.
    
    For now, let's implement the straightforward rank-14 version and see if we can improve.
    """
    # Use two Strassen 2x2x2 decompositions for the two 2x2 blocks of B
    terms = []
    strassen = _strassen_2x2x2_terms()
    
    # For B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # Block 1: B1 = [[b00, b01], [b10, b11]], output C1 = [[c00, c01], [c10, c11]]
    # Block 2: B2 = [[b02, b03], [b12, b13]], output C2 = [[c02, c03], [c12, c13]]
    
    for t in range(7):
        U_t, V_t, W_t = strassen[t]
        # V_t for B1: [b00, b01, b10, b11] -> indices [0, 1, 4, 5] in 2x4 flat
        V1 = [0, 0, 0, 0, 0, 0, 0, 0]
        V1[0] = V_t[0]  # b00
        V1[1] = V_t[1]  # b01
        V1[4] = V_t[2]  # b10
        V1[5] = V_t[3]  # b11
        # W for C1: [c00, c01, c10, c11] -> indices [0, 1, 4, 5] in 2x4 flat
        W1 = [0, 0, 0, 0, 0, 0, 0, 0]
        W1[0] = W_t[0]  # c00
        W1[1] = W_t[1]  # c01
        W1[4] = W_t[2]  # c10
        W1[5] = W_t[3]  # c11
        terms.append((U_t, V1, W1))
    
    for t in range(7):
        U_t, V_t, W_t = strassen[t]
        # V_t for B2: [b02, b03, b12, b13] -> indices [2, 3, 6, 7] in 2x4 flat
        V2 = [0, 0, 0, 0, 0, 0, 0, 0]
        V2[2] = V_t[0]  # b02
        V2[3] = V_t[1]  # b03
        V2[6] = V_t[2]  # b12
        V2[7] = V_t[3]  # b13
        # W for C2: [c02, c03, c12, c13] -> indices [2, 3, 6, 7] in 2x4 flat
        W2 = [0, 0, 0, 0, 0, 0, 0, 0]
        W2[2] = W_t[0]  # c02
        W2[3] = W_t[1]  # c03
        W2[6] = W_t[2]  # c12
        W2[7] = W_t[3]  # c13
        terms.append((U_t, V2, W2))
    
    return terms

def _verify_2x2x4():
    """Verify the 2x2x4 decomposition numerically."""
    terms = _strassen_2x2x4_terms()
    for trial in range(100):
        a, b, c, d = [random.randint(-5, 5) for _ in range(4)]
        B = [[random.randint(-5, 5) for _ in range(4)] for _ in range(2)]
        A_flat = [a, b, c, d]
        B_flat = [B[0][0], B[0][1], B[0][2], B[0][3], B[1][0], B[1][1], B[1][2], B[1][3]]
        
        # Expected C
        C00 = a*B[0][0] + b*B[1][0]
        C01 = a*B[0][1] + b*B[1][1]
        C02 = a*B[0][2] + b*B[1][2]
        C03 = a*B[0][3] + b*B[1][3]
        C10 = c*B[0][0] + d*B[1][0]
        C11 = c*B[0][1] + d*B[1][1]
        C12 = c*B[0][2] + d*B[1][2]
        C13 = c*B[0][3] + d*B[1][3]
        expected = [C00, C01, C02, C03, C10, C11, C12, C13]
        
        # Computed C
        computed = [0, 0, 0, 0, 0, 0, 0, 0]
        for U, V, W in terms:
            u_val = sum(U[i] * A_flat[i] for i in range(4))
            v_val = sum(V[i] * B_flat[i] for i in range(8))
            for j in range(8):
                computed[j] += W[j] * u_val * v_val
        
        if computed != expected:
            return False
    return True

def construct():
    """
    Construct a rank-42 decomposition for 2x4x6 matrix multiplication using
    Strassen-based blocks.
    
    A is 2x4, B is 4x6, C is 2x6.
    Split A into two 2x2 blocks: A1 = A[:, 0:2], A2 = A[:, 2:4]
    Split B into two 2x6 blocks: B1 = B[0:2, :], B2 = B[2:4, :]
    C = A1*B1 + A2*B2
    
    For each 2x2x6 product, split into three 2x2x2 products and use Strassen.
    Total rank: 2 * 3 * 7 = 42
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    strassen_terms = _strassen_2x2x2_terms()
    
    # Verify Strassen
    if not _verify_strassen():
        return _schoolbook_decomposition()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    for a_block in range(2):  # A1, A2
        for b_block in range(3):  # three 2x2 sub-blocks of B
            for t in range(7):
                col = a_block * 21 + b_block * 7 + t
                U_t, V_t, W_t = strassen_terms[t]
                
                # A_flat indices for the 2x2 A block
                if a_block == 0:
                    a_indices = [0, 1, 4, 5]
                else:
                    a_indices = [2, 3, 6, 7]
                for i in range(4):
                    U[a_indices[i]][col] = U_t[i]
                
                # B_flat indices for the 2x2 B block
                b_row_start = a_block * 2
                b_col_start = b_block * 2
                b_indices = [
                    b_row_start * 6 + b_col_start,
                    b_row_start * 6 + b_col_start + 1,
                    (b_row_start + 1) * 6 + b_col_start,
                    (b_row_start + 1) * 6 + b_col_start + 1
                ]
                for i in range(4):
                    V[b_indices[i]][col] = V_t[i]
                
                # W_flat indices for the 2x2 C block
                w_indices = [
                    2 * b_block,
                    2 * b_block + 1,
                    6 + 2 * b_block,
                    6 + 2 * b_block + 1
                ]
                for i in range(4):
                    W[w_indices[i]][col] = W_t[i]
    
    # Verify the full decomposition
    for trial in range(100):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return _schoolbook_decomposition()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
