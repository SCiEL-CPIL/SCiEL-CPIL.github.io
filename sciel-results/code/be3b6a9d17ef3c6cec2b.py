import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Returns a rank-26 exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses Strassen-based block decomposition:
    A (3x3) split into blocks:
      A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    B (3x4) split into blocks:
      B11 (2x2), B12 (2x2), B21 (1x2), B22 (1x2)
    C (3x4) blocks:
      C11 (2x2), C12 (2x2), C21 (1x2), C22 (1x2)
    
    Rank contributions:
      C11: 7 (Strassen 2x2x2)
      C12: 7 (Strassen 2x2x2)
      C21: 4 (rank 1x2x2)
      C22: 4 (rank 1x2x2)
      Cross-terms A12*B21, A12*B22, A21*B11, A21*B12: 4 each? No...
      
    Actually let me re-derive this carefully.
    
    Standard block multiplication:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Strassen for C11 (2x2 = 2x2 * 2x2): 7 multiplications
    Strassen for C12 (2x2 = 2x2 * 2x2): 7 multiplications
    C21 (1x2 = 1x2 * 2x2 + 1x1 * 1x2): need to compute
    C22 (1x2 = 1x2 * 2x2 + 1x1 * 1x2): need to compute
    
    For C21 = A21*B11 + A22*B21:
      A21 is 1x2, B11 is 2x2, A22 is 1x1, B21 is 1x2
      A21*B11: (1x2)*(2x2) = 1x2, needs 4 mults (schoolbook)
      A22*B21: (1x1)*(1x2) = 1x2, needs 2 mults
      Total for C21: 6
    
    Similarly C22: 6
    
    Total: 7+7+6+6 = 26
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    rank = 26
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 9 x 26
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 12 x 26
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 12 x 26
    
    # Indexing:
    # A is 3x3: A[i][j], i in 0..2, j in 0..2
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, col 2 (2x1)
    #   A21: row 2, cols 0-1 (1x2)
    #   A22: row 2, col 2 (1x1)
    # 
    # B is 3x4: B[j][k], j in 0..2, k in 0..3
    #   B11: rows 0-1, cols 0-1 (2x2)
    #   B12: rows 0-1, cols 2-3 (2x2)
    #   B21: row 2, cols 0-1 (1x2)
    #   B22: row 2, cols 2-3 (1x2)
    #
    # C is 3x4: C[i][k], i in 0..2, k in 0..3
    #   C11: rows 0-1, cols 0-1 (2x2)
    #   C12: rows 0-1, cols 2-3 (2x2)
    #   C21: row 2, cols 0-1 (1x2)
    #   C22: row 2, cols 2-3 (1x2)
    #
    # W index: W[k*n + i] = W[k*3 + i]
    # U index: U[i*m + j] = U[i*3 + j]
    # V index: V[j*p + k] = V[j*4 + k]
    
    col = 0
    
    # Helper to add to matrices
    def add_U(u_idx, val):
        U[u_idx][col] += val
    
    def add_V(v_idx, val):
        V[v_idx][col] += val
    
    def add_W(w_idx, val):
        W[w_idx][col] += val
    
    # C11 = A11*B11 + A12*B21 (2x2 result)
    # Strassen for the A11*B11 part (2x2x2):
    # Let A11 = [[a00, a01], [a10, a11]]
    # Let B11 = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C11_00 = M1 + M4 - M5 + M7
    # C11_01 = M3 + M5
    # C11_10 = M2 + M4
    # C11_11 = M1 - M2 + M3 + M6
    
    # U indices for A11:
    # a00 -> U[0*3+0] = U[0]
    # a01 -> U[0*3+1] = U[1]
    # a10 -> U[1*3+0] = U[3]
    # a11 -> U[1*3+1] = U[4]
    
    # V indices for B11:
    # b00 -> V[0*4+0] = V[0]
    # b01 -> V[0*4+1] = V[1]
    # b10 -> V[1*4+0] = V[4]
    # b11 -> V[1*4+1] = V[5]
    
    # W indices for C11:
    # C11_00 -> W[0*3+0] = W[0]
    # C11_01 -> W[1*3+0] = W[1]
    # C11_10 -> W[0*3+1] = W[3]
    # C11_11 -> W[1*3+1] = W[4]
    
    # M1
    add_U(0, 1); add_U(4, 1)
    add_V(0, 1); add_V(5, 1)
    add_W(0, 1); add_W(4, 1)
    col += 1
    
    # M2
    add_U(3, 1); add_U(4, 1)
    add_V(0, 1)
    add_W(3, 1); add_W(4, -1)
    col += 1
    
    # M3
    add_U(0, 1)
    add_V(1, 1); add_V(5, -1)
    add_W(1, 1); add_W(4, 1)
    col += 1
    
    # M4
    add_U(4, 1)
    add_V(4, 1); add_V(0, -1)
    add_W(0, 1); add_W(3, 1)
    col += 1
    
    # M5
    add_U(0, 1); add_U(1, 1)
    add_V(5, 1)
    add_W(0, -1); add_W(1, 1)
    col += 1
    
    # M6
    add_U(3, 1); add_U(0, -1)
    add_V(0, 1); add_V(1, 1)
    add_W(4, 1)
    col += 1
    
    # M7
    add_U(1, 1); add_U(4, -1)
    add_V(4, 1); add_V(5, 1)
    add_W(0, 1)
    col += 1
    
    # Now add A12*B21 contribution to C11
    # A12: a02 -> U[0*3+2] = U[2], a12 -> U[1*3+2] = U[5]
    # B21: b20 -> V[2*4+0] = V[8], b21 -> V[2*4+1] = V[9]
    # C11_00 += a02*b20
    # C11_01 += a02*b21
    # C11_10 += a12*b20
    # C11_11 += a12*b21
    
    add_U(2, 1); add_V(8, 1); add_W(0, 1)
    col += 1
    
    add_U(2, 1); add_V(9, 1); add_W(1, 1)
    col += 1
    
    add_U(5, 1); add_V(8, 1); add_W(3, 1)
    col += 1
    
    add_U(5, 1); add_V(9, 1); add_W(4, 1)
    col += 1
    
    # C12 = A11*B12 + A12*B22 (2x2 result)
    # B12: b02 -> V[0*4+2] = V[2], b03 -> V[0*4+3] = V[3]
    #       b12 -> V[1*4+2] = V[6], b13 -> V[1*4+3] = V[7]
    # B22: b22 -> V[2*4+2] = V[10], b23 -> V[2*4+3] = V[11]
    # C12_00 -> W[0*3+2] = W[2]
    # C12_01 -> W[1*3+2] = W[5]
    # C12_10 -> W[0*3+3] = W[6]
    # C12_11 -> W[1*3+3] = W[7]
    
    # Strassen for A11*B12:
    # M1 = (a00+a11)(b02+b12)
    add_U(0, 1); add_U(4, 1)
    add_V(2, 1); add_V(6, 1)
    add_W(2, 1); add_W(7, 1)
    col += 1
    
    # M2 = (a10+a11)b02
    add_U(3, 1); add_U(4, 1)
    add_V(2, 1)
    add_W(6, 1); add_W(7, -1)
    col += 1
    
    # M3 = a00(b03-b13)
    add_U(0, 1)
    add_V(3, 1); add_V(7, -1)
    add_W(5, 1); add_W(7, 1)
    col += 1
    
    # M4 = a11(b12-b02)
    add_U(4, 1)
    add_V(6, 1); add_V(2, -1)
    add_W(2, 1); add_W(6, 1)
    col += 1
    
    # M5 = (a00+a01)b13
    add_U(0, 1); add_U(1, 1)
    add_V(7, 1)
    add_W(2, -1); add_W(5, 1)
    col += 1
    
    # M6 = (a10-a00)(b02+b03)
    add_U(3, 1); add_U(0, -1)
    add_V(2, 1); add_V(3, 1)
    add_W(7, 1)
    col += 1
    
    # M7 = (a01-a11)(b12+b13)
    add_U(1, 1); add_U(4, -1)
    add_V(6, 1); add_V(7, 1)
    add_W(2, 1)
    col += 1
    
    # A12*B22 contribution to C12:
    add_U(2, 1); add_V(10, 1); add_W(2, 1)
    col += 1
    
    add_U(2, 1); add_V(11, 1); add_W(5, 1)
    col += 1
    
    add_U(5, 1); add_V(10, 1); add_W(6, 1)
    col += 1
    
    add_U(5, 1); add_V(11, 1); add_W(7, 1)
    col += 1
    
    # C21 = A21*B11 + A22*B21 (1x2 result)
    # A21: a20 -> U[2*3+0] = U[6], a21 -> U[2*3+1] = U[7]
    # A22: a22 -> U[2*3+2] = U[8]
    # C21_00 -> W[0*3+2] = W[8]
    # C21_01 -> W[1*3+2] = W[9]
    
    # A21*B11:
    # C21_00 += a20*b00 + a21*b10
    # C21_01 += a20*b01 + a21*b11
    
    add_U(6, 1); add_V(0, 1); add_W(8, 1)
    col += 1
    
    add_U(7, 1); add_V(4, 1); add_W(8, 1)
    col += 1
    
    add_U(6, 1); add_V(1, 1); add_W(9, 1)
    col += 1
    
    add_U(7, 1); add_V(5, 1); add_W(9, 1)
    col += 1
    
    # A22*B21:
    # C21_00 += a22*b20
    # C21_01 += a22*b21
    
    add_U(8, 1); add_V(8, 1); add_W(8, 1)
    col += 1
    
    add_U(8, 1); add_V(9, 1); add_W(9, 1)
    col += 1
    
    # C22 = A21*B12 + A22*B22 (1x2 result)
    # C22_00 -> W[0*3+3] = W[10]
    # C22_01 -> W[1*3+3] = W[11]
    
    # A21*B12:
    # C22_00 += a20*b02 + a21*b12
    # C22_01 += a20*b03 + a21*b13
    
    add_U(6, 1); add_V(2, 1); add_W(10, 1)
    col += 1
    
    add_U(7, 1); add_V(6, 1); add_W(10, 1)
    col += 1
    
    add_U(6, 1); add_V(3, 1); add_W(11, 1)
    col += 1
    
    add_U(7, 1); add_V(7, 1); add_W(11, 1)
    col += 1
    
    # A22*B22:
    # C22_00 += a22*b22
    # C22_01 += a22*b23
    
    add_U(8, 1); add_V(10, 1); add_W(10, 1)
    col += 1
    
    add_U(8, 1); add_V(11, 1); add_W(11, 1)
    col += 1
    
    # Verify col == rank
    assert col == 26, f"Expected 26 columns, got {col}"
    
    # Convert to lists of ints (all entries are integers)
    U_list = [[int(U[i][j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[int(V[i][j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[int(W[i][j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Quick self-test
if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randint(-5, 5, size=(n, m))
    B = np.random.randint(-5, 5, size=(m, p))
    C_expected = A @ B
    
    # Compute using decomposition
    C_computed = np.zeros((n, p))
    for c in range(rank):
        u_vec = np.array(U)[:, c]
        v_vec = np.array(V)[:, c]
        w_vec = np.array(W)[:, c]
        scalar = float(np.dot(u_vec, A.flatten()) * np.dot(v_vec, B.flatten()))
        # Distribute to C
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += w_vec[k * n + i] * scalar
    
    print("Expected:")
    print(C_expected)
    print("Computed:")
    print(C_computed)
    print("Max error:", np.max(np.abs(C_expected - C_computed)))
    print("Rank:", rank)
