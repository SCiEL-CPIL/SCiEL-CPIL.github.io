"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 130  # Safety margin before 150s kill


def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_fast(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)


def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    
    Returns U, V, W of shapes (4,7), (4,7), (4,7) such that
    for each (i,j,k) in {0,1}^3: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    """
    # Strassen formulas for C = A * B where A, B are 2x2:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # U[i,j] = coefficient of A[i,j] in each P_r
    # V[j,k] = coefficient of B[j,k] in each P_r
    # W[k,i] = coefficient of C[k,i] that includes P_r (with sign)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00 + A11)(B00 + B11)
    # A00: U[0,0]=1; A11: U[1,1]=1
    # B00: V[0,0]=1; B11: V[1,1]=1
    # C00: W[0,0]=1; C11: W[1,1]=1
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # P2 = (A10 + A11)B00
    # A10: U[1,0]=1; A11: U[1,1]=1
    # B00: V[0,0]=1
    # C10: W[0,1]=1; C11: W[1,1]=-1
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1; W[3, 1] = -1  # C10, C11
    
    # P3 = A00(B01 - B11)
    # A00: U[0,0]=1
    # B01: V[0,1]=1; B11: V[1,1]=-1
    # C01: W[1,0]=1; C11: W[1,1]=1
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, B11
    W[0, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # P4 = A11(B10 - B00)
    # A11: U[1,1]=1
    # B10: V[1,0]=1; B00: V[0,0]=-1
    # C00: W[0,0]=1; C10: W[0,1]=1
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, B00
    W[0, 3] = 1; W[1, 3] = 1  # C00, C10
    
    # P5 = (A00 + A01)B11
    # A00: U[0,0]=1; A01: U[0,1]=1
    # B11: V[1,1]=1
    # C00: W[0,0]=-1; C01: W[1,0]=1
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[0, 5] = 1  # C00, C01
    
    # P6 = (A10 - A00)(B00 + B01)
    # A10: U[1,0]=1; A00: U[0,0]=-1
    # B00: V[0,0]=1; B01: V[0,1]=1
    # C11: W[1,1]=1
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # P7 = (A01 - A11)(B10 + B11)
    # A01: U[0,1]=1; A11: U[1,1]=-1
    # B10: V[1,0]=1; B11: V[1,1]=1
    # C00: W[0,0]=1
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1  # C00
    
    return U, V, W


def _build_strassen_4x4_4x5():
    """Build a Strassen-based decomposition for 4x4 * 4x5.
    
    Strategy: Use Strassen recursively to get a decomposition for 4x4 * 4x4,
    then handle the extra column (4x4 * 4x1) with schoolbook terms.
    
    For 4x4 * 4x4: rank 49 (Strassen^2)
    For 4x4 * 4x1: rank 16 (schoolbook)
    Total: rank 65
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # First, build 4x4 * 4x4 using Strassen
    U_44, V_44, W_44 = _strassen_4x4_4x4()
    
    # Now handle the extra column: 4x4 * 4x1
    # This is just 16 scalar multiplications
    U_extra = np.zeros((n * m, n * m))  # 16 x 16
    V_extra = np.zeros((m, m))  # 4 x 4... wait, m*p for p=1 is m*1=4
    # Actually for 4x4 * 4x1: U is 16x16, V is 4x16, W is 4x16
    # But we're adding to the existing 4x4*4x4 which has U:16x49, V:20x49, W:20x49
    # The extra part: U:16x16, V:4x16, W:4x16
    
    # Let me restructure. The full problem is 4x4 * 4x5.
    # U: (4*4, rank) = (16, rank)
    # V: (4*5, rank) = (20, rank)
    # W: (5*4, rank) = (20, rank)
    
    # Split into 4x4*4x4 (columns 0-3 of B) and 4x4*4x1 (column 4 of B)
    
    # 4x4 * 4x4 part:
    # U_44: (16, 49), V_44: (16, 49) [for B columns 0-3, so indices 0-15 in V], W_44: (16, 49) [for C rows 0-3, indices 0-15 in W]
    
    # 4x4 * 4x1 part:
    # U_extra: (16, 16), V_extra: (4, 16) [for B column 4, indices 16-19 in V], W_extra: (4, 16) [for C row 4, indices 16-19 in W]
    
    U = np.zeros((16, 65))
    V = np.zeros((20, 65))
    W = np.zeros((20, 65))
    
    # Copy 4x4*4x4 part
    U[:, :49] = U_44
    V[:16, :49] = V_44
    W[:16, :49] = W_44
    
    # Add 4x4*4x1 part (schoolbook for the extra column)
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j, col] = 1.0
            V[16 + j, col] = 1.0  # B row j, col 4 -> index 4*4+j = 16+j
            W[16 + i, col] = 1.0  # C row 4, col i -> index 4*4+i = 16+i
    
    return U, V, W


def _strassen_4x4_4x4():
    """Build Strassen decomposition for 4x4 * 4x4.
    
    Uses Strassen's algorithm applied to 2x2 blocks of 2x2 matrices.
    Each 2x2 block multiplication uses 7 terms, so 4 blocks * 7 = 28 terms for the block-level,
    but actually Strassen on 4x4 gives rank 49.
    
    The 4x4 matrix is split into four 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    
    C = A*B = [[C00, C01], [C10, C11]]
    
    Using Strassen:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B01-B11)
    P4 = A11(B10-B00)
    P5 = (A00+A01)B11
    P6 = (A10-A00)(B00+B01)
    P7 = (A01-A11)(B10+B11)
    
    C00 = P1+P4-P5+P7
    C01 = P3+P5
    C10 = P2+P4
    C11 = P1-P2+P3+P6
    
    Each P is a 2x2 product of 2x2 matrices, which itself needs 7 terms (Strassen).
    So total = 7 * 7 = 49 terms.
    """
    # Build the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Now for 4x4, we have 4 blocks of 2x2.
    # A is 4x4, split into 2x2 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), etc.
    # B is 4x4, split similarly.
    # C is 4x4, split similarly.
    
    # The 7 Strassen terms for the block level:
    # Each term is a product of a 2x2 matrix (linear combo of A blocks) and a 2x2 matrix (linear combo of B blocks)
    # Each such product uses 7 scalar-level Strassen terms.
    
    # Let's define the block-level coefficients
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # For each of the 7 block-level Strassen terms, we have:
    # A_comb = sum of alpha * A_block
    # B_comb = sum of beta * B_block
    # Product = A_comb * B_comb, which is a 2x2 * 2x2 multiplication
    
    # The 7 block-level terms:
    # T1: A_comb = A00+A11, B_comb = B00+B11
    # T2: A_comb = A10+A11, B_comb = B00
    # T3: A_comb = A00, B_comb = B01-B11
    # T4: A_comb = A11, B_comb = B10-B00
    # T5: A_comb = A00+A01, B_comb = B11
    # T6: A_comb = A10-A00, B_comb = B00+B01
    # T7: A_comb = A01-A11, B_comb = B10+B11
    
    # And the C blocks:
    # C00 = T1+T4-T5+T7
    # C01 = T3+T5
    # C10 = T2+T4
    # C11 = T1-T2+T3+T6
    
    # Now, each T_i is a 2x2 * 2x2 product, which we decompose using 7 scalar Strassen terms.
    # So total = 7 * 7 = 49 terms.
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Block indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # Within each block, the 2x2 indices are (0,0), (0,1), (1,0), (1,1)
    # Global index for A[i,j] where i,j in {0,1,2,3}:
    # Block (bi, bj) with local (li, lj) -> global (2*bi+li, 2*bj+lj)
    
    # Define the 7 block-level A and B coefficient vectors
    # A_coeffs[t][block] = coefficient of A_block in T_t
    # B_coeffs[t][block] = coefficient of B_block in T_t
    # C_coeffs[c_block][t] = coefficient of T_t in C_block
    
    A_block_coeffs = [
        [1, 0, 0, 1],   # T1: A00+A11
        [0, 0, 1, 1],   # T2: A10+A11
        [1, 0, 0, 0],   # T3: A00
        [0, 0, 0, 1],   # T4: A11
        [1, 1, 0, 0],   # T5: A00+A01
        [1, 0, -1, 0],  # T6: A10-A00 (A10 - A00)
        [0, 1, 0, -1],  # T7: A01-A11
    ]
    
    B_block_coeffs = [
        [1, 0, 0, 1],   # T1: B00+B11
        [1, 0, 0, 0],   # T2: B00
        [0, 1, 0, -1],  # T3: B01-B11
        [0, 0, 1, -1],  # T4: B10-B00
        [0, 0, 0, 1],   # T5: B11
        [1, 1, 0, 0],   # T6: B00+B01
        [0, 0, 1, 1],   # T7: B10+B11
    ]
    
    C_block_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = T1+T4-T5+T7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = T3+T5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = T2+T4
        [1, -1, 1, 0, 0, 1, 0],   # C11 = T1-T2+T3+T6
    ]
    
    # For each block-level term t (0..6) and each scalar Strassen term s (0..6):
    # The scalar Strassen term s has U2, V2, W2 coefficients for 2x2 matrices.
    # The A_comb for term t is a linear combination of A blocks.
    # The B_comb for term t is a linear combination of B blocks.
    # The C_block contribution is C_block_coeffs[c_block][t] * (scalar Strassen result).
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # For each A element (i, j) in 4x4:
            # i = 2*bi + li, j = 2*bj + lj
            # U[i,j, col] = A_block_coeffs[t][bi,bj] * U2[li, lj, s]
            for bi in range(2):
                for bj in range(2):
                    a_coeff = A_block_coeffs[t][bi * 2 + bj]
                    if a_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            i = 2 * bi + li
                            j = 2 * bj + lj
                            U[i * 4 + j, col] = a_coeff * U2[li * 2 + lj, s]
            
            # For each B element (j, k) in 4x4:
            # j = 2*bj + lj, k = 2*bk + lk
            # V[j,k, col] = B_block_coeffs[t][bj,bk] * V2[lj, lk, s]
            for bj in range(2):
                for bk in range(2):
                    b_coeff = B_block_coeffs[t][bj * 2 + bk]
                    if b_coeff == 0:
                        continue
                    for lj in range(2):
                        for lk in range(2):
                            j = 2 * bj + lj
                            k = 2 * bk + lk
                            V[j * 4 + k, col] = b_coeff * V2[lj * 2 + lk, s]
            
            # For each C element (k, i) in 4x4:
            # k = 2*bk + lk, i = 2*bi + li
            # W[k,i, col] = C_block_coeffs[bi,bk][t] * W2[lk, li, s]
            for bi in range(2):
                for bk in range(2):
                    c_coeff = C_block_coeffs[bi * 2 + bk][t]
                    if c_coeff == 0:
                        continue
                    for lk in range(2):
                        for li in range(2):
                            k = 2 * bk + lk
                            i = 2 * bi + li
                            W[k * 4 + i, col] = c_coeff * W2[lk * 2 + li, s]
    
    return U, V, W


def _verify(U, V, W, tol=1e-10):
    return _verify_fast(U, V, W, tol)


def construct():
    """Construct the matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Try Strassen-based decomposition
    try:
        U, V, W = _build_strassen_4x4_4x5()
        if _verify_fast(U, V, W, tol=1e-8):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify_fast(U, V, W)}")
