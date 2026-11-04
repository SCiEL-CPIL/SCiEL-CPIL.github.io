"""Program to find a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify_numpy(U, V, W, n=4, m=5, p=5, tol=1e-9):
    """Fast numpy verification."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _verify_exact(U, V, W, n=4, m=5, p=5):
    """Exact verification using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != Fraction(1):
                    return False
    return True

def _schoolbook(n=4, m=5, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen's 7-term algorithm for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4,7), (4,7), (4,7)
    where U is indexed by [i*2+j], V by [j*2+k], W by [k*2+i]
    """
    # Strassen terms:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U[i*2+j, r] for i,j in {0,1}
    # V[j*2+k, r] for j,k in {0,1}
    # W[k*2+i, r] for k,i in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
    # A11: U[0,0]=1, A22: U[3,0]=1
    # B11: V[0,0]=1, B22: V[3,0]=1
    # C11: W[0,0]=1, C22: W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11 -> C21 += 1
    # A21: U[2,1]=1, A22: U[3,1]=1
    # B11: V[0,1]=1
    # C21: W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A11*(B12-B22) -> C12 += 1
    # A11: U[0,2]=1
    # B12: V[1,2]=1, B22: V[3,2]=-1
    # C12: W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A22*(B21-B11) -> C11 += 1, C21 += 1
    # A22: U[3,3]=1
    # B21: V[2,3]=1, B11: V[0,3]=-1
    # C11: W[0,3]=1, C21: W[2,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11+A12)*B22 -> C11 -= 1, C12 += 1
    # A11: U[0,4]=1, A12: U[1,4]=1
    # B22: V[3,4]=1
    # C11: W[0,4]=-1, C12: W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)*(B11+B12) -> C22 += 1
    # A21: U[2,5]=1, A11: U[0,5]=-1
    # B11: V[0,5]=1, B12: V[1,5]=1
    # C22: W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22) -> C11 += 1
    # A12: U[1,6]=1, A22: U[3,6]=-1
    # B21: V[2,6]=1, B22: V[3,6]=1
    # C11: W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_4x5x5():
    """
    Construct a decomposition for 4x5x5 using block structure.
    Split A (4x5) into A_top (2x5) and A_bot (2x5)
    Split B (5x5) into B_left (5x2) and B_right (5x3)
    
    C_top_left = A_top * B_left (2x2) - use Strassen-like
    C_top_right = A_top * B_right (2x3)
    C_bot_left = A_bot * B_left (2x2) - use Strassen-like  
    C_bot_right = A_bot * B_right (2x3)
    
    For 2x2x2: rank 7 (Strassen)
    For 2x3x2: rank 6 (schoolbook, or possibly less)
    For 2x2x3: rank 6 (schoolbook, or possibly less)
    
    Total: 7 + 6 + 7 + 6 = 26? That seems too good.
    
    Wait, I need to be more careful. The middle dimension is 5, not 2.
    
    Let me reconsider. For C[i,k] = sum_j A[i,j] B[j,k]:
    
    For C_top_left (2x2): C[i,k] = sum_{j=0}^{4} A_top[i,j] B_left[j,k] for i,k in {0,1}
    This is a 2x5x2 tensor. Rank is at most 10 by schoolbook.
    
    I can split the sum over j into j=0,1 and j=2,3,4:
    C[i,k] = sum_{j=0}^{1} A_top[i,j] B_left[j,k] + sum_{j=2}^{4} A_top[i,j] B_left[j,k]
    
    The first part is 2x2x2 (rank 7 via Strassen)
    The second part is 2x3x2 (rank 6 via schoolbook)
    
    So total for C_top_left: 7 + 6 = 13
    
    Similarly for C_bot_left: 13
    
    For C_top_right (2x3): C[i,k] = sum_{j=0}^{4} A_top[i,j] B_right[j,k] for i in {0,1}, k in {0,1,2}
    Split j: 2x2x3 (rank 6) + 2x3x3 (rank 9) = 15
    
    For C_bot_right: 15
    
    Total: 13 + 13 + 15 + 15 = 56
    
    That's better than 100! Let me implement this.
    """
    n, m, p = 4, 5, 5
    
    # We'll build U, V, W with total rank = 56
    # Columns 0-12: C_top_left (A_top * B_left)
    #   0-6: Strassen for j=0,1
    #   7-12: schoolbook for j=2,3,4
    # Columns 13-25: C_bot_left (A_bot * B_left)
    #   13-19: Strassen for j=0,1
    #   20-25: schoolbook for j=2,3,4
    # Columns 26-40: C_top_right (A_top * B_right)
    #   26-31: schoolbook for j=0,1 (2x2x3)
    #   32-40: schoolbook for j=2,3,4 (2x3x3)
    # Columns 41-55: C_bot_right (A_bot * B_right)
    #   41-46: schoolbook for j=0,1 (2x2x3)
    #   47-55: schoolbook for j=2,3,4 (2x3x3)
    
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 56
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(base_col, i_offset, j_offset, k_offset):
        """
        Add Strassen terms for a 2x2x2 sub-problem.
        i_offset: row offset in A (0 or 2)
        j_offset: column offset in A / row offset in B (0, 1, 2, 3, or 4)
        k_offset: column offset in B (0 or 1)
        """
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            col = base_col + r
            # U indices: (i_offset + i) * m + (j_offset + j) for i,j in {0,1}
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j][r] != 0:
                        U[(i_offset + i) * m + (j_offset + j)][col] += U_s[i*2+j][r]
            # V indices: (j_offset + j) * p + (k_offset + k) for j,k in {0,1}
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k][r] != 0:
                        V[(j_offset + j) * p + (k_offset + k)][col] += V_s[j*2+k][r]
            # W indices: (k_offset + k) * n + (i_offset + i) for k,i in {0,1}
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i][r] != 0:
                        W[(k_offset + k) * n + (i_offset + i)][col] += W_s[k*2+i][r]
    
    # Helper to add a schoolbook 2xa x b x 2 block
    def add_schoolbook_2xa_bx2(base_col, i_offset, j_start, j_count, k_offset):
        """
        Add schoolbook terms for 2 x j_count x 2 sub-problem.
        """
        for j in range(j_count):
            for k in range(2):
                col = base_col + j * 2 + k
                for i in range(2):
                    U[(i_offset + i) * m + (j_start + j)][col] = 1
                    V[(j_start + j) * p + (k_offset + k)][col] = 1
                    W[(k_offset + k) * n + (i_offset + i)][col] = 1
    
    # Helper to add a schoolbook 2xa x b x 3 block
    def add_schoolbook_2xa_bx3(base_col, i_offset, j_start, j_count, k_offset):
        """
        Add schoolbook terms for 2 x j_count x 3 sub-problem.
        """
        for j in range(j_count):
            for k in range(3):
                col = base_col + j * 3 + k
                for i in range(2):
                    U[(i_offset + i) * m + (j_start + j)][col] = 1
                    V[(j_start + j) * p + (k_offset + k)][col] = 1
                    W[(k_offset + k) * n + (i_offset + i)][col] = 1
    
    # C_top_left: A_top (rows 0-1) * B_left (cols 0-1)
    # j=0,1: Strassen (7 terms)
    add_strassen_2x2x2(0, 0, 0, 0)
    # j=2,3,4: schoolbook 2x3x2 (6 terms)
    add_schoolbook_2xa_bx2(7, 0, 2, 3, 0)
    
    # C_bot_left: A_bot (rows 2-3) * B_left (cols 0-1)
    # j=0,1: Strassen (7 terms)
    add_strassen_2x2x2(13, 2, 0, 0)
    # j=2,3,4: schoolbook 2x3x2 (6 terms)
    add_schoolbook_2xa_bx2(20, 2, 2, 3, 0)
    
    # C_top_right: A_top (rows 0-1) * B_right (cols 2-4)
    # j=0,1: schoolbook 2x2x3 (6 terms)
    add_schoolbook_2xa_bx3(26, 0, 0, 2, 2)
    # j=2,3,4: schoolbook 2x3x3 (9 terms)
    add_schoolbook_2xa_bx3(32, 0, 2, 3, 2)
    
    # C_bot_right: A_bot (rows 2-3) * B_right (cols 2-4)
    # j=0,1: schoolbook 2x2x3 (6 terms)
    add_schoolbook_2xa_bx3(41, 2, 0, 2, 2)
    # j=2,3,4: schoolbook 2x3x3 (9 terms)
    add_schoolbook_2xa_bx3(47, 2, 2, 3, 2)
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = 4, 5, 5
    
    # Try the structured decomposition
    result = _construct_4x5x5()
    
    # Verify with numpy
    if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
        # Verify with exact arithmetic
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
