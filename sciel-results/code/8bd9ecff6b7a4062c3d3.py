"""Tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_3x3():
    """
    Build a 3x3 Strassen-like decomposition.
    Split 3x3 into blocks and use 7 products.
    Returns (U, V, W) with shapes (9, rank), (9, rank), (9, rank)
    """
    # For 3x3, we can use a block decomposition where we split into:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # where A11, A22, A33 are 1x1, and off-diagonals are 1x1 or 1x2/2x1
    
    # Actually, let's use a simpler approach: decompose 3x3 as:
    # Use the fact that 3x3 multiplication can be done with fewer than 27 products.
    # Known lower bound is 19, best known is 23.
    # For simplicity, let's use a block decomposition:
    # Split into 2x2 block + 1x1 block
    # A = [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]
    # B = [[B11(2x2), B12(2x1)], [B21(1x2), B22(1x1)]]
    # C = [[C11(2x2), C12(2x1)], [C21(1x2), C22(1x1)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Use Strassen for the 2x2 parts:
    U2, V2, W2 = _build_strassen_2x2()
    
    # For C11: A11(2x2) * B11(2x2) -> rank 7
    # For C11: A12(2x1) * B21(1x2) -> rank 2 (outer product)
    # For C12: A11(2x2) * B12(2x1) -> rank 7
    # For C12: A12(2x1) * B22(1x1) -> rank 2
    # For C21: A21(1x2) * B11(2x2) -> rank 7
    # For C21: A22(1x1) * B21(1x2) -> rank 2
    # For C22: A21(1x2) * B12(2x1) -> rank 2
    # For C22: A22(1x1) * B22(1x1) -> rank 1
    
    # Total: 7+2+7+2+7+2+2+1 = 30
    # But we can do better by combining.
    
    # Actually, let's just use schoolbook for 3x3 for now: rank 27
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W

def _build_2x2x1():
    """
    Build 2x2x1 decomposition: A(2x2) * B(2x1) -> C(2x1)
    c[0] = a00*b0 + a01*b1
    c[1] = a10*b0 + a11*b1
    Rank 4 (schoolbook) or can we do better?
    Actually this is just 2 dot products of length 2, so rank 4 is minimal.
    """
    rank = 4
    U = [[0]*rank for _ in range(4)]  # 2*2
    V = [[0]*rank for _ in range(2)]  # 2*1
    W = [[0]*rank for _ in range(2)]  # 1*2
    
    # c0 = a00*b0 + a01*b1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    
    # c1 = a10*b0 + a11*b1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _build_1x2x2():
    """
    Build 1x2x2 decomposition: A(1x2) * B(2x2) -> C(1x2)
    c[0] = a0*b00 + a1*b10
    c[1] = a0*b01 + a1*b11
    Rank 4.
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]  # 1*2
    V = [[0]*rank for _ in range(4)]  # 2*2
    W = [[0]*rank for _ in range(2)]  # 2*1
    
    # c0 = a0*b00 + a1*b10
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
    
    # c1 = a0*b01 + a1*b11
    U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    
    return U, V, W

def _build_2x1x2():
    """
    Build 2x1x2 decomposition: A(2x1) * B(1x2) -> C(2x2)
    c[i,k] = a[i] * b[k]
    This is an outer product, rank 2.
    """
    rank = 2
    U = [[1, 0], [0, 1]]  # 2*1
    V = [[1, 1]]          # 1*2
    W = [[1, 0], [0, 1]]  # 2*2: W[k*2+i, r]
    # W[0*2+0, 0] = 1, W[0*2+1, 1] = 1? No...
    # c[0,0] = a[0]*b[0] -> U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # c[0,1] = a[0]*b[1] -> U[0,1]*V[1,1]*W[1,0]... 
    
    # Let me redo this carefully.
    # T[i,j,k,l] where A is 2x1, B is 1x2, C is 2x2
    # j is in {0} (since m=1), l is in {0} (since m=1)
    # T[i,0,k,0] = 1 for all i,k
    # U shape: (2*1, rank) = (2, rank)
    # V shape: (1*2, rank) = (2, rank)
    # W shape: (2*2, rank) = (4, rank)
    
    # We want: sum_r U[i, r] * V[k, r] * W[k*2+i, r] = 1 for all i,k
    
    # With rank 2:
    # r=0: U[0,0]=1, U[1,0]=0; V[0,0]=1, V[1,0]=0; W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0
    #   This gives c[0,0] += 1, others 0
    # r=1: U[0,1]=0, U[1,1]=1; V[0,1]=0, V[1,1]=1; W[0,1]=0, W[1,1]=0, W[2,1]=0, W[3,1]=1
    #   This gives c[1,1] += 1, others 0
    
    # That only covers 2 of 4 entries. We need rank 4 actually? No...
    # Let me think again.
    # c[i,k] = a[i] * b[k]
    # This is rank 1 as a bilinear map! We can do it with rank 1.
    # U[i, 0] = 1 for all i
    # V[k, 0] = 1 for all k
    # W[k*2+i, 0] = 1 for all k,i
    
    rank = 1
    U = [[1], [1]]
    V = [[1], [1]]
    W = [[1], [1], [1], [1]]
    
    return U, V, W

def _build_1x1x1():
    """
    Build 1x1x1 decomposition: rank 1.
    """
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return U, V, W

def _build_2x2x2():
    """
    Build 2x2x2 decomposition using Strassen: rank 7.
    """
    return _build_strassen_2x2()

def _build_2x1x1():
    """
    Build 2x1x1 decomposition: rank 2.
    A(2x1) * B(1x1) -> C(2x1)
    c[0] = a[0]*b[0], c[1] = a[1]*b[0]
    """
    rank = 2
    U = [[1, 0], [0, 1]]
    V = [[1, 1]]
    W = [[1, 0], [0, 1]]
    return U, V, W

def _build_1x1x2():
    """
    Build 1x1x2 decomposition: rank 2.
    A(1x1) * B(1x2) -> C(1x2)
    c[0] = a[0]*b[0], c[1] = a[0]*b[1]
    """
    rank = 2
    U = [[1, 1]]
    V = [[1, 0], [0, 1]]
    W = [[1, 0], [0, 1]]
    return U, V, W

def _build_1x2x1():
    """
    Build 1x2x1 decomposition: rank 2.
    A(1x2) * B(2x1) -> C(1x1)
    c[0] = a[0]*b[0] + a[1]*b[1]
    """
    rank = 2
    U = [[1, 1]]
    V = [[1, 0], [0, 1]]
    W = [[1, 1]]
    return U, V, W

def _build_2x2x1_strassen():
    """
    Build 2x2x1 using Strassen-like approach.
    A(2x2) * B(2x1) -> C(2x1)
    
    Actually, this is just two dot products of length 2.
    c[0] = a00*b0 + a01*b1
    c[1] = a10*b0 + a11*b1
    
    Minimum rank is 4 (you need 4 scalar multiplications).
    """
    return _build_2x2x1()

def _build_1x2x2_strassen():
    """
    Build 1x2x2.
    A(1x2) * B(2x2) -> C(1x2)
    c[0] = a0*b00 + a1*b10
    c[1] = a0*b01 + a1*b11
    Minimum rank is 4.
    """
    return _build_1x2x2()

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    Each decomp is (U, V, W, i_range, j_range, k_range) where ranges are (start, end) exclusive.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_4x5_5x6_optimized():
    """
    Build 4x5 * 5x6 decomposition using block structure.
    
    Strategy: Split the 5 middle dimension into 2+3 or 3+2.
    A(4x5) = [A_left(4x2) | A_right(4x3)]
    B(5x6) = [B_top(2x6); B_bottom(3x6)]
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left(4x2) * B_top(2x6):
      Split 6 into 3+3:
      A_left * [B_top_left | B_top_right]
      = [A_left*B_top_left | A_left*B_top_right]
      Each is 4x2 * 2x3 -> 4x3
      
    For A_right(4x3) * B_bottom(3x6):
      Split 6 into 3+3:
      = [A_right*B_bottom_left | A_right*B_bottom_right]
      Each is 4x3 * 3x3 -> 4x3
    
    So we need:
    - 4x2 * 2x3 (twice)
    - 4x3 * 3x3 (twice)
    
    For 4x2 * 2x3:
      Split 4 into 2+2:
      [A11(2x2); A21(2x2)] * [B11(2x3)]
      = [A11*B11; A21*B11]
      Each is 2x2 * 2x3 -> 2x3
      Use Strassen for 2x2 part: rank 7 each, total 14
    
    For 4x3 * 3x3:
      Split 4 into 2+2:
      [A11(2x3); A21(2x3)] * B(3x3)
      = [A11*B; A21*B]
      Each is 2x3 * 3x3 -> 2x3
      Split 3 into 2+1:
      A11 = [A11_left(2x2) | A11_right(2x1)]
      B = [B_top(2x3); B_bottom(1x3)]
      A11*B = A11_left*B_top + A11_right*B_bottom
      A11_left*B_top: 2x2 * 2x3 -> use Strassen, rank 7
      A11_right*B_bottom: 2x1 * 1x3 -> rank 3 (outer product style)
      Total for 2x3 * 3x3: 7 + 3 = 10
      Total for 4x3 * 3x3: 20
    
    Total rank: 14 + 14 + 20 + 20 = 68
    
    But wait, let me reconsider. Let me use a more direct approach.
    
    Actually, let me try a different split.
    Split the 5 into 2+3:
    A(4x5) = [A1(4x2) | A2(4x3)]
    B(5x6) = [B1(2x6); B2(3x6)]
    C = A1*B1 + A2*B2
    
    A1*B1: 4x2 * 2x6
    A2*B2: 4x3 * 3x6
    
    For 4x2 * 2x6:
      Split 4 into 2+2, 6 into 3+3:
      [A11(2x2); A21(2x2)] * [B11(2x3) | B12(2x3)]
      = [A11*B11 + A21*B11 | A11*B12 + A21*B12]  -- no, that's wrong
      
      Actually:
      C = A*B where A is 4x2, B is 2x6
      C[i,k] = sum_j A[i,j]*B[j,k]
      
      Split rows of A into 2+2: A = [A_top(2x2); A_bottom(2x2)]
      Split cols of C (and B) into 3+3: B = [B_left(2x3) | B_right(2x3)]
      
      C = [A_top*B_left + A_bottom*B_left | A_top*B_right + A_bottom*B_right]
      
      Wait no: C_top(2x6) = A_top(2x2) * B(2x6)
      C_bottom(2x6) = A_bottom(2x2) * B(2x6)
      
      So:
      C_top = A_top * B = [A_top*B_left | A_top*B_right]
      C_bottom = A_bottom * B = [A_bottom*B_left | A_bottom*B_right]
      
      Each 2x2 * 2x3: use Strassen on the 2x2 part.
      2x2 * 2x3: rank 7 (Strassen)
      
      Total for 4x2 * 2x6: 4 * 7 = 28
    
    For 4x3 * 3x6:
      Split rows of A into 2+2: A = [A_top(2x3); A_bottom(2x3)]
      Split cols of C (and B) into 3+3: B = [B_left(3x3) | B_right(3x3)]
      
      C_top = A_top * B = [A_top*B_left | A_top*B_right]
      C_bottom = A_bottom * B = [A_bottom*B_left | A_bottom*B_right]
      
      Each 2x3 * 3x3:
        Split 3 into 2+1:
        A_top = [A_top_left(2x2) | A_top_right(2x1)]
        B_left = [B_left_top(2x3); B_left_bottom(1x3)]
        A_top*B_left = A_top_left*B_left_top + A_top_right*B_left_bottom
        2x2 * 2x3: rank 7
        2x1 * 1x3: rank 3 (it's an outer product: c[i,k] = a[i]*b[k])
        Total: 10
      
      Total for 4x3 * 3x6: 4 * 10 = 40
    
    Total: 28 + 40 = 68
    
    Let me implement this.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # We'll build each sub-decomposition and combine.
    decomps = []
    
    # 1. A_top(2x2) * B_left(2x3) -> C_top_left(2x3)
    #    i in [0,2), j in [0,2), k in [0,3)
    #    Use Strassen 2x2 * 2x2 for the first 2 cols, then 2x2 * 2x1 for the last col.
    #    Actually, 2x2 * 2x3 can be done as:
    #    C[:, 0:2] = A * B[:, 0:2]  (2x2 * 2x2, Strassen rank 7)
    #    C[:, 2] = A * B[:, 2]      (2x2 * 2x1, rank 4)
    #    Total: 11
    
    # Let me just use schoolbook for the small ones to be safe, and Strassen where it helps.
    
    # 2x2 * 2x3:
    #   2x2 * 2x2 (Strassen, rank 7) + 2x2 * 2x1 (schoolbook, rank 4) = 11
    #   Or: 2x2 * 2x3 directly with some optimization.
    #   Let's use: split B into 2+1 cols.
    
    def _build_2x2x2x3():
        """2x2 * 2x3 -> 2x3, rank 11 (Strassen for 2x2*2x2 + schoolbook for 2x2*2x1)"""
        U_s, V_s, W_s = _build_strassen_2x2()  # 2x2*2x2, rank 7
        # 2x2 * 2x1:
        U_1, V_1, W_1 = _build_2x2x1()  # rank 4
        
        # Combine:
        # First part: 2x2 * 2x2 for k in [0,2)
        # Second part: 2x2 * 2x1 for k in [2,3)
        
        rank_total = 7 + 4
        U = [[0]*rank_total for _ in range(4)]
        V = [[0]*rank_total for _ in range(6)]
        W = [[0]*rank_total for _ in range(6)]
        
        # Part 1: Strassen 2x2*2x2
        for r in range(4):
            for c in range(7):
                U[r][c] = U_s[r][c]
                V[r][c] = V_s[r][c]
                W[r][c] = W_s[r][c]
        
        # Part 2: 2x2*2x1, shifted by 7
        # V for 2x2*2x1 has shape (2*1, 4) = (2, 4), but we need (2*3, 4) = (6, 4)
        # The B matrix is 2x1, so V rows are indexed by j*1+k where j in [0,2), k in [0,1)
        # In the full 2x3 context, j in [0,2), k in [0,3)
        # For the 2x1 part, k is fixed at 2 (the third column)
        # So V[j*3+2, c+7] = V_1[j*1+0, c]
        
        for j in range(2):
            for c in range(4):
                V[j*3 + 2][7 + c] = V_1[j][c]
        
        # U is the same
        for r in range(4):
            for c in range(4):
                U[r][7 + c] = U_1[r][c]
        
        # W for 2x2*2x1 has shape (1*2, 4) = (2, 4), indexed by k*2+i where k in [0,1), i in [0,2)
        # In the full 2x3 context, W is (3*2, rank) = (6, rank), indexed by k*2+i
        # For the 2x1 part, k=2, so W[2*2+i, 7+c] = W_1[0*2+i, c] = W_1[i, c]
        
        for i in range(2):
            for c in range(4):
                W[2*2 + i][7 + c] = W_1[i][c]
        
        return U, V, W
    
    def _build_2x3x3x3():
        """2x3 * 3x3 -> 2x3, rank 10 (Strassen 2x2*2x3 + schoolbook 2x1*1x3)"""
        # A(2x3) = [A_left(2x2) | A_right(2x1)]
        # B(3x3) = [B_top(2x3); B_bottom(1x3)]
        # C = A_left * B_top + A_right * B_bottom
        # A_left * B_top: 2x2 * 2x3, rank 11
        # A_right * B_bottom: 2x1 * 1x3, rank 3
        
        U_ab, V_ab, W_ab = _build_2x2x2x3()  # 2x2 * 2x3, rank 11
        # 2x1 * 1x3:
        # c[i,k] = a[i] * b[k]
        # Rank 3 (or even 1 if we use outer product trick)
        # Actually rank 1: U[i,0]=1, V[0,0]=1, W[k*2+i,0]=1
        # But V has shape (1*3, rank) = (3, rank), and the single element is at j=0,k=0..2
        # V[k, 0] = 1 for k in [0,3)
        
        rank_bc = 3
        U_bc = [[0]*rank_bc for _ in range(2)]  # 2*1
        V_bc = [[0]*rank_bc for _ in range(3)]  # 1*3
        W_bc = [[0]*rank_bc for _ in range(6)]  # 3*2
        
        for i in range(2):
            U_bc[i][0] = 1
        for k in range(3):
            V_bc[k][0] = 1
        for i in range(2):
            for k in range(3):
                W_bc[k*2 + i][0] = 1
        
        # Combine:
        rank_total = 11 + 3
        U = [[0]*rank_total for _ in range(6)]  # 2*3
        V = [[0]*rank_total for _ in range(9)]  # 3*3
        W = [[0]*rank_total for _ in range(6)]  # 3*2
        
        # Part 1: 2x2 * 2x3 (A_left * B_top)
        # A_left is at positions (i, j) where i in [0,2), j in [0,2)
        # B_top is at positions (j, k) where j in [0,2), k in [0,3)
        # C is at positions (i, k) where i in [0,2), k in [0,3)
        
        for i in range(2):
            for j in range(2):
                for c in range(11):
                    U[i*3 + j][c] = U_ab[i*2 + j][c]
        
        for j in range(2):
            for k in range(3):
                for c in range(11):
                    V[j*3 + k][c] = V_ab[j*2 + k][c]
        
        for i in range(2):
            for k in range(3):
                for c in range(11):
                    W[k*2 + i][c] = W_ab[k*2 + i][c]
        
        # Part 2: 2x1 * 1x3 (A_right * B_bottom)
        # A_right is at positions (i, j=2) where i in [0,2)
        # B_bottom is at positions (j=2, k) where k in [0,3)
        # C is at positions (i, k) where i in [0,2), k in [0,3)
        
        for i in range(2):
            for c in range(3):
                U[i*3 + 2][11 + c] = U_bc[i][c]
        
        for k in range(3):
            for c in range(3):
                V[2*3 + k][11 + c] = V_bc[k][c]
        
        for i in range(2):
            for k in range(3):
                for c in range(3):
                    W[k*2 + i][11 + c] += W_bc[k*2 + i][c]
        
        return U, V, W
    
    # Now build the full 4x5 * 5x6:
    # A(4x5) = [A1(4x2) | A2(4x3)]
    # B(5x6) = [B1(2x6); B2(3x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x2 * 2x6
    #   Split 4 into 2+2:
    #   A1 = [A1_top(2x2); A1_bottom(2x2)]
    #   Split 6 into 3+3:
    #   B1 = [B1_left(2x3) | B1_right(2x3)]
    #   C1_top = A1_top * B1 = [A1_top*B1_left | A1_top*B1_right]  (2x2 * 2x3, rank 11 each)
    #   C1_bottom = A1_bottom * B1 = [A1_bottom*B1_left | A1_bottom*B1_right]  (2x2 * 2x3, rank 11 each)
    #   Total: 4 * 11 = 44
    
    # A2*B2: 4x3 * 3x6
    #   Split 4 into 2+2:
    #   A2 = [A2_top(2x3); A2_bottom(2x3)]
    #   Split 6 into 3+3:
    #   B2 = [B2_left(3x3) | B2_right(3x3)]
    #   C2_top = A2_top * B2 = [A2_top*B2_left | A2_top*B2_right]  (2x3 * 3x3, rank 10 each)
    #   C2_bottom = A2_bottom * B2 = [A2_bottom*B2_left | A2_bottom*B2_right]  (2x3 * 3x3, rank 10 each)
    #   Total: 4 * 10 = 40
    
    # Grand total: 44 + 40 = 84
    
    # Hmm, that's worse than the original 78. Let me try a different approach.
    
    # Let me try: 4x5 * 5x6
    # Split 5 into 2+3:
    # A = [A1(4x2) | A2(4x3)]
    # B = [B1(2x6); B2(3x6)]
    
    # For A1*B1 (4x2 * 2x6):
    #   Use Strassen on the 2x2 blocks of A1:
    #   A1 = [A11(2x2); A12(2x2)]
    #   B1 is 2x6
    #   C1 = A1*B1
    #   C1_top = A11*B1 (2x2 * 2x6)
    #   C1_bottom = A12*B1 (2x2 * 2x6)
    #   2x2 * 2x6: split 6 into 3+3
    #     2x2 * 2x3 (rank 11) + 2x2 * 2x3 (rank 11) = 22
    #   Total for 4x2 * 2x6: 44
    
    # For A2*B2 (4x3 * 3x6):
    #   A2 = [A21(2x3); A22(2x3)]
    #   C2_top = A21*B2 (2x3 * 3x6)
    #   C2_bottom = A22*B2 (2x3 * 3x6)
    #   2x3 * 3x6: split 6 into 3+3
    #     2x3 * 3x3 (rank 10) + 2x3 * 3x3 (rank 10) = 20
    #   Total for 4x3 * 3x6: 40
    
    # Total: 84
    
    # Let me try splitting 5 into 1+4:
    # A = [A1(4x1) | A2(4x4)]
    # B = [B1(1x6); B2(4x6)]
    # A1*B1: 4x1 * 1x6 -> rank 6 (outer product, or even 1 with the trick)
    #   Actually rank 1: U[i,0]=1, V[k,0]=1, W[k*4+i,0]=1
    # A2*B2: 4x4 * 4x6
    #   Split 4x4 * 4x6:
    #   A2 = 4x4, B2 = 4x6
    #   Use Strassen on 4x4:
    #   4x4 * 4x6:
    #     Split 4 into 2+2, 6 into 3+3
    #     [A11(2x2) A12(2x2)] * [B11(2x3) B12(2x3)]
    #     [A21(2x2) A22(2x2)]   [B21(2x3) B22(2x3)]
    #     C11 = A11*B11 + A12*B21 (2x2*2x3 + 2x2*2x3 = 11+11 = 22)
    #     C12 = A11*B12 + A12*B22 (11+11 = 22)
    #     C21 = A21*B11 + A22*B21 (22)
    #     C22 = A21*B12 + A22*B22 (22)
    #     Total: 88
    #   But with Strassen on the 2x2 level:
    #     Use 7 Strassen products for the 2x2 blocks:
    #     Each Strassen product involves 2x2 * 2x2 = rank 7
    #     But we have 2x2 * 2x3, not 2x2 * 2x2.
    #     
    #     Actually, for 4x4 * 4x6, we can use:
    #     Strassen on the 4x4 part: 7 products of 2x2 * 2x2 (rank 7 each) = 49
    #     But the B matrix is 4x6, not 4x4.
    #     
    #     Let me think differently. 4x4 * 4x6:
    #     = 4x4 * [4x4 | 4x2]
    #     = [4x4*4x4 | 4x4*4x2]
    #     4x4*4x4: Strassen, rank 49
    #     4x4*4x2: 
    #       Split 4 into 2+2, 2 into 1+1:
    #       [A11 A12; A21 A22] * [B1(4x1) | B2(4x1)]
    #       C1 = A*B1, C2 = A*B2
    #       Each 4x4 * 4x1:
    #         Split 4 into 2+2:
    #         [A11(2x2); A21(2x2)] * [B1_top(2x1); B1_bottom(2x1)]
    #         C1_top = A11*B1_top (2x2*2x1, rank 4)
    #         C1_bottom = A21*B1_top + A11*B1_bottom + A21*B1_bottom... no
    #         C1_top = A11*B1_top (2x2 * 2x1)
    #         C1_bottom = A21*B1_top (2x2 * 2x1) -- wait, that's not right either.
    #         
    #         Actually:
    #         A(4x4) * B(4x1) -> C(4x1)
    #         A = [A11(2x2) A12(2x2); A21(2x2) A22(2x2)]
    #         B = [B11(2x1); B12(2x1)]
    #         C = [C11(2x1); C21(2x1)]
    #         C11 = A11*B11 + A12*B12
    #         C21 = A21*B11 + A22*B12
    #         Each 2x2*2x1: rank 4
    #         Total: 4*4 = 16
    #     
    #     So 4x4*4x2: 2 * 16 = 32
    #     Total 4x4*4x6: 49 + 32 = 81
    #     
    #   Total: 1 + 81 = 82
    
    # Hmm, still not better than 78.
    
    # Let me try the original approach but with better sub-decompositions.
    # Original: 4x4*4x4 (Strassen, 49) + 4x1*1x6 (rank 1) + 4x4*4x2 (28) = 78
    # The 4x4*4x2 was 28. Can we do better?
    # 4x4 * 4x2:
    #   Split 4 into 2+2, 2 into 1+1:
    #   A = [A11(2x2) A12(2x2); A21(2x2) A22(2x2)]
    #   B = [B1(4x1) | B2(4x1)]
    #   C = [C1(4x1) | C2(4x1)]
    #   C1 = A*B1, C2 = A*B2
    #   Each 4x4*4x1: rank 16 (as computed above)
    #   Total: 32
    #   But original was 28 using a different method.
    
    # Let me check: the original used Strassen on 2x2 blocks for 4x4x2.
    # 4x4 * 4x2:
    #   Split into 2x2 blocks:
    #   A = [A11(2x2) A12(2x2); A21(2x2) A22(2x2)]
    #   B = [B11(2x1) B12(2x1); B21(2x1) B22(2x1)]  -- wait, B is 4x2, so:
    #   B = [B11(2x2); B21(2x2)] where B11 is rows 0-1, B21 is rows 2-3
    #   No, B is 4x2, so B = [B_top(2x2); B_bottom(2x2)]
    #   C = A*B
    #   C_top = A_top*B + A_bottom*B... no
    #   C[i,k] = sum_j A[i,j]*B[j,k]
    #   C_top(2x2) = A_top(2x4) * B(4x2)... 
    #   
    #   Actually, the original code had:
    #   products = [
    #       (0, 0, 0, 0, 1),   # A11 * B11 -> C11
    #       (0, 2, 2, 0, 1),   # A12 * B21 -> C11
    #       (2, 0, 0, 2, 1),   # A21 * B11 -> C21
    #       (2, 2, 2, 2, 1),   # A22 * B21 -> C21
    #   ]
    #   Each is 2x2 * 2x2 -> 2x2, Strassen rank 7
    #   Total: 4*7 = 28
    
    # That seems correct. Let me keep the original approach.
    
    # Actually, let me try a completely different approach.
    # Use the fact that we can combine some of the terms.
    
    # For now, let's just use the original decomposition which gives rank 78.
    # But let me try to improve the 4x4*4x2 part.
    
    # 4x4 * 4x2:
    #   We can use Strassen on the 4x4 part:
    #   A(4x4) * B(4x2)
    #   = A * [B1(4x1) | B2(4x1)]
    #   = [A*B1 | A*B2]
    #   Each A*B_i is 4x4 * 4x1.
    #   
    #   4x4 * 4x1:
    #     Use Strassen: 7 products of 2x2 * 2x1 (rank 4 each) = 28
    #     Or: split 4 into 2+2:
    #     A = [A11(2x2) A12(2x2); A21(2x2) A22(2x2)]
    #     B = [B11(2x1); B12(2x1)]
    #     C11 = A11*B11 + A12*B12 (2x2*2x1 + 2x2*2x1 = 4+4 = 8)
    #     C21 = A21*B11 + A22*B12 (8)
    #     Total: 16
    #   
    #   So 4x4*4x2: 2*16 = 32
    #   But the block method gives 28. So the block method is better.
    
    # Let me try yet another approach for the full 4x5*5x6.
    # Split 5 into 3+2:
    # A = [A1(4x3) | A2(4x2)]
    # B = [B1(3x6); B2(2x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x3 * 3x6
    # A2*B2: 4x2 * 2x6
    
    # 4x3 * 3x6:
    #   Split 4 into 2+2, 6 into 3+3:
    #   A1 = [A11(2x3); A12(2x3)]
    #   B1 = [B11(3x3) | B12(3x3)]
    #   C11 = A11*B11 (2x3*3x3, rank 10)
    #   C12 = A11*B12 (10)
    #   C21 = A12*B11 (10)
    #   C22 = A12*B12 (10)
    #   Total: 40
    
    # 4x2 * 2x6:
    #   Split 4 into 2+2, 6 into 3+3:
    #   A2 = [A21(2x2); A22(2x2)]
    #   B2 = [B21(2x3) | B22(2x3)]
    #   C11 = A21*B21 (2x2*2x3, rank 11)
    #   C12 = A21*B22 (11)
    #   C21 = A22*B21 (11)
    #   C22 = A22*B22 (11)
    #   Total: 44
    
    # Grand total: 40 + 44 = 84. Worse.
    
    # Let me try 5 = 1+4:
    # A1*B1: 4x1 * 1x6, rank 1
    # A2*B2: 4x4 * 4x6
    #   4x4 * 4x6:
    #     = 4x4 * [4x4 | 4x2]
    #     = [4x4*4x4 | 4x4*4x2]
    #     4x4*4x4: Strassen, 49
    #     4x4*4x2: 28 (block method)
    #     Total: 77
    # Grand total: 1 + 77 = 78. Same as original.
    
    # Let me try to improve 4x4*4x4.
    # Known best for 4x4 is 49 (Strassen). Can we do better?
    # Actually, the best known for 4x4 is 49, and it's conjectured to be optimal.
    
    # Let me try a different overall strategy.
    # Use the Laderman algorithm or other known algorithms.
    
    # For now, let's just return the original decomposition.
    
    # Build the original decomposition:
    U1, V1, W1 = _build_strassen_4x4()
    decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
    
    U2, V2, W2 = _build_4x1x6_optimized()
    decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
    
    U3, V3, W3 = _build_strassen_4x4x2()
    decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
    
    return _combine_decompositions([decomp1, decomp2, decomp3], 4, 5, 6)

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_4x1x6_optimized():
    """
    Build 4x1x6 decomposition with rank 1.
    """
    rank = 1
    U = [[1] for _ in range(4)]
    V = [[1] for _ in range(6)]
    W = [[1] for _ in range(24)]
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using Strassen on 2x2 blocks.
    Rank 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    products = [
        (0, 0, 0, 0, 1),
        (0, 2, 2, 0, 1),
        (2, 0, 0, 2, 1),
        (2, 2, 2, 2, 1),
    ]
    
    for s, (a_rs, a_cs, b_rs, c_rs, c_sign) in enumerate(products):
        for t in range(7):
            col = s * 7 + t
            
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_rs + la_i
                        global_j = a_cs + la_j
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
            
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_rs + lb_j
                        global_k = lb_k
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
            
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = lc_k
                        global_i = c_rs + lc_i
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c * c_sign
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_4x5_5x6_optimized()
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
