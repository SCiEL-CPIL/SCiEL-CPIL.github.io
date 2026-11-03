"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np

def strassen_factors():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U: (4,7), V: (4,7), W: (4,7)
    T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A * B where A is 2x2, B is 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For tensor: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # where i in {0,1}, j in {0,1}, k in {0,1}
    # We need T[i,j,k] = 1 for all (i,j,k)
    
    # The 7 rank-1 tensors correspond to the 7 Strassen products
    # Each product m_c involves a linear combination of A entries and B entries
    # A[i,j] * B[j,k] contributes to C[i,k]
    
    # Let's define:
    # U[i,j,c] = coefficient of A[i,j] in product c
    # V[j,k,c] = coefficient of B[j,k] in product c  
    # W[k,i,c] = coefficient of C[i,k] in product c (with sign)
    
    # M1 = (A00+A11)*(B00+B11):
    #   A part: A00 coeff 1, A11 coeff 1, others 0
    #   B part: B00 coeff 1, B11 coeff 1, others 0
    #   C part: C00 gets +1, C11 gets +1
    #   So W[k,i,0]: C00 -> k=0,i=0 => W[0,0,0]=1; C11 -> k=1,i=1 => W[1,1,0]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1: (A00+A11)*(B00+B11)
    # A00: i=0,j=0 => U[0,0]=1; A11: i=1,j=1 => U[3,0]=1
    # B00: j=0,k=0 => V[0,0]=1; B11: j=1,k=1 => V[3,0]=1
    # C00: k=0,i=0 => W[0,0]=1; C11: k=1,i=1 => W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)*B00
    # A10: i=1,j=0 => U[2,1]=1; A11: i=1,j=1 => U[3,1]=1
    # B00: j=0,k=0 => V[0,1]=1
    # C10: k=0,i=1 => W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A00*(B01-B11)
    # A00: i=0,j=0 => U[0,2]=1
    # B01: j=0,k=1 => V[1,2]=1; B11: j=1,k=1 => V[3,2]=-1
    # C01: k=1,i=0 => W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11*(B10-B00)
    # A11: i=1,j=1 => U[3,3]=1
    # B10: j=1,k=0 => V[2,3]=1; B00: j=0,k=0 => V[0,3]=-1
    # C00: k=0,i=0 => W[0,3]=1; C10: k=0,i=1 => W[2,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)*B11
    # A00: i=0,j=0 => U[0,4]=1; A01: i=0,j=1 => U[1,4]=1
    # B11: j=1,k=1 => V[3,4]=1
    # C00: k=0,i=0 => W[0,4]=-1; C01: k=1,i=0 => W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)*(B00+B01)
    # A10: i=1,j=0 => U[2,5]=1; A00: i=0,j=0 => U[0,5]=-1
    # B00: j=0,k=0 => V[0,5]=1; B01: j=0,k=1 => V[1,5]=1
    # C11: k=1,i=1 => W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)*(B10+B11)
    # A01: i=0,j=1 => U[1,6]=1; A11: i=1,j=1 => U[3,6]=-1
    # B10: j=1,k=0 => V[2,6]=1; B11: j=1,k=1 => V[3,6]=1
    # C00: k=0,i=0 => W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def schoolbook_factors_2x2x2():
    """Return (U, V, W) for 2x2x2 schoolbook decomposition, rank 8."""
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1
                V[j*2+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def build_block_decomposition(n, m, p):
    """
    Build a block decomposition for n x m x p matrix multiplication.
    
    Strategy: partition into blocks and use Strassen where possible.
    
    For 3x5x6:
    - We can partition the 3 rows into blocks of size 2 and 1
    - The 5 cols into blocks of size 2, 2, 1
    - The 6 rows of B into blocks of size 2, 2, 2
    
    Let's think about this more carefully.
    
    The tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    We want to decompose this as sum of rank-1 tensors.
    
    Block approach: partition i into blocks, j into blocks, k into blocks.
    For each combination of (i_block, j_block, k_block), we need to compute
    the sub-tensor T[i_block, j_block, k_block].
    
    If all three blocks have size 2, we can use Strassen (rank 7) instead of schoolbook (rank 8).
    
    For 3x5x6:
    - i blocks: [2, 1] (sizes)
    - j blocks: [2, 2, 1] (sizes)  
    - k blocks: [2, 2, 2] (sizes)
    
    Number of block combinations: 2 * 3 * 3 = 18 blocks
    
    For each block combination (ib, jb, kb):
    - If size_i == 2, size_j == 2, size_k == 2: use Strassen, rank 7
    - Otherwise: use schoolbook, rank = size_i * size_j * size_k
    
    Let's compute:
    Block sizes for i: [2, 1]
    Block sizes for j: [2, 2, 1]
    Block sizes for k: [2, 2, 2]
    
    Combinations:
    (2,2,2): 2*2*2=8 schoolbook, 7 strassen -> 1 such block (ib=0,jb=0,kb=0)
              Wait, let me recount.
    
    ib=0 (size 2), jb=0 (size 2), kb=0 (size 2): Strassen, rank 7
    ib=0 (size 2), jb=0 (size 2), kb=1 (size 2): Strassen, rank 7
    ib=0 (size 2), jb=0 (size 2), kb=2 (size 2): Strassen, rank 7
    ib=0 (size 2), jb=1 (size 2), kb=0 (size 2): Strassen, rank 7
    ib=0 (size 2), jb=1 (size 2), kb=1 (size 2): Strassen, rank 7
    ib=0 (size 2), jb=1 (size 2), kb=2 (size 2): Strassen, rank 7
    ib=0 (size 2), jb=2 (size 1), kb=0 (size 2): schoolbook 2*1*2=4
    ib=0 (size 2), jb=2 (size 1), kb=1 (size 2): schoolbook 2*1*2=4
    ib=0 (size 2), jb=2 (size 1), kb=2 (size 2): schoolbook 2*1*2=4
    ib=1 (size 1), jb=0 (size 2), kb=0 (size 2): schoolbook 1*2*2=4
    ib=1 (size 1), jb=0 (size 2), kb=1 (size 2): schoolbook 1*2*2=4
    ib=1 (size 1), jb=0 (size 2), kb=2 (size 2): schoolbook 1*2*2=4
    ib=1 (size 1), jb=1 (size 2), kb=0 (size 2): schoolbook 1*2*2=4
    ib=1 (size 1), jb=1 (size 2), kb=1 (size 2): schoolbook 1*2*2=4
    ib=1 (size 1), jb=1 (size 2), kb=2 (size 2): schoolbook 1*2*2=4
    ib=1 (size 1), jb=2 (size 1), kb=0 (size 2): schoolbook 1*1*2=2
    ib=1 (size 1), jb=2 (size 1), kb=1 (size 2): schoolbook 1*1*2=2
    ib=1 (size 1), jb=2 (size 1), kb=2 (size 2): schoolbook 1*1*2=2
    
    Total rank = 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    
    That matches the expected score of 84!
    """
    n, m, p = 3, 5, 6
    
    # Define block partitions
    # i: [0,1] size 2, [2] size 1
    i_blocks = [(0, 2), (2, 3)]  # (start, end)
    # j: [0,1] size 2, [2,3] size 2, [4] size 1
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    # k: [0,1] size 2, [2,3] size 2, [4,5] size 2
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    U = []
    V = []
    W = []
    
    for ib, (i_start, i_end) in enumerate(i_blocks):
        size_i = i_end - i_start
        for jb, (j_start, j_end) in enumerate(j_blocks):
            size_j = j_end - j_start
            for kb, (k_start, k_end) in enumerate(k_blocks):
                size_k = k_end - k_start
                
                if size_i == 2 and size_j == 2 and size_k == 2:
                    # Use Strassen
                    sU, sV, sW = strassen_factors()
                    for c in range(7):
                        u_col = [0] * 0  # placeholder
                        # We'll build columns separately
                        pass
                    
                    for c in range(7):
                        # U column: for each (i,j) in this block, U[i*m+j, c]
                        u_vals = [0] * (n * m)
                        for di in range(2):
                            for dj in range(2):
                                u_vals[(i_start + di) * m + (j_start + dj)] = sU[di*2+dj][c]
                        
                        v_vals = [0] * (m * p)
                        for dj in range(2):
                            for dk in range(2):
                                v_vals[(j_start + dj) * p + (k_start + dk)] = sV[dj*2+dk][c]
                        
                        w_vals = [0] * (p * n)
                        for dk in range(2):
                            for di in range(2):
                                w_vals[(k_start + dk) * n + (i_start + di)] = sW[dk*2+di][c]
                        
                        U.append(u_vals)
                        V.append(v_vals)
                        W.append(w_vals)
                else:
                    # Schoolbook
                    for di in range(size_i):
                        for dj in range(size_j):
                            for dk in range(size_k):
                                u_vals = [0] * (n * m)
                                u_vals[(i_start + di) * m + (j_start + dj)] = 1
                                
                                v_vals = [0] * (m * p)
                                v_vals[(j_start + dj) * p + (k_start + dk)] = 1
                                
                                w_vals = [0] * (p * n)
                                w_vals[(k_start + dk) * n + (i_start + di)] = 1
                                
                                U.append(u_vals)
                                V.append(v_vals)
                                W.append(w_vals)
    
    # Transpose: U should be (n*m, rank), currently list of columns
    rank = len(U)
    U_t = [[U[c][i] for c in range(rank)] for i in range(n * m)]
    V_t = [[V[c][i] for c in range(rank)] for i in range(m * p)]
    W_t = [[W[c][i] for c in range(rank)] for i in range(p * n)]
    
    return U_t, V_t, W_t

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build block decomposition with Strassen
    U, V, W = build_block_decomposition(n, m, p)
    
    rank = len(U[0])
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
