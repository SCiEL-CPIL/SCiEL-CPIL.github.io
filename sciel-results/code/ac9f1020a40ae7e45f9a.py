"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for M[2,2,13].
    
    Strategy: Group the 13 columns of B into 6 pairs of 2 columns + 1 single column.
    - Each pair of 2 columns: 2x2 * 2x2 multiplication, use Strassen (rank 7)
    - Single column: 2x2 * 2x1 multiplication, schoolbook (rank 4)
    Total rank: 6*7 + 4 = 46
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # We'll build U, V, W matrices
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (26, rank)
    # W has shape (p*n, rank) = (26, rank)
    
    rank = 6 * 7 + 4  # 46
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_offset = 0
    
    # For each pair of columns (col_k, col_{k+1}) where k is even
    # We have a 2x2 * 2x2 subproblem for columns k and k+1
    # The subproblem: A (2x2) * B_pair (2x2) = C_pair (2x2)
    # where B_pair has columns [b_k, b_{k+1}]
    
    # Strassen for 2x2 * 2x2:
    # Let A = [[a11, a12], [a21, a22]]
    #     B = [[b11, b12], [b21, b22]]
    #     C = [[c11, c12], [c21, c22]]
    # 
    # Strassen products:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # The entries of A are indexed as:
    # A[0,0] -> index 0 in U (row 0)
    # A[0,1] -> index 1 in U (row 1)
    # A[1,0] -> index 2 in U (row 2)
    # A[1,1] -> index 3 in U (row 3)
    
    # The entries of B_pair (2x2) correspond to B[0,k], B[0,k+1], B[1,k], B[1,k+1]
    # B[0,k] -> V row 0*p + k = k
    # B[0,k+1] -> V row 0*p + (k+1) = k+1
    # B[1,k] -> V row 1*p + k = p + k = 13 + k
    # B[1,k+1] -> V row 1*p + (k+1) = 13 + k + 1
    
    # The entries of C_pair (2x2) correspond to C[0,k], C[0,k+1], C[1,k], C[1,k+1]
    # W index: k*n + i = k*2 + i for C[i,k]
    # C[0,k] -> W row k*2 + 0 = 2k
    # C[0,k+1] -> W row (k+1)*2 + 0 = 2k+2
    # C[1,k] -> W row k*2 + 1 = 2k+1
    # C[1,k+1] -> W row (k+1)*2 + 1 = 2k+3
    
    # Strassen coefficient table for 2x2*2x2:
    # Each Strassen product m_j involves specific linear combinations of A and B entries
    # and contributes to specific C entries.
    
    # m1 = (a11 + a22)(b11 + b22): 
    #   U contributions: U[0]+=1, U[3]+=1
    #   V contributions: V[0]+=1, V[2]+=1 (where V[0]=B[0,0]=b11, V[2]=B[1,1]=b22)
    #   W contributions: W[0]+=1, W[3]+=1 (c11 and c22 get +m1)
    # Wait, let me re-index. In our V, for a pair starting at column k:
    #   B[0,k] is at V index 0*p + k = k (this is b11 in the subproblem)
    #   B[0,k+1] is at V index k+1 (this is b12)
    #   B[1,k] is at V index p + k = 13+k (this is b21)
    #   B[1,k+1] is at V index p + k + 1 = 14+k (this is b22)
    
    # Let me define helper for Strassen on a 2x2*2x2 block with column offset k
    def add_strassen(k, col_start):
        """Add Strassen decomposition for columns k and k+1 starting at column col_start."""
        nonlocal col_offset
        # Map subproblem indices to global V indices
        # b11 = B[0,k] -> V index k
        # b12 = B[0,k+1] -> V index k+1
        # b21 = B[1,k] -> V index 13+k
        # b22 = B[1,k+1] -> V index 14+k
        
        # Map subproblem C indices to global W indices
        # c11 = C[0,k] -> W index 2*k
        # c12 = C[0,k+1] -> W index 2*(k+1)
        # c21 = C[1,k] -> W index 2*k+1
        # c22 = C[1,k+1] -> W index 2*(k+1)+1
        
        # Strassen products:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # m1
        c = col_start
        U[0, c] += 1; U[3, c] += 1
        V[k, c] += 1; V[14+k, c] += 1
        W[2*k, c] += 1; W[2*k+3, c] += 1
        
        # m2
        c = col_start + 1
        U[2, c] += 1; U[3, c] += 1
        V[k, c] += 1
        W[2*k+1, c] += 1; W[2*k+3, c] -= 1
        
        # m3
        c = col_start + 2
        U[0, c] += 1
        V[k+1, c] += 1; V[14+k, c] -= 1
        W[2*k, c] += 1; W[2*k+2, c] += 1
        
        # m4
        c = col_start + 3
        U[3, c] += 1
        V[13+k, c] += 1; V[k, c] -= 1
        W[2*k, c] += 1; W[2*k+1, c] += 1
        
        # m5
        c = col_start + 4
        U[0, c] += 1; U[1, c] += 1
        V[14+k, c] += 1
        W[2*k, c] -= 1; W[2*k+2, c] += 1
        
        # m6
        c = col_start + 5
        U[2, c] += 1; U[0, c] -= 1
        V[k, c] += 1; V[k+1, c] += 1
        W[2*k+3, c] += 1
        
        # m7
        c = col_start + 6
        U[1, c] += 1; U[3, c] -= 1
        V[13+k, c] += 1; V[14+k, c] += 1
        W[2*k, c] += 1; W[2*k+3, c] += 1
        
        col_offset = col_start + 7
    
    # For each pair of columns
    for pair_idx in range(6):
        k = 2 * pair_idx  # columns 0,1; 2,3; 4,5; 6,7; 8,9; 10,11
        col_start = 7 * pair_idx
        add_strassen(k, col_start)
    
    # Single column 12: 2x2 * 2x1, schoolbook rank 4
    # C[0,12] = a11*B[0,12] + a12*B[1,12]
    # C[1,12] = a21*B[0,12] + a22*B[1,12]
    
    col_start = 42
    # Product 1: a11 * B[0,12]
    c = col_start
    U[0, c] = 1
    V[12, c] = 1
    W[24, c] = 1  # W index for C[0,12] = 2*12 + 0 = 24
    
    # Product 2: a12 * B[1,12]
    c = col_start + 1
    U[1, c] = 1
    V[25, c] = 1  # V index for B[1,12] = 13+12 = 25
    W[24, c] = 1
    
    # Product 3: a21 * B[0,12]
    c = col_start + 2
    U[2, c] = 1
    V[12, c] = 1
    W[25, c] = 1  # W index for C[1,12] = 2*12 + 1 = 25
    
    # Product 4: a22 * B[1,12]
    c = col_start + 3
    U[3, c] = 1
    V[25, c] = 1
    W[25, c] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
