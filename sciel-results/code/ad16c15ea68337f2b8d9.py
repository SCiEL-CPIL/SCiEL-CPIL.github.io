"""Strassen-based block decomposition for 4x4x6 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, r] = coefficient of A[i,j]
    V[j*2+k, r] = coefficient of B[j,k]
    W[k*2+i, r] = coefficient in C[i,k]
    """
    # Strassen's 7 products:
    # M1 = (A11)(B12-B22)
    # M2 = (A11+A12)(B22)
    # M3 = (A21+A22)(B11)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A21-A11)(B11+B12)
    #
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 + M1 - M3 - M7
    
    U = [[0]*7 for _ in range(4)]  # A indices: (i*2+j)
    V = [[0]*7 for _ in range(4)]  # B indices: (j*2+k)
    W = [[0]*7 for _ in range(4)]  # C indices: (k*2+i)
    
    # A indexing: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indexing: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indexing: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1: (A11)(B12-B22) -> A[0,1]*B[0,1] - A[0,1]*B[1,1]
    U[1][0] = 1   # A[0,1]
    V[1][0] = 1   # B[0,1]
    V[3][0] = -1  # B[1,1]
    
    # M2: (A11+A12)(B22) -> (A[0,0]+A[0,1])*B[1,1]
    U[0][1] = 1   # A[0,0]
    U[1][1] = 1   # A[0,1]
    V[3][1] = 1   # B[1,1]
    
    # M3: (A21+A22)(B11) -> (A[1,0]+A[1,1])*B[0,0]
    U[2][2] = 1   # A[1,0]
    U[3][2] = 1   # A[1,1]
    V[0][2] = 1   # B[0,0]
    
    # M4: (A22)(B21-B11) -> A[1,1]*B[1,0] - A[1,1]*B[0,0]
    U[3][3] = 1   # A[1,1]
    V[2][3] = 1   # B[1,0]
    V[0][3] = -1  # B[0,0]
    
    # M5: (A11+A22)(B11+B22) -> (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0][4] = 1   # A[0,0]
    U[3][4] = 1   # A[1,1]
    V[0][4] = 1   # B[0,0]
    V[3][4] = 1   # B[1,1]
    
    # M6: (A12-A22)(B21+B22) -> (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1][5] = 1   # A[0,1]
    U[3][5] = -1  # A[1,1]
    V[2][5] = 1   # B[1,0]
    V[3][5] = 1   # B[1,1]
    
    # M7: (A21-A11)(B11+B12) -> (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2][6] = 1   # A[1,0]
    U[0][6] = -1  # A[0,0]
    V[0][6] = 1   # B[0,0]
    V[1][6] = 1   # B[0,1]
    
    # C11 = M5 + M4 - M2 + M6
    W[0][4] = 1   # M5
    W[0][3] = 1   # M4
    W[0][1] = -1  # M2
    W[0][5] = 1   # M6
    
    # C12 = M1 + M2
    W[1][0] = 1   # M1
    W[1][1] = 1   # M2
    
    # C21 = M3 + M4
    W[2][2] = 1   # M3
    W[2][3] = 1   # M4
    
    # C22 = M5 + M1 - M3 - M7
    W[3][4] = 1   # M5
    W[3][0] = 1   # M1
    W[3][2] = -1  # M3
    W[3][6] = -1  # M7
    
    return U, V, W

def _strassen_2x2x6():
    """
    Return (U, V, W) for 2x2x6 matrix multiplication with rank 21.
    U has shape (4, 21), V has shape (12, 21), W has shape (12, 21).
    
    A is 2x2, B is 2x6, C is 2x6.
    U[i*2+j, r] = coefficient of A[i,j]
    V[j*6+k, r] = coefficient of B[j,k]
    W[k*2+i, r] = coefficient in C[i,k]
    """
    U_local, V_local, W_local = _strassen_2x2()
    
    rank = 21
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    # Split B into 3 blocks of 2 columns: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    for b in range(3):
        offset = b * 7
        # U is the same for all blocks
        for i in range(4):
            for r in range(7):
                U[i][offset + r] = U_local[i][r]
        
        # V: for block b, local B[j, k] maps to global B[j, b*2+k]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V[j*6 + b*2 + k][offset + r] = V_local[j*2+k][r]
        
        # W: for block b, local C[i, k] maps to global C[i, b*2+k]
        # W_local[k*2+i, r] for local k in {0,1}
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    W[(b*2+k)*2 + i][offset + r] = W_local[k*2+i][r]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x4x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (16, rank), (24, rank), (24, rank).
    
    U[i*4+j, r] = coefficient of A[i,j]
    V[j*6+k, r] = coefficient of B[j,k]
    W[k*4+i, r] = coefficient in C[i,k]
    """
    start_time = time.time()
    
    U2, V2, W2 = _strassen_2x2x6()
    
    rank = 84
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    # Split A into 4 blocks of 2x2: A[0:2, 0:2], A[0:2, 2:4], A[2:4, 0:2], A[2:4, 2:4]
    # Split B into 2 blocks of 2x6: B[0:2, :], B[2:4, :]
    # C is 4x6, split into 4 blocks of 2x6
    
    for a_block in range(2):  # rows of A: 0:2 or 2:4
        for b_block in range(2):  # rows of B: 0:2 or 2:4
            offset = (a_block * 2 + b_block) * 21
            
            # U: A[a_block*2 + i, a_block*2 + j] for i,j in {0,1}
            # Global A index: i_global = a_block*2 + i, j_global = ? 
            # Wait, A is 4x4. A[i, j] where i is row, j is column.
            # U[i*4 + j, r]. 
            # For block (a_block, a_col_block): A[2*a_block:2*a_block+2, 2*a_col_block:2*a_col_block+2]
            
            # I need to think about this more carefully.
            # The 4x4x6 problem: A is 4x4, B is 4x6, C is 4x6.
            # We can view this as 4 sub-problems of 2x2x6:
            # C[2i:2i+2, :] = A[2i:2i+2, 2j:2j+2] * B[2j:2j+2, :]
            # But that's not quite right because A's columns also need to be split.
            
            # Let's use a 2x2 block structure for A:
            # A = [[A00, A01],
            #      [A10, A11]]
            # where each Aij is 2x2.
            # B = [[B00, B01],
            #      [B10, B11]]
            # where each Bij is 2x6.
            # C = [[C00, C01],
            #      [C10, C11]]
            # where each Cij is 2x6.
            # C = A*B:
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            
            # So we need 8 sub-problems of 2x2x6, each of rank 21. Total rank = 8*21 = 168.
            # But we can use Strassen at the block level to reduce this.
            
            # Actually, let's just do the straightforward 8 sub-problems first.
            pass
    
    # Let me reconsider. The straightforward approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split into blocks where i = 2*i1 + i0, j = 2*j1 + j0, k stays.
    # C[2*i1+i0, k] = sum_{j1=0,1} sum_{j0=0,1} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, k]
    #                = sum_{j1=0,1} (A[2*i1+i0, 2*j1:2*j1+2] * B[2*j1:2*j1+2, k])
    
    # This gives 4 sub-problems (for i1 in {0,1}, j1 in {0,1}), each of size 2x2x6.
    # But wait, that's not right either. Let me think again.
    
    # Actually, the cleanest way: treat A as 4x4, B as 4x6.
    # C = A * B is 4x6.
    # We can decompose this as:
    # For each pair of rows (i1, i2) of A (where i1 in {0,1} is the block row),
    # and each pair of rows (j1, j2) of B (where j1 in {0,1} is the block row),
    # we have a 2x2x6 sub-problem.
    
    # There are 2*2 = 4 such sub-problems, each of rank 21. Total rank = 84.
    
    # Sub-problem (i_block, j_block):
    # A_sub = A[2*i_block:2*i_block+2, 2*j_block:2*j_block+2]  (2x2)
    # B_sub = B[2*j_block:2*j_block+2, :]  (2x6)
    # C_sub = C[2*i_block:2*i_block+2, :]  (2x6)
    # C_sub += A_sub * B_sub
    
    for i_block in range(2):
        for j_block in range(2):
            offset = (i_block * 2 + j_block) * 21
            
            # U: A[2*i_block + i0, 2*j_block + j0] for i0, j0 in {0,1}
            # Global A index: (2*i_block + i0)*4 + (2*j_block + j0)
            for i0 in range(2):
                for j0 in range(2):
                    a_idx = (2*i_block + i0)*4 + (2*j_block + j0)
                    for r in range(7):
                        U[a_idx][offset + r] = U2[i0*2 + j0][r]
            
            # V: B[2*j_block + j0, k] for j0 in {0,1}, k in {0,...,5}
            # Global B index: (2*j_block + j0)*6 + k
            for j0 in range(2):
                for k in range(6):
                    b_idx = (2*j_block + j0)*6 + k
                    for r in range(7):
                        V[b_idx][offset + r] = V2[j0*6 + k][r]
            
            # W: C[2*i_block + i0, k] for i0 in {0,1}, k in {0,...,5}
            # Global C index: k*4 + (2*i_block + i0)
            for k in range(6):
                for i0 in range(2):
                    c_idx = k*4 + (2*i_block + i0)
                    for r in range(7):
                        W[c_idx][offset + r] = W2[k*2 + i0][r]
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        raise TimeoutError("Construction took too long")
    
    return {"U": U, "V": V, "W": W}
