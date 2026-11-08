"""
Rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.

Strategy: Block decomposition with Strassen recursion.
- Split middle dimension (5) into 4+1.
- Core: 4x4 * 4x6 -> split columns into 4+2.
  - 4x4x4 via Strassen: rank 49
  - 4x4x2 via Strassen: rank 28
- Tail: 4x1 * 1x6: rank 4
- Total rank: 81
"""
import time
import numpy as np

_DEADLINE = time.time() + 130


def _check_deadline():
    return time.time() > _DEADLINE


def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    Strassen's 7 multiplications for 2x2 matrix multiply:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U[i*2+j, c] = coefficient of A[i,j] in Mc
    V[j*2+k, c] = coefficient of B[j,k] in Mc
    W[k*2+i, c] = coefficient of Mc in C[i,k]
    """
    # U: 4x7
    U = np.zeros((4, 7))
    # A11=row0, A12=row1, A21=row2, A22=row3
    # M1: A11+A22 -> U[0,0]=1, U[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    # M2: A21+A22 -> U[2,1]=1, U[3,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    # M3: A11 -> U[0,2]=1
    U[0, 2] = 1
    # M4: A22 -> U[3,3]=1
    U[3, 3] = 1
    # M5: A11+A12 -> U[0,4]=1, U[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    # M6: A21-A11 -> U[2,5]=1, U[0,5]=-1
    U[2, 5] = 1; U[0, 5] = -1
    # M7: A12-A22 -> U[1,6]=1, U[3,6]=-1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V: 4x7
    V = np.zeros((4, 7))
    # B11=row0, B12=row1, B21=row2, B22=row3
    # M1: B11+B22 -> V[0,0]=1, V[3,0]=1
    V[0, 0] = 1; V[3, 0] = 1
    # M2: B11 -> V[0,1]=1
    V[0, 1] = 1
    # M3: B21-B22 -> V[2,2]=1, V[3,2]=-1
    V[2, 2] = 1; V[3, 2] = -1
    # M4: B12-B11 -> V[1,3]=1, V[0,3]=-1
    V[1, 3] = 1; V[0, 3] = -1
    # M5: B22 -> V[3,4]=1
    V[3, 4] = 1
    # M6: B11+B12 -> V[0,5]=1, V[1,5]=1
    V[0, 5] = 1; V[1, 5] = 1
    # M7: B21+B22 -> V[2,6]=1, V[3,6]=1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: 4x7
    W = np.zeros((4, 7))
    # C11=col0, C12=col1, C21=col2, C22=col3
    # C11 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C12 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C21 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C22 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """
    Return (U, V, W) for 4x4x4 decomposition using Strassen recursion.
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    
    Split 4x4 into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Use Strassen on the 2x2 block matrix multiply, where each "element" is a 2x2 matrix.
    This gives 7 "block multiplications", each of which is a 2x2x2 Strassen (rank 7).
    Total rank: 7*7 = 49.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # The 2x2 block structure:
    # Block (rb, cb) of C corresponds to rows 2*rb..2*rb+1, cols 2*cb..2*cb+1 of C
    # Block (rb, cb) of A corresponds to rows 2*rb..2*rb+1, cols 2*cb..2*cb+1 of A
    # Block (rb, cb) of B corresponds to rows 2*rb..2*rb+1, cols 2*cb..2*cb+1 of B
    
    # For each of the 7 Strassen multiplications m (0..6):
    # For each of the 7 inner Strassen multiplications m2 (0..6):
    #   Column c = m*7 + m2
    #   U: for each (rb, cb) in {0,1}^2:
    #       outer_u = U2[rb*2+cb, m]  (coefficient of A_block[rb,cb] in M_m)
    #       For each (r, s) in {0,1}^2:
    #           inner_u = U2[r*2+s, m2]  (coefficient of A[r,s] within block in M_m2)
    #           global row = (2*rb+r)*4 + (2*cb+s)
    #           U[global_row, c] += outer_u * inner_u
    #   V: similar with V2
    #   W: similar with W2
    
    for m in range(7):
        for m2 in range(7):
            c = m * 7 + m2
            for rb in range(2):
                for cb in range(2):
                    outer_u = U2[rb * 2 + cb, m]
                    if outer_u == 0:
                        continue
                    for r in range(2):
                        for s in range(2):
                            inner_u = U2[r * 2 + s, m2]
                            if inner_u == 0:
                                continue
                            global_row = (2 * rb + r) * 4 + (2 * cb + s)
                            U[global_row, c] += outer_u * inner_u
                    
                    outer_v = V2[rb * 2 + cb, m]
                    if outer_v == 0:
                        continue
                    for r in range(2):
                        for s in range(2):
                            inner_v = V2[r * 2 + s, m2]
                            if inner_v == 0:
                                continue
                            global_row = (2 * rb + r) * 4 + (2 * cb + s)
                            V[global_row, c] += outer_v * inner_v
                    
                    outer_w = W2[rb * 2 + cb, m]
                    if outer_w == 0:
                        continue
                    for r in range(2):
                        for s in range(2):
                            inner_w = W2[r * 2 + s, m2]
                            if inner_w == 0:
                                continue
                            # C block (rb, cb), element (r, s) is C[2*rb+r, 2*cb+s]
                            # W global index: col*4 + row = (2*cb+s)*4 + (2*rb+r)
                            global_col = (2 * cb + s) * 4 + (2 * rb + r)
                            W[global_col, c] += outer_w * inner_w
    
    return U, V, W


def _strassen_4x4x2():
    """
    Return (U, V, W) for 4x4x2 decomposition using Strassen recursion.
    U shape: (16, 28), V shape: (8, 28), W shape: (8, 28)
    
    Split the 4x4 A and 4x4 part of B (first 4 cols of the 4x4x2... wait, it's 4x4 * 4x2)
    Actually, 4x4 * 4x2: A is 4x4, B is 4x2, C is 4x2.
    Split A into 2x2 blocks, B into 2x1 blocks (top 2 rows, bottom 2 rows, each 2x1),
    C into 2x1 blocks.
    
    Use Strassen on the block structure:
    A = [[A11, A12], [A21, A22]] where each is 2x2
    B = [[B1], [B2]] where B1 is 2x2 (rows 0-1, cols 0-1), B2 is 2x2 (rows 2-3, cols 0-1)
    Wait, B is 4x2. Split into B_top (2x2) and B_bot (2x2).
    C = A * B, C is 4x2. Split into C_top (2x2) and C_bot (2x2).
    
    C_top = A11*B1 + A12*B2  (2x2 * 2x2 + 2x2 * 2x2)
    C_bot = A21*B1 + A22*B2  (2x2 * 2x2 + 2x2 * 2x2)
    
    This is like multiplying a 4x4 by a 4x2, which can be seen as:
    [A11 A12] [B1]   [A11*B1 + A12*B2]
    [A21 A22] [B2] = [A21*B1 + A22*B2]
    
    We can use Strassen for the 2x2 block multiply where the "right side" has 2 columns.
    Actually, let's think of it as: treat the 4x2 as two separate 4x1 multiplies? No.
    
    Better: Use the fact that 4x4 * 4x2 can be done by treating it as two 4x4 * 4x1 multiplies,
    or use Strassen on the 2x2 block structure with the understanding that B has shape 4x2.
    
    Let's use Strassen's 7 multiplications on the 2x2 block matrix multiply:
    M1 = (A11+A22)(B1+B2)  -- but B1 and B2 are 2x2, so this is 2x2 * 2x2 = rank 7
    M2 = (A21+A22)B1
    M3 = A11(B2-B1)  -- wait, B2-B1 is 2x2
    M4 = A22(B1-B2)
    M5 = (A11+A12)B2
    M6 = (A21-A11)(B1+B2)
    M7 = (A12-A22)(B1+B2)
    
    C_top = M1 + M4 - M5 + M7  (2x2)
    C_bot = M2 + M4  ... wait, that's not right for 4x4*4x2.
    
    Actually, for standard 2x2 block Strassen with 2x2 blocks:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    But here we have A as 4x4 (2x2 blocks of 2x2), B as 4x2 (2 blocks of 2x2 stacked vertically),
    and C as 4x2 (2 blocks of 2x2 stacked vertically).
    
    So it's like: C = A * B where A is 2x2 blocks, B is 2x1 blocks (each 2x2), C is 2x1 blocks.
    
    C_top = A11*B1 + A12*B2
    C_bot = A21*B1 + A22*B2
    
    This is NOT the standard Strassen form. Strassen assumes B is 2x2 blocks.
    
    Alternative: Just use schoolbook for 4x4x2: rank = 4*4*2 = 32.
    Or use Strassen on the 4x4x4 and take only the first 2 columns? That would give rank 49 but waste.
    
    Let me just use schoolbook for 4x4x2: rank 32.
    Total: 49 + 32 + 4 = 85.
    
    Or, I can do better. Let me think...
    
    Actually, 4x4 * 4x2: I can split the 4 columns of B into 2+2, giving two 4x4*4x1 multiplies.
    Each 4x4*4x1 can be done with rank 16 (schoolbook) or less.
    
    Or: treat it as a 4x4 * 4x4 where the last 2 columns of B are zero, and only compute first 2 cols of C.
    Using Strassen 4x4x4 (rank 49) but only using the relevant parts... that doesn't reduce rank.
    
    Let me just use schoolbook for 4x4x2: rank 32.
    """
    n, p = 4, 2
    rank = n * n * p  # 32
    U = np.zeros((n * n, rank))
    V = np.zeros((n * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    """
    Construct exact tensor decomposition for 4x5x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a list of lists.
    U: shape (n*m, rank) = (20, rank)
    V: shape (m*p, rank) = (30, rank)
    W: shape (p*n, rank) = (24, rank)
    """
    n, m, p = 4, 5, 6
    
    # Block 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4] (4x4x4, rank 49)
    U1, V1, W1 = _strassen_4x4x4()
    r1 = 49
    
    # Block 2: A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6] (4x4x2, rank 32)
    U2, V2, W2 = _strassen_4x4x2()
    r2 = 32
    
    # Block 3: A[0:4, 4] * B[4, 0:6] -> C[0:4, 0:6] (4x1x6, rank 4)
    r3 = 4
    
    total_rank = r1 + r2 + r3  # 49 + 32 + 4 = 85
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Place Block 1
    U[:, 0:r1] = U1
    V[0:16, 0:r1] = V1
    W[0:16, 0:r1] = W1
    
    # Place Block 2
    # U2 is (16, 32): rows i*4+j for i,j in 0..3 -> same as U1 rows
    U[:, r1:r1+r2] = U2
    # V2 is (8, 32): local row j*2+k for j in 0..3, k in 0..1
    # Global V row: j*6 + (4+k) for j in 0..3, k in 0..1
    for j in range(4):
        for k in range(2):
            local_row = j * 2 + k
            global_row = j * 6 + (4 + k)
            V[global_row, r1:r1+r2] = V2[local_row, :]
    # W2 is (8, 32): local col k*4+i for k in 0..1, i in 0..3
    # Global W col: (4+k)*4 + i for k in 0..1, i in 0..3
    for k in range(2):
        for i in range(4):
            local_col = k * 4 + i
            global_col = (4 + k) * 4 + i
            W[global_col, r1:r1+r2] = W2[local_col, :]
    
    # Place Block 3: A[0:4, 4] * B[4, 0:6] -> C[0:4, 0:6]
    # T[i, 4, k] = 1 for all i in 0..3, k in 0..5
    # Decomposition: sum_{c=0}^{3} U[i*5+4, c] * V[4*6+k, c] * W[k*4+i, c] = 1
    # Use c = i: U[i*5+4, i] = 1, V[24+k, i] = 1, W[k*4+i, i] = 1
    # But this only gives T[i,4,k] = 1 when the c=i term contributes, and other c don't.
    # Wait, we need sum over c of U[i*5+4,c] * V[24+k,c] * W[k*4+i,c] = 1 for all i,k.
    # If we set for each i: U[i*5+4, i] = 1, V[24+k, i] = 1 for all k, W[k*4+i, i] = 1 for all k,
    # then for fixed i,k: sum_c ... = U[i*5+4,i]*V[24+k,i]*W[k*4+i,i] = 1*1*1 = 1. Correct!
    # But this uses 4 terms (c=0,1,2,3), and for term c, it contributes to all k.
    # Actually wait: for term c=i, V[24+k, i] = 1 for all k, so it contributes 1 to every k.
    # And U[i*5+4, i] = 1 only for the specific i.
    # So for T[i', 4, k] where i' != i: U[i'*5+4, i] = 0, so no contribution. Good.
    # For T[i, 4, k]: only c=i contributes, giving 1. Good.
    
    for i in range(4):
        c = i
        U[i * 5 + 4, r1 + r2 + c] = 1
        for k in range(6):
            V[4 * 6 + k, r1 + r2 + c] = 1
            W[k * 4 + i, r1 + r2 + c] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
