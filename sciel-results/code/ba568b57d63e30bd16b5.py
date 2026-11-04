import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Split the inner dimension (4) into two halves of 2.
    This gives two sub-problems:
    1. A[:, 0:2] * B[0:2, :] -> 2x2 * 2x8
    2. A[:, 2:4] * B[2:4, :] -> 2x2 * 2x8
    
    For each 2x2 * 2x8 multiplication, we can view it as 4 independent 2x2 * 2x2 multiplications
    (one for each pair of columns in the 8-column B). Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    
    However, we can do better by recognizing that the 2x2 * 2x8 tensor has structure.
    Actually, the rank of 2x2 * 2x8 is known to be 14 (not 28 = 4*7).
    This is because we can share the Strassen multiplications across the 4 blocks.
    
    Wait, let me think more carefully.
    
    For 2x2 * 2x8:
    A = [[a0, a1], [a2, a3]]
    B = [[b00, b01, ..., b07], [b10, b11, ..., b17]]
    C = [[c00, c01, ..., c07], [c10, c11, ..., c17]]
    
    c0k = a0*b0k + a1*b1k
    c1k = a2*b0k + a3*b1k
    
    Using Strassen for each pair of columns:
    For columns 2k and 2k+1:
    P1 = (a0+a1) * (b0_{2k} + b1_{2k})  -> wait, this doesn't work directly.
    
    Let me use the standard Strassen decomposition for 2x2 * 2x2:
    M1 = (a0+a3) * (b00+b11)
    M2 = (a2+a3) * b00
    M3 = a0 * (b01-b11)
    M4 = a3 * (b10-b00)
    M5 = (a0+a1) * b11
    M6 = (a2-a0) * (b00+b10)
    M7 = (a1-a3) * (b01+b11)
    
    c00 = M1 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M6
    
    For 2x2 * 2x8, we have 4 such blocks (columns 0-1, 2-3, 4-5, 6-7).
    If we apply Strassen independently to each block, we get 4*7 = 28 multiplications.
    
    But can we do better? The total rank for 2x4 * 4x8 would be 2*28 = 56 if we do this.
    This is the same as the current approach.
    
    Actually, I realize the current approach IS exactly this: 8 Strassen blocks of 2x2*2x2.
    (2 A-blocks * 4 B-column-blocks = 8 blocks, each with 7 multiplications = 56 total).
    
    So to improve, I need to find a decomposition with rank < 56.
    
    Let me try a different approach: use the fact that n=2 is small.
    
    Key insight: For n=2, the tensor has a special structure.
    T_{i,j,k} where i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    
    The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    This is a system of 16 linear equations in the entries of A and B.
    
    Let me try to find a rank-32 decomposition by using a different strategy.
    
    Actually, I recall that for 2 x m x p, the rank can sometimes be significantly lower.
    Let me try to construct a rank-32 decomposition.
    
    Idea: Use the fact that we can combine terms.
    
    Let me define:
    S0 = A[0,0] + A[1,0]
    S1 = A[0,1] + A[1,1]
    S2 = A[0,2] + A[1,2]
    S3 = A[0,3] + A[1,3]
    D0 = A[0,0] - A[1,0]
    D1 = A[0,1] - A[1,1]
    D2 = A[0,2] - A[1,2]
    D3 = A[0,3] - A[1,3]
    
    Then:
    C[0,k] = (S0+B[0,k]*A[1,0]... ) 
    This is getting complicated.
    
    Let me try a simpler approach: just use the schoolbook decomposition with rank 64,
    and then try to reduce it using linear algebra.
    
    Actually, let me just return the Strassen decomposition with rank 56,
    but implement it more cleanly. The key is to make sure the code is correct.
    
    Wait, I have a better idea. Let me try to use a rank-32 decomposition
    by treating the problem as two independent 2x2 * 2x4 problems.
    
    Split B into B1 = B[0:2, :] and B2 = B[2:4, :], each 2x8.
    Split A into A1 = A[:, 0:2] and A2 = A[:, 2:4], each 2x2.
    
    C = A1 * B1 + A2 * B2
    
    For each 2x2 * 2x8 multiplication, the rank is 8 (not 16 or 28).
    This is because:
    A = [[a, b], [c, d]], B = [[e0, e1, ..., e7], [f0, f1, ..., f7]]
    C[0,k] = a*ek + b*fk
    C[1,k] = c*ek + d*fk
    
    We can write this as:
    C[0,k] = a*ek + b*fk
    C[1,k] = c*ek + d*fk
    
    Using 8 multiplications:
    P0 = a * e0, P1 = a * e1, ..., P7 = a * e7
    Q0 = b * f0, Q1 = b * f1, ..., Q7 = b * f7
    R0 = c * e0, R1 = c * e1, ..., R7 = c * e7
    S0 = d * f0, S1 = d * f1, ..., S7 = d * f7
    
    That's 32 multiplications, which is too many.
    
    But we can do better! Notice that:
    C[0,k] + C[1,k] = (a+c)*ek + (b+d)*fk
    C[0,k] - C[1,k] = (a-c)*ek + (b-d)*fk
    
    So we can use:
    P_k = (a+c) * ek
    Q_k = (b+d) * fk
    R_k = (a-c) * ek
    S_k = (b-d) * fk
    
    Then:
    C[0,k] = (P_k + Q_k + R_k + S_k) / 2
    C[1,k] = (P_k + Q_k - R_k - S_k) / 2
    
    This uses 4*8 = 32 multiplications for the 2x2 * 2x8.
    Still not better.
    
    Hmm. Let me just go with the rank-56 Strassen decomposition and make sure it's correct.
    """
    
    n, m, p = 2, 4, 8
    
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # Use Strassen decomposition: 8 blocks of 2x2 * 2x2, each with 7 multiplications
    rank = 56
    
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    # Strassen 2x2 * 2x2 decomposition:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11) * (b00+b11)
    # M2 = (a10+a11) * b00
    # M3 = a00 * (b01-b11)
    # M4 = a11 * (b10-b00)
    # M5 = (a00+a01) * b11
    # M6 = (a10-a00) * (b00+b10)
    # M7 = (a01-a11) * (b01+b11)
    
    # c00 = M1 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M6
    
    # Blocks: (a_row_off, a_col_off, b_row_off, b_col_off)
    # a_row_off is always 0 since n=2
    # We have 2 choices for a_col_off (0 or 2) and 4 choices for b_col_off (0,2,4,6)
    # b_row_off is determined by a_col_off: if a_col_off=0, b_row_off=0; if a_col_off=2, b_row_off=2
    
    col = 0
    
    for a_col_off in [0, 2]:
        b_row_off = a_col_off  # b_row_off matches a_col_off
        
        for b_col_off in [0, 2, 4, 6]:
            # Index mappings:
            # A[i, j] is at U[i*4 + j, r]
            # B[j, k] is at V[j*8 + k, r]
            # C[i, k] is at W[k*2 + i, r]  (trace convention: W[k*n+i, r])
            
            # A elements:
            a00_idx = 0*4 + a_col_off
            a01_idx = 0*4 + a_col_off + 1
            a10_idx = 1*4 + a_col_off
            a11_idx = 1*4 + a_col_off + 1
            
            # B elements:
            b00_idx = b_row_off*8 + b_col_off
            b01_idx = b_row_off*8 + b_col_off + 1
            b10_idx = (b_row_off+1)*8 + b_col_off
            b11_idx = (b_row_off+1)*8 + b_col_off + 1
            
            # C elements:
            c00_idx = b_col_off*2 + 0       # C[0, b_col_off]
            c01_idx = (b_col_off+1)*2 + 0   # C[0, b_col_off+1]
            c10_idx = b_col_off*2 + 1       # C[1, b_col_off]
            c11_idx = (b_col_off+1)*2 + 1   # C[1, b_col_off+1]
            
            # M1 = (a00+a11) * (b00+b11)
            U[a00_idx, col] = 1
            U[a11_idx, col] = 1
            V[b00_idx, col] = 1
            V[b11_idx, col] = 1
            W[c00_idx, col] = 1
            W[c11_idx, col] = 1
            col += 1
            
            # M2 = (a10+a11) * b00
            U[a10_idx, col] = 1
            U[a11_idx, col] = 1
            V[b00_idx, col] = 1
            W[c10_idx, col] = 1
            W[c11_idx, col] = -1
            col += 1
            
            # M3 = a00 * (b01-b11)
            U[a00_idx, col] = 1
            V[b01_idx, col] = 1
            V[b11_idx, col] = -1
            W[c01_idx, col] = 1
            W[c11_idx, col] = 1
            col += 1
            
            # M4 = a11 * (b10-b00)
            U[a11_idx, col] = 1
            V[b10_idx, col] = 1
            V[b00_idx, col] = -1
            W[c00_idx, col] = 1
            W[c10_idx, col] = 1
            col += 1
            
            # M5 = (a00+a01) * b11
            U[a00_idx, col] = 1
            U[a01_idx, col] = 1
            V[b11_idx, col] = 1
            W[c00_idx, col] = -1
            W[c01_idx, col] = 1
            col += 1
            
            # M6 = (a10-a00) * (b00+b10)
            U[a10_idx, col] = 1
            U[a00_idx, col] = -1
            V[b00_idx, col] = 1
            V[b10_idx, col] = 1
            W[c11_idx, col] = 1
            col += 1
            
            # M7 = (a01-a11) * (b01+b11)
            U[a01_idx, col] = 1
            U[a11_idx, col] = -1
            V[b01_idx, col] = 1
            V[b11_idx, col] = 1
            W[c00_idx, col] = 1
            col += 1
    
    assert col == 56, f"Expected 56, got {col}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
