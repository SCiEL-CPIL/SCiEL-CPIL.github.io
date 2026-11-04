"""Matrix multiplication 4x4x6 tensor decomposition using Kronecker product of Strassen structures."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) where:
      U[i*2+j, r] = coefficient for A[i,j] in rank-r
      V[j*2+k, r] = coefficient for B[j,k] in rank-r
      W[k*2+i, r] = coefficient for C[i,k] in rank-r  (trace convention W[k*n+i])
    
    The 7 rank-1 tensors correspond to:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B10-B11)
      M4 = A11*(B01-B11)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B10)
      M7 = (A01-A11)*(B01+B11)
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Index: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
    # Wait, let me use the standard Strassen formulas carefully.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1   # C00: k=0,i=0 -> W[0*2+0]=W[0]
    W[1, 0] = -1  # C01: k=1,i=0 -> W[1*2+0]=W[1]... 
    
    # Wait, I need to be more careful. Let me redo this.
    # W[k*n+i, r] where n=2. So W[k*2+i, r].
    # C[i,k] corresponds to W[k*2+i, r].
    
    # Let me restart with explicit mapping.
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    def u(idx_i, idx_j, col, val):
        U[idx_i * 2 + idx_j, col] += val
    
    def v(idx_j, idx_k, col, val):
        V[idx_j * 2 + idx_k, col] += val
    
    def w(idx_i, idx_k, col, val):
        # C[i,k] -> W[k*2+i, col]
        W[idx_k * 2 + idx_i, col] += val
    
    # M1 = (A00+A11)*(B00+B11)
    u(0,0,0,1); u(1,1,0,1)
    v(0,0,0,1); v(1,1,0,1)
    w(0,0,0,1); w(0,1,0,-1); w(1,0,0,1); w(1,1,0,-1)
    # C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
    
    # M2 = (A10+A11)*B00
    u(1,0,1,1); u(1,1,1,1)
    v(0,0,1,1)
    w(1,0,1,1)
    # C10 += M2
    
    # M3 = A00*(B10-B11)
    u(0,0,2,1)
    v(1,0,2,1); v(1,1,2,-1)
    w(0,1,2,1); w(1,1,2,-1)
    # C01 += M3, C11 -= M3
    
    # M4 = A11*(B01-B11)
    u(1,1,3,1)
    v(0,1,3,1); v(1,1,3,-1)
    w(0,0,3,-1); w(1,0,3,1)
    # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)*B11
    u(0,0,4,1); u(0,1,4,1)
    v(1,1,4,1)
    w(0,0,4,1); w(0,1,4,1)
    # C00 += M5, C01 += M5
    
    # M6 = (A10-A00)*(B00+B10)
    u(1,0,5,1); u(0,0,5,-1)
    v(0,0,5,1); v(1,0,5,1)
    w(1,1,5,1)
    # C11 += M6
    
    # M7 = (A01-A11)*(B01+B11)
    u(0,1,6,1); u(1,1,6,-1)
    v(0,1,6,1); v(1,1,6,1)
    w(0,0,6,-1); w(1,1,6,-1)
    # C00 -= M7, C11 -= M7
    
    return U, V, W

def _strassen_2x2x3():
    """
    Return a rank-11 decomposition for 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    U shape: (4, 11), V shape: (6, 11), W shape: (6, 11)
    W[k*2+i, r] for C[i,k].
    
    We can construct this by viewing the 2x2x3 multiplication as:
    Split B into two 2x2 matrices (columns 0-1 and column 2 duplicated?) 
    Actually, a cleaner approach: use the fact that 2x2x3 can be seen as
    computing C[:,0:2] via 2x2x2 Strassen and C[:,2] separately, but that gives 7+4=11.
    
    More precisely:
    - For C[:,0:2] (2x2 output from 2x2 * 2x2): use rank-7 Strassen
    - For C[:,2] (2x1 output from 2x2 * 2x1): use rank-4 schoolbook
    
    Total rank = 7 + 4 = 11.
    """
    n = 2
    m = 2
    p = 3
    rank = 11
    U = np.zeros((n*m, rank), dtype=float)  # (4, 11)
    V = np.zeros((m*p, rank), dtype=float)  # (6, 11)
    W = np.zeros((p*n, rank), dtype=float)  # (6, 11)
    
    # Part 1: Strassen for C[:, 0:2] (2x2x2)
    U_s, V_s, W_s = _strassen_2x2x2()
    # U_s: (4,7), V_s: (4,7), W_s: (4,7)
    # V_s[j*2+k] for k in {0,1} -> maps to V[j*3+k] for k in {0,1}
    # W_s[k*2+i] for k in {0,1} -> maps to W[k*2+i] for k in {0,1}
    
    U[:, 0:7] = U_s
    for j in range(2):
        for k in range(2):
            V[j*3 + k, 0:7] = V_s[j*2 + k, 0:7]
    for k in range(2):
        for i in range(2):
            W[k*2 + i, 0:7] = W_s[k*2 + i, 0:7]
    
    # Part 2: Schoolbook for C[:, 2] (2x2 * 2x1)
    # C[i,2] = sum_j A[i,j] * B[j,2]
    # 4 terms: (i,j) in {(0,0),(0,1),(1,0),(1,1)}
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        col = 7 + idx
        U[i*2 + j, col] = 1
        V[j*3 + 2, col] = 1  # B[j, 2]
        W[2*2 + i, col] = 1  # C[i, 2] -> W[2*2+i] = W[4+i]
    
    return U, V, W

def construct():
    """
    Construct rank-77 decomposition for 4x4x6 matrix multiplication.
    
    Strategy: Kronecker product of (2x2x2 Strassen, rank 7) and (2x2x3 Strassen, rank 11).
    Total rank = 7 * 11 = 77.
    
    The 4x4x6 multiplication tensor T446[i,j,k] = delta_{i,j} * delta_{j,k} (in tensor form)
    can be written as the Kronecker product of T222 and T223 when we view:
    - 4 = 2*2, 4 = 2*2, 6 = 2*3
    - A is 4x4, viewed as 2x2 blocks of 2x2
    - B is 4x6, viewed as 2x2 blocks of 2x3  
    - C is 4x6, viewed as 2x2 blocks of 2x3
    
    The Kronecker product of decompositions gives a valid decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Get the base decompositions
    U1, V1, W1 = _strassen_2x2x2()   # rank 7, shapes (4,7), (4,7), (4,7)
    U2, V2, W2 = _strassen_2x2x3()   # rank 11, shapes (4,11), (6,11), (6,11)
    
    rank1 = 7
    rank2 = 11
    total_rank = rank1 * rank2  # 77
    
    # Kronecker product structure:
    # The 4x4x6 tensor = (2x2x2) ⊗ (2x2x3) in the appropriate sense.
    # 
    # For the Kronecker product of two tensor decompositions:
    # If T1 = sum_r u1[r] ⊗ v1[r] ⊗ w1[r] and T2 = sum_s u2[s] ⊗ v2[s] ⊗ w2[s],
    # then T1 ⊗ T2 = sum_{r,s} (u1[r]⊗u2[s]) ⊗ (v1[r]⊗v2[s]) ⊗ (w1[r]⊗w2[s])
    #
    # We need to map indices correctly.
    # 
    # A is 4x4 = (2x2) ⊗ (2x2): A[i1*2+i2, j1*2+j2]
    # B is 4x6 = (2x2) ⊗ (2x3): B[j1*2+j2, k1*3+k2]
    # C is 4x6 = (2x2) ⊗ (2x3): C[i1*2+i2, k1*3+k2]
    #
    # Wait, this doesn't quite work because the dimensions don't factor as a simple tensor product
    # in the standard way for matrix multiplication. Let me think more carefully.
    #
    # Actually, for matrix multiplication, the tensor is T[i,j,k] = 1 if the indices are consistent.
    # For 4x4x6: T[i,j,k] = 1 where i in [0,4), j in [0,4), k in [0,6).
    # The multiplication is C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # The Kronecker product approach works when we can write:
    # 4 = n1*n2, 4 = m1*m2, 6 = p1*p2
    # with n1=2,m1=2,p1=2 and n2=2,m2=2,p2=3.
    #
    # Then C[i1*n2+i2, k1*p2+k2] = sum_{j1,j2} A[i1*n2+i2, j1*m2+j2] * B[j1*m2+j2, k1*p2+k2]
    #
    # This is NOT a simple Kronecker product of two independent multiplications because
    # the summation is over both j1 and j2 coupled.
    #
    # However, we CAN use a block structure. Let's think of it differently:
    #
    # Write A as a 2x2 block matrix where each block is 2x2:
    # A = [[A00, A01], [A10, A11]] where Aab is 2x2
    # B as a 2x3 block matrix where each block is 2x2:
    # B = [[B00, B01, B02], [B10, B11, B12]] where Bab is 2x2
    #
    # Then C = A*B is a 2x3 block matrix where each block is 2x2:
    # C = [[C00, C01, C02], [C10, C11, C12]]
    # C_ab = sum_c A_ac * B_cb
    #
    # This is a 2x2x3 "super-matrix" multiplication where each entry is a 2x2 matrix.
    # The super-multiplication has rank equal to (rank of 2x2x3 scalar mult) * (rank of 2x2x2 scalar mult)
    # = 11 * 7 = 77 if we use the Kronecker product of the decompositions.
    #
    # Specifically, for each pair (r,s) where r indexes the 2x2x2 Strassen rank and
    # s indexes the 2x2x3 decomposition rank:
    # The rank-(r,s) term contributes to all blocks.
    #
    # Let me construct this explicitly.
    
    U = np.zeros((n*m, total_rank), dtype=float)  # (16, 77)
    V = np.zeros((m*p, total_rank), dtype=float)  # (24, 77)
    W = np.zeros((p*n, total_rank), dtype=float)  # (24, 77)
    
    # For each (r, s) pair:
    for r in range(rank1):
        for s in range(rank2):
            col = r * rank2 + s
            
            # U part: A[i,j] where i = i1*2+i2, j = j1*2+j2
            # The 2x2x2 Strassen gives coefficients for A[i1,j1] (block level)
            # The 2x2x3 Strassen gives coefficients for A[i2,j2] (within block)
            # Wait, I need to be more careful about which factor corresponds to which.
            
            # Let me reconsider. The "super" multiplication is:
            # C_block[i1,j1] (2x2 matrix) = sum_{c} A_block[i1,c] * B_block[c,j1]
            # where A_block[i1,c] is 2x2, B_block[c,j1] is 2x2, result is 2x2.
            #
            # Each block multiplication A_block[i1,c] * B_block[c,j1] is a 2x2x2 multiplication.
            # The super-structure is 2x2x3 (2 block rows, 2 block cols for A; 2 block rows, 3 block cols for B).
            #
            # So the total decomposition is:
            # For the super-structure (2x2x3), use the rank-11 decomposition.
            # For each block multiplication (2x2x2), use the rank-7 Strassen decomposition.
            #
            # The Kronecker product:
            # U_total[i1*2+i2, j1*2+j2, r1*11+r2] = U_super[i1,j1, r1] * U_block[i2,j2, r2]
            # V_total[j1*2+j2, k1*3+k2, r1*11+r2] = V_super[j1,k1, r1] * V_block[j2,k2, r2]
            # W_total[k1*3+k2, i1*2+i2, r1*11+r2] = W_super[k1,i1, r1] * W_block[k2,i2, r2]
            #
            # Wait, I need to double-check the index conventions.
            #
            # U has shape (n*m, rank) = (16, 77). Index: U[i*m+j, col] where i in [0,4), j in [0,4).
            # i = i1*2 + i2, j = j1*2 + j2 where i1,j1 in [0,2) (block indices), i2,j2 in [0,2) (within block).
            #
            # V has shape (m*p, rank) = (24, 77). Index: V[j*p+k, col] where j in [0,4), k in [0,6).
            # j = j1*2 + j2, k = k1*3 + k2 where j1 in [0,2), k1 in [0,3), j2 in [0,2), k2 in [0,3).
            #
            # W has shape (p*n, rank) = (24, 77). Index: W[k*n+i, col] where k in [0,6), i in [0,4).
            # k = k1*3 + k2, i = i1*2 + i2.
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i*4 + j, col] = U1[i1*2 + j1, r] * U2[i2*2 + j2, s]
            
            for j1 in range(2):
                for k1 in range(3):
                    for j2 in range(2):
                        for k2 in range(3):
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            V[j*6 + k, col] = V1[j1*2 + k1, r] * V2[j2*3 + k2, s]
            
            for k1 in range(3):
                for i1 in range(2):
                    for k2 in range(3):
                        for i2 in range(2):
                            k = k1 * 3 + k2
                            i = i1 * 2 + i2
                            W[k*4 + i, col] = W1[k1*2 + i1, r] * W2[k2*2 + i2, s]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
