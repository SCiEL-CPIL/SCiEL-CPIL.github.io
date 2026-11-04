"""Program for rank-minimal tensor decomposition of 3x4 * 4x8 matrix multiplication."""
import numpy as np
from collections import defaultdict
from itertools import product
import time

DIMENSIONS = [3, 4, 8]

# Known small-rank decompositions
def strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Rank 7."""
    # Returns U (4,7), V (4,7), W (4,7)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors: for each product, which A entries
    # A index: i*m+j = i*2+j
    # M1: A11+A22 -> U[0]=1, U[3]=1
    # M2: A21+A22 -> U[1]=1, U[3]=1
    # M3: A11 -> U[0]=1
    # M4: A22 -> U[3]=1
    # M5: A11+A12 -> U[0]=1, U[1]=1
    # M6: A21-A11 -> U[0]=-1, U[2]=1
    # M7: A12-A22 -> U[1]=1, U[3]=-1
    
    U[0, 0] = 1  # M1
    U[3, 0] = 1  # M1
    U[1, 1] = 1  # M2
    U[3, 1] = 1  # M2
    U[0, 2] = 1  # M3
    U[3, 3] = 1  # M4
    U[0, 4] = 1  # M5
    U[1, 4] = 1  # M5
    U[0, 5] = -1  # M6
    U[2, 5] = 1  # M6
    U[1, 6] = 1  # M7
    U[3, 6] = -1  # M7
    
    # V vectors: for each product, which B entries
    # B index: j*p+k = j*2+k
    # M1: B11+B22 -> V[0]=1, V[3]=1
    # M2: B11 -> V[0]=1
    # M3: B12-B22 -> V[1]=1, V[3]=-1
    # M4: B21-B11 -> V[2]=1, V[0]=-1
    # M5: B22 -> V[3]=1
    # M6: B11+B12 -> V[0]=1, V[1]=1
    # M7: B21+B22 -> V[2]=1, V[3]=1
    
    V[0, 0] = 1
    V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1
    V[3, 2] = -1
    V[2, 3] = 1
    V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1
    V[1, 5] = 1
    V[2, 6] = 1
    V[3, 6] = 1
    
    # W vectors: for each product, which C entries
    # C index: k*n+i = k*2+i
    # C11: M1 + M4 - M5 + M7 -> W[0]: +1 (M1), +1 (M4), -1 (M5), +1 (M7)
    # C12: M3 + M5 -> W[1]: +1 (M3), +1 (M5)
    # C21: M2 + M4 -> W[2]: +1 (M2), +1 (M4)
    # C22: M1 - M2 + M3 + M6 -> W[3]: +1 (M1), -1 (M2), +1 (M3), +1 (M6)
    
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # M5
    W[0, 6] = 1   # M7
    
    W[1, 2] = 1   # M3
    W[1, 4] = 1   # M5
    
    W[2, 1] = 1   # M2
    W[2, 3] = 1   # M4
    
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return U, V, W

def schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition. Rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def tensor_product(U1, V1, W1, U2, V2, W2):
    """Tensor product of two decompositions.
    If (U1,V1,W1) decomposes n1*m1*p1 and (U2,V2,W2) decomposes n2*m2*p2,
    then the tensor product decomposes (n1*n2)*(m1*m2)*(p1*p2).
    """
    n1, m1, p1 = U1.shape[0] // (V1.shape[0] // W1.shape[0] // n1), V1.shape[0] // W1.shape[0], W1.shape[0] // n1
    # Actually, let's compute dimensions properly
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # From U1: n1*m1 = U1.shape[0], from V1: m1*p1 = V1.shape[0]
    # So m1 = gcd or we can solve: n1*m1 = U1.shape[0], m1*p1 = V1.shape[0]
    # We need to pass dimensions explicitly
    
    # Let me redo this function to take dimensions
    pass

def tensor_product_with_dims(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Tensor product of two decompositions with explicit dimensions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U index: i*m+j where i = i1*n2+i2, j = j1*m2+j2
            # U[i1*n2+i2, j1*m2+j2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1 * m1 + j1, c1]
                    if u1 != 0:
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2 = U2[i2 * m2 + j2, c2]
                                if u2 != 0:
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    U[i * m + j, c] += u1 * u2
            
            # V index: j*p+k where j = j1*m2+j2, k = k1*p2+k2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V1[j1 * p1 + k1, c1]
                    if v1 != 0:
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2 = V2[j2 * p2 + k2, c2]
                                if v2 != 0:
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    V[j * p + k, c] += v1 * v2
            
            # W index: k*n+i where k = k1*p2+k2, i = i1*n2+i2
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W1[k1 * n1 + i1, c1]
                    if w1 != 0:
                        for k2 in range(p2):
                            for i2 in range(n2):
                                w2 = W2[k2 * n2 + i2, c2]
                                if w2 != 0:
                                    k = k1 * p2 + k2
                                    i = i1 * n2 + i2
                                    W[k * n + i, c] += w1 * w2
    
    return U, V, W

def add_decompositions(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (block sum)."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    U = np.zeros((U1.shape[0], r))
    V = np.zeros((V1.shape[0], r))
    W = np.zeros((W1.shape[0], r))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W

def strassen_34x48():
    """
    Decompose 3x4 * 4x8 using structure.
    Split: 3x4 = 2x4 + 1x4, 4x8 = 4x4 + 4x4
    Use Strassen for 2x4 * 4x4 = (2x2 * 2x2) tensor (1x2 * 2x2)
    And schoolbook for 1x4 * 4x4.
    """
    # 2x4 * 4x4: split as (2x2) tensor (1x2) for A, (2x2) tensor (2x1) for B? 
    # Actually: A is 2x4, B is 4x4.
    # View A as 2 blocks of 1x2? No.
    # A: 2 rows, 4 cols. B: 4 rows, 4 cols.
    # Split A into two 1x4? No, that doesn't help with Strassen.
    # Better: A is 2x4, B is 4x4.
    # Write A = [A1; A2] where A1, A2 are 1x4.
    # Write B = [B1; B2] where B1, B2 are 2x4? No, B is 4x4.
    # B = [B11 B12; B21 B22] where each is 2x2.
    # This doesn't factor nicely for tensor product.
    
    # Alternative: 3x4 * 4x8
    # Split 3 = 2+1, 4 = 4, 8 = 4+4
    # C = [A1; A2] * [B1 | B2] where A1 is 2x4, A2 is 1x4, B1 is 4x4, B2 is 4x4
    # C = [A1*B1 + A1*B2 | ...] no, C is 3x8.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split i into i1 in {0,1} and i2={0,1} (3 = 2+1, but not 2*2)
    # Let's use: 3x4 * 4x8
    # Block: A = [A_top (2x4); A_bottom (1x4)]
    # B = [B_left (4x4) | B_right (4x4)]
    # C_top_left = A_top * B_left (2x4)
    # C_top_right = A_top * B_right (2x4)
    # C_bot_left = A_bottom * B_left (1x4)
    # C_bot_right = A_bottom * B_right (1x4)
    
    # For A_top (2x4) * B_left (4x4):
    # Split A_top into two 1x4? No.
    # A_top is 2x4, B_left is 4x4.
    # We can write this as: A_top = [A_top1 A_top2] where A_top1, A_top2 are 2x2.
    # B_left = [B_left1; B_left2] where B_left1, B_left2 are 2x4? No.
    # B_left is 4x4. Split into 2x2 blocks: B_left = [[B11, B12], [B21, B22]]
    # A_top = [A11, A12] (1x2 of 2x2? No, A_top is 2x4 = [A11 A12] where A11, A12 are 2x2)
    # Then A_top * B_left = [A11 B11 + A12 B21, A11 B12 + A12 B22]
    # This is like multiplying [A11 A12] (2x4) by [[B11, B12], [B21, B22]] (4x4)
    # = A11*[B11; B21] + A12*[B12; B22]
    # Each term is 2x2 * 4x2 = not standard.
    
    # Let me try a different split.
    # 3x4 * 4x8
    # Use the fact that 4 = 2*2, 8 = 2*4
    # A: 3x4, B: 4x8
    # Split A as 3x(2*2), B as (2*2)x(2*4)
    # This is getting complicated. Let me just use schoolbook with some Strassen for sub-blocks.
    
    # Simplest approach that improves over 96:
    # 3x4 * 4x8
    # Split: A = [A1 (2x4); A2 (1x4)], B = [B1 (4x4) | B2 (4x4)]
    # C11 = A1*B1 (2x4 * 4x4 = 2x4)
    # C12 = A1*B2 (2x4 * 4x4 = 2x4)
    # C21 = A2*B1 (1x4 * 4x4 = 1x4)
    # C22 = A2*B2 (1x4 * 4x4 = 1x4)
    
    # For C11 = A1*B1: A1 is 2x4, B1 is 4x4.
    # Further split: A1 = [A1a A1b] (2x2 each), B1 = [[B1a, B1b], [B1c, B1d]] (2x2 each)
    # C11 = A1a * [B1a; B1c] + A1b * [B1b; B1d]
    # A1a * [B1a; B1c]: 2x2 * 4x2. This is not a standard matmul.
    
    # Let me try yet another approach.
    # 3x4 * 4x8.
    # Note that 4*4*4 = 64, and we can use Strassen for 2x2x2 blocks.
    # 3x4 * 4x8: 
    # Split 3 = 2+1, 4 = 2+2, 8 = 4+4
    # This gives 8 sub-products of various sizes.
    
    # Actually, let me just construct a decomposition using:
    # - Strassen for the 2x2x2 core (7 terms)
    # - Extend to cover all dimensions
    
    # 3x4 * 4x8
    # i in 0..2, j in 0..3, k in 0..7
    # 
    # Split: i = i1 + 2*i2 where i1 in 0..1, i2 in 0..1 (but 3 = 2*1 + 1)
    # Let i1 in 0..1, i2 in 0..1, but we need 3 values.
    # i = 0,1,2. Map to (i1, i2): (0,0), (1,0), (0,1)? Not clean.
    
    # Let me use a direct approach with Strassen on sub-blocks.
    # 3x4 * 4x8
    # = [A0 A1 A2] * B where A0,A1,A2 are 1x4, B is 4x8
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # For the 2x2x2 Strassen trick, we need to group indices.
    # Let's group: i into {0,1} and {2}, j into {0,1} and {2,3}, k into {0..3} and {4..7}
    
    # This is getting complex. Let me use a simpler construction:
    # Use schoolbook for all terms, but combine where possible using Strassen.
    
    # For 2x2x2 sub-problems within the 3x4x8:
    # The 3x4x8 tensor contains multiple 2x2x2 sub-tensors.
    # Specifically, for i in {0,1}, j in {0,1}, k in {0,1}: one 2x2x2 block.
    # For i in {0,1}, j in {0,1}, k in {2,3}: another, etc.
    
    # There are: ceil(3/2) * ceil(4/2) * ceil(8/2) = 2 * 2 * 4 = 16 potential 2x2x2 blocks
    # But they overlap, so we can't just sum Strassen decompositions.
    
    # Let me use a different strategy: explicit rank construction.
    # The rank of 3x4x8 matmul is at most 96 (schoolbook).
    # Known lower bound: at least 3*4*8/sqrt(3*4*8) or similar, but let's aim for < 96.
    
    # Use the following decomposition:
    # Split 3x4 * 4x8 into:
    # 1) 2x4 * 4x8 using a rank-reduced method
    # 2) 1x4 * 4x8 using schoolbook (rank 32)
    # 
    # For 2x4 * 4x8:
    # Split as (2x2 * 2x8) + (0x2 * 2x8)? No.
    # A is 2x4, B is 4x8.
    # A = [A11 A12] (2x2 each), B = [[B11, B12], [B21, B22]] (2x4 each)
    # C = A11*[B11; B21] + A12*[B12; B22]
    # A11*[B11; B21]: 2x2 * 4x4. This is a 2x4 matrix.
    # We can use Strassen for 2x2*2x2 on sub-blocks.
    # [B11; B21] is 4x4 = [[B11, B12], [B21, B22]]... wait no.
    # B is 4x8. B = [B11 B12; B21 B22] where B11 is 2x4, B12 is 2x4, etc.
    # So [B11; B21] is 4x4 (stacking B11 and B21 vertically).
    # A11 is 2x2. A11 * [B11; B21] is 2x4.
    # = [A11[0,:] * B11 + A11[1,:] * B21, ...] 
    # This is a 2x4 result.
    # Each entry: C[i,k] = A11[i,0]*B11[0,k] + A11[i,1]*B21[1,k]
    # This is a 2x1 * 4x4? No.
    
    # I think the cleanest approach for a guaranteed improvement is:
    # Use Strassen decomposition for all 2x2x2 aligned sub-blocks, and schoolbook for the rest.
    # But alignment is tricky with 3x4x8.
    
    # Let me just use a verified construction:
    # Rank 88 decomposition using the structure 3=2+1, 4=2+2, 8=4+4.
    
    # For the 2x2x4 part (i in 0,1; j in 0,1,2,3; k in 0,1,2,3):
    # Split j into j1 in 0,1 and j2 in 0,1 (j = j1 + 2*j2)
    # Split k into k1 in 0,1 and k2 in 0,1 (k = k1 + 2*k2)
    # For fixed j2, k2: we have 2x2x2 sub-problem (i in 0,1; j1 in 0,1; k1 in 0,1)
    # There are 2*2 = 4 such sub-problems.
    # Each can be done with Strassen rank 7. Total: 4*7 = 28 for the 2x4x4 part.
    
    # For the 1x4x4 part (i=2; j in 0..3; k in 0..3):
    # Schoolbook: 4*4 = 16 terms.
    
    # For the 2x4x4 part with k in 4..7:
    # Similarly, 4 sub-problems of 2x2x2, each rank 7. Total: 28.
    
    # For the 1x4x4 part with k in 4..7:
    # Schoolbook: 16 terms.
    
    # Total: 28 + 16 + 28 + 16 = 88.
    
    # Let me construct this.
    U = []
    V = []
    W = []
    
    # Part 1: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # j = j1 + 2*j2, k = k1 + 2*k2, j1,j2,k1,k2 in {0,1}
    for j2 in range(2):
        for k2 in range(2):
            U_sub, V_sub, W_sub = strassen_222()
            # U_sub has shape (4, 7): rows are i*m+j = i*2+j1 for i in 0,1, j1 in 0,1
            # But we need U rows for i in 0,1; j in 0,1,2,3 (full 4)
            # U_sub row index: i*2 + j1. We need to map to i*4 + j = i*4 + j1 + 2*j2
            U_full = np.zeros((12, 7))  # 3*4 = 12
            V_full = np.zeros((32, 7))  # 4*8 = 32
            W_full = np.zeros((24, 7))  # 8*3 = 24
            
            for i in range(2):
                for j1 in range(2):
                    row_sub = i * 2 + j1
                    row_full = i * 4 + j1 + 2 * j2
                    U_full[row_full, :] = U_sub[row_sub, :]
            
            for j1 in range(2):
                for k1 in range(2):
                    row_sub = j1 * 2 + k1
                    row_full = (j1 + 2 * j2) * 8 + k1 + 4 * k2
                    V_full[row_full, :] = V_sub[row_sub, :]
            
            for k1 in range(2):
                for i in range(2):
                    row_sub = k1 * 2 + i
                    row_full = (k1 + 4 * k2) * 3 + i
                    W_full[row_full, :] = W_sub[row_sub, :]
            
            U.append(U_full)
            V.append(V_full)
            W.append(W_full)
    
    # Part 2: i=2, j in {0,1,2,3}, k in {0,1,2,3}
    # Schoolbook: for each j, k: U[2*4+j, c]=1, V[j*8+k, c]=1, W[k*3+2, c]=1
    for j in range(4):
        for k in range(4):
            U_term = np.zeros((12, 1))
            V_term = np.zeros((32, 1))
            W_term = np.zeros((24, 1))
            U_term[2 * 4 + j, 0] = 1
            V_term[j * 8 + k, 0] = 1
            W_term[k * 3 + 2, 0] = 1
            U.append(U_term)
            V.append(V_term)
            W.append(W_term)
    
    # Part 3: i in {0,1}, j in {0,1,2,3}, k in {4,5,6,7}
    # k = 4 + k2 where k2 in 0,1,2,3? No, k in 4..7.
    # Let k = 4 + k_local where k_local in 0..3.
    # Split j = j1 + 2*j2, k_local = k1 + 2*k2
    for j2 in range(2):
        for k2 in range(2):
            U_sub, V_sub, W_sub = strassen_222()
            U_full = np.zeros((12, 7))
            V_full = np.zeros((32, 7))
            W_full = np.zeros((24, 7))
            
            for i in range(2):
                for j1 in range(2):
                    row_sub = i * 2 + j1
                    row_full = i * 4 + j1 + 2 * j2
                    U_full[row_full, :] = U_sub[row_sub, :]
            
            for j1 in range(2):
                for k1 in range(2):
                    row_sub = j1 * 2 + k1
                    row_full = (j1 + 2 * j2) * 8 + (4 + k1 + 2 * k2)
                    V_full[row_full, :] = V_sub[row_sub, :]
            
            for k1 in range(2):
                for i in range(2):
                    row_sub = k1 * 2 + i
                    row_full = (4 + k1 + 2 * k2) * 3 + i
                    W_full[row_full, :] = W_sub[row_sub, :]
            
            U.append(U_full)
            V.append(V_full)
            W.append(W_full)
    
    # Part 4: i=2, j in {0,1,2,3}, k in {4,5,6,7}
    for j in range(4):
        for k in range(4):
            U_term = np.zeros((12, 1))
            V_term = np.zeros((32, 1))
            W_term = np.zeros((24, 1))
            U_term[2 * 4 + j, 0] = 1
            V_term[j * 8 + (4 + k), 0] = 1
            W_term[(4 + k) * 3 + 2, 0] = 1
            U.append(U_term)
            V.append(V_term)
            W.append(W_term)
    
    # Concatenate all parts
    U_final = np.hstack(U)
    V_final = np.hstack(V)
    W_final = np.hstack(W)
    
    return U_final, V_final, W_final

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline
    
    # Try the rank 88 construction first
    U, V, W = strassen_34x48()
    
    # Verify the decomposition
    n, m, p = 3, 4, 8
    rank = U.shape[1]
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if j==j else 0
    # Actually, the multiplication tensor T[i,j,k] = 1 if always, and we need
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Let's verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                error += abs(val - 1.0)
    
    if error < 1e-9:
        # Convert to lists of ints/floats
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Round to nearest integer or half-integer
        def round_half(x):
            return round(x * 2) / 2
        
        U_list = [[round_half(v) for v in row] for row in U_list]
        V_list = [[round_half(v) for v in row] for row in V_list]
        W_list = [[round_half(v) for v in row] for row in W_list]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_s, V_s, W_s = schoolbook_nm_p(n, m, p)
    U_list = U_s.tolist()
    V_list = V_s.tolist()
    W_list = W_s.tolist()
    
    def round_half(x):
        return round(x * 2) / 2
    
    U_list = [[round_half(v) for v in row] for row in U_list]
    V_list = [[round_half(v) for v in row] for row in V_list]
    W_list = [[round_half(v) for v in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
