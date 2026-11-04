"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_7():
    """Return (U, V, W) for the rank-7 Strassen 2x2x2 decomposition.
    U has shape (4,7): rows (i,j) in {00,01,10,11}
    V has shape (4,7): rows (j,k) in {00,01,10,11}
    W has shape (4,7): rows (k,i) in {00,01,10,11}
    Convention: A[i,j], B[j,k], C[i,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    """
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = A11*(B10-B00)
    # M3 = (A00-A11)*(B11+B10)
    # M4 = (A01+A11)*B00
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B01+B11)
    # M7 = (A01-A10)*(B00+B10)
    #
    # C00 = M1 + M2 - M3 + M5
    # C01 = M4 + M3
    # C10 = M6 + M1
    # C11 = M1 - M2 + M4 + M7
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # Row ordering: (i,j): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # V row: (j,k): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # W row: (k,i): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    
    # M1: (A00+A11)*(B00+B11)
    U[0,0]=1; U[3,0]=1  # A00, A11
    V[0,0]=1; V[3,0]=1  # B00, B11
    W[0,0]=1; W[1,1]=1; W[2,2]=1; W[3,3]=-1  # C00+=, C01+=, C10+=, C11-=
    
    # M2: A11*(B10-B00)
    U[3,1]=1  # A11
    V[2,1]=1; V[0,1]=-1  # B10, -B00
    W[0,0]=-1; W[3,3]=1  # C00-=, C11+=
    
    # M3: (A00-A11)*(B11+B10)
    U[0,2]=1; U[3,2]=-1  # A00, -A11
    V[3,2]=1; V[2,2]=1  # B11, B10
    W[0,0]=-1; W[1,1]=1  # C00-=, C01+=
    
    # M4: (A01+A11)*B00
    U[1,3]=1; U[3,3]=1  # A01, A11
    V[0,3]=1  # B00
    W[1,1]=1; W[3,3]=1  # C01+=, C11+=
    
    # M5: (A00+A10)*B11
    U[0,4]=1; U[2,4]=1  # A00, A10
    V[3,4]=1  # B11
    W[0,0]=1  # C00+=
    
    # M6: (A10-A00)*(B01+B11)
    U[2,5]=1; U[0,5]=-1  # A10, -A00
    V[1,5]=1; V[3,5]=1  # B01, B11
    W[2,2]=1  # C10+=
    
    # M7: (A01-A10)*(B00+B10)
    U[1,6]=1; U[2,6]=-1  # A01, -A10
    V[0,6]=1; V[2,6]=1  # B00, B10
    W[3,3]=1  # C11+=
    
    return U, V, W

def construct():
    """Construct rank-98 decomposition for 4x4 x 4x8 multiplication.
    
    Strategy: Decompose 4x4x4 multiplication using Kronecker product of two
    Strassen rank-7 decompositions (giving rank 49). Then for the 4x8 output,
    the multiplication is C = A * B where A is 4x4 and B is 4x8. We can view
    B as two 4x4 blocks: B = [B1 | B2]. Then C = A*B1 | A*B2. We use the
    same rank-49 decomposition for each block, giving total rank 98.
    """
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_7()
    
    # Build 4x4x4 decomposition via Kronecker product (rank 49)
    # For 4x4: A is 4x4, B is 4x4, C is 4x4
    # U4 has shape (16, 49): rows indexed by (i1,i2,j1,j2)
    # V4 has shape (16, 49): rows indexed by (j1,j2,k1,k2)
    # W4 has shape (16, 49): rows indexed by (k1,k2,i1,i2)
    
    # Kronecker: each term (t1,t2) in the 49 terms
    # U4[(i1*4+i2)*4 + (j1*4+j2), t1*7+t2] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    # V4[(j1*4+j2)*4 + (k1*4+k2), t1*7+t2] = V2[j1*2+k1, t1] * V2[j2*2+k2, t2]
    # W4[(k1*4+k2)*4 + (i1*4+i2), t1*7+t2] = W2[k1*2+i1, t1] * W2[k2*2+i2, t2]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k1 in range(2):
                                for k2 in range(2):
                                    # U4 index: (i1,i2,j1,j2) -> ((i1*2+i2)*4 + (j1*2+j2))
                                    u4_idx = (i1*2+i2)*4 + (j1*2+j2)
                                    U4[u4_idx, col] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
                                    # V4 index: (j1,j2,k1,k2) -> ((j1*2+j2)*4 + (k1*2+k2))
                                    v4_idx = (j1*2+j2)*4 + (k1*2+k2)
                                    V4[v4_idx, col] = V2[j1*2+k1, t1] * V2[j2*2+k2, t2]
                                    # W4 index: (k1,k2,i1,i2) -> ((k1*2+k2)*4 + (i1*2+i2))
                                    w4_idx = (k1*2+k2)*4 + (i1*2+i2)
                                    W4[w4_idx, col] = W2[k1*2+i1, t1] * W2[k2*2+i2, t2]
    
    # Now for 4x4 x 4x8:
    # A is 4x4 (16 elements), B is 4x8 (32 elements), C is 8x4 (32 elements)
    # B = [B1 | B2] where B1, B2 are 4x4
    # C = [C1 | C2] where C1, C2 are 4x4
    # C1 = A * B1, C2 = A * B2
    # 
    # U has shape (16, rank): rows indexed by (i,j) for A[i,j]
    # V has shape (32, rank): rows indexed by (j,k) for B[j,k]
    # W has shape (32, rank): rows indexed by (k,i) for C[k,i]
    
    # For B1 (columns 0-3): B[j, k] with k in 0..3 -> V index j*4+k
    # For B2 (columns 4-7): B[j, k] with k in 4..7 -> V index j*4+(k-4)
    # 
    # Using the same rank-49 decomposition for both blocks:
    # Block 1 (B1, C1): use columns 0..48 of rank-49
    # Block 2 (B2, C2): use columns 49..97 of rank-49
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # Block 1: B1 = B[:, 0:4], C1 = C[0:4, :]
    U[:, 0:49] = U4
    # V for B1: V[j*4 + k, col] = V4[j*4+k, col] for col in 0..48
    V[0:16, 0:49] = V4
    # W for C1: W[k*4 + i, col] = W4[k*4+i, col] for col in 0..48
    W[0:16, 0:49] = W4
    
    # Block 2: B2 = B[:, 4:8], C2 = C[4:8, :]
    U[:, 49:98] = U4
    # V for B2: V[j*4 + (k+4), col+49] = V4[j*4+k, col]
    V[0:16, 49:98] = V4
    # W for C2: W[(k+4)*4 + i, col+49] = W4[k*4+i, col]
    W[16:32, 49:98] = W4
    
    # Convert to list of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
