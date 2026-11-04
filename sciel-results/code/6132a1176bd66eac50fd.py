"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Uses block-wise Strassen: partition 6x6 into 3x3 blocks of 2x2.
    The 3x3 block multiplication uses schoolbook (27 terms), each 2x2 block
    multiplication uses Strassen (rank 7). Total rank = 27 * 7 = 189.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # all 6
    
    # Build rank-7 Strassen decomposition for 2x2x2
    def build_2x2x2_strassen():
        """
        Build rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        
        Indexing convention:
        U[i*m+j, r] for i,j in {0,1} -> U has shape (4, 7)
        V[j*p+k, r] for j,k in {0,1} -> V has shape (4, 7)  
        W[k*n+i, r] for k,i in {0,1} -> W has shape (4, 7)
        
        Strassen's 7 products:
        m1 = (A00+A01)*(B00+B01)
        m2 = (A10+A11)*B00
        m3 = A00*(B01-B11)
        m4 = A11*(B10-B00)
        m5 = (A00+A11)*(B00+B11)
        m6 = (A01-A11)*(B10+B11)
        m7 = (A00-A10)*(B01+B10)
        
        Output:
        C00 = m1 + m4 - m5 + m7
        C01 = m3 + m5
        C10 = m2 + m5
        C11 = m1 - m2 + m4 + m6
        """
        r2 = 7
        U2 = np.zeros((4, r2))
        V2 = np.zeros((4, r2))
        W2 = np.zeros((4, r2))
        
        # U indexing: U[i*2+j, r]
        # V indexing: V[j*2+k, r]
        # W indexing: W[k*2+i, r]
        
        # r=0: m1 = (A00+A01)*(B00+B01)
        # U: A00=1, A01=1 -> U[0,0]=1, U[1,0]=1
        # V: B00=1, B01=1 -> V[0,0]=1, V[1,0]=1
        # W: C00=+1, C01=0, C10=0, C11=+1 -> W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=1
        U2[0,0] = 1; U2[1,0] = 1
        V2[0,0] = 1; V2[1,0] = 1
        W2[0,0] = 1; W2[1,0] = 0; W2[2,0] = 0; W2[3,0] = 1
        
        # r=1: m2 = (A10+A11)*B00
        # U: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
        # V: B00=1 -> V[0,1]=1
        # W: C00=0, C01=0, C10=+1, C11=-1 -> W[0,1]=0, W[1,1]=0, W[2,1]=1, W[3,1]=-1
        U2[2,1] = 1; U2[3,1] = 1
        V2[0,1] = 1
        W2[0,1] = 0; W2[1,1] = 0; W2[2,1] = 1; W2[3,1] = -1
        
        # r=2: m3 = A00*(B01-B11)
        # U: A00=1 -> U[0,2]=1
        # V: B01=1, B11=-1 -> V[1,2]=1, V[3,2]=-1
        # W: C01=+1 -> W[1,2]=1
        U2[0,2] = 1
        V2[1,2] = 1; V2[3,2] = -1
        W2[1,2] = 1
        
        # r=3: m4 = A11*(B10-B00)
        # U: A11=1 -> U[3,3]=1
        # V: B10=1, B00=-1 -> V[2,3]=1, V[0,3]=-1
        # W: C00=+1, C11=+1 -> W[0,3]=1, W[3,3]=1
        U2[3,3] = 1
        V2[0,3] = -1; V2[2,3] = 1
        W2[0,3] = 1; W2[3,3] = 1
        
        # r=4: m5 = (A00+A11)*(B00+B11)
        # U: A00=1, A11=1 -> U[0,4]=1, U[3,4]=1
        # V: B00=1, B11=1 -> V[0,4]=1, V[3,4]=1
        # W: C00=-1, C01=+1, C10=+1, C11=-1 -> W[0,4]=-1, W[1,4]=1, W[2,4]=1, W[3,4]=-1
        U2[0,4] = 1; U2[3,4] = 1
        V2[0,4] = 1; V2[3,4] = 1
        W2[0,4] = -1; W2[1,4] = 1; W2[2,4] = 1; W2[3,4] = -1
        
        # r=5: m6 = (A01-A11)*(B10+B11)
        # U: A01=1, A11=-1 -> U[1,5]=1, U[3,5]=-1
        # V: B10=1, B11=1 -> V[2,5]=1, V[3,5]=1
        # W: C11=+1 -> W[3,5]=1
        U2[1,5] = 1; U2[3,5] = -1
        V2[2,5] = 1; V2[3,5] = 1
        W2[3,5] = 1
        
        # r=6: m7 = (A00-A10)*(B01+B10)
        # U: A00=1, A10=-1 -> U[0,6]=1, U[2,6]=-1
        # V: B01=1, B10=1 -> V[1,6]=1, V[2,6]=1
        # W: C00=+1 -> W[0,6]=1
        U2[0,6] = 1; U2[2,6] = -1
        V2[1,6] = 1; V2[2,6] = 1
        W2[0,6] = 1
        
        return U2, V2, W2, r2
    
    # Build schoolbook rank-27 decomposition for 3x3x3
    def build_3x3x3_schoolbook():
        n3, m3, p3 = 3, 3, 3
        r3 = 27
        U3 = np.zeros((n3 * m3, r3))
        V3 = np.zeros((m3 * p3, r3))
        W3 = np.zeros((p3 * n3, r3))
        
        col = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U3[i * m3 + j, col] = 1
                    V3[j * p3 + k, col] = 1
                    W3[k * n3 + i, col] = 1
                    col += 1
        
        return U3, V3, W3, r3
    
    # Combine using Kronecker product structure
    def combine_blockwise(U3, V3, W3, r3, U2, V2, W2, r2):
        """
        Combine 3x3x3 (block level) with 2x2x2 (within block) decompositions.
        
        6x6 matrix partitioned as 3x3 blocks of 2x2.
        U6 has shape (36, rank) where rank = r3 * r2 = 189
        U6[(i3*2+i2)*6 + (j3*2+j2), r3_idx*r2 + r2_idx] =
            U3[i3*3+j3, r3_idx] * U2[i2*2+j2, r2_idx]
        
        Similarly for V and W.
        """
        n, m, p = 6, 6, 6
        rank = r3 * r2
        
        U6 = np.zeros((n * m, rank))
        V6 = np.zeros((m * p, rank))
        W6 = np.zeros((p * n, rank))
        
        # Build U6
        for i3 in range(3):
            for j3 in range(3):
                for r3_idx in range(r3):
                    val3 = U3[i3 * 3 + j3, r3_idx]
                    if val3 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            for r2_idx in range(r2):
                                val2 = U2[i2 * 2 + j2, r2_idx]
                                if val2 == 0:
                                    continue
                                row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                                col = r3_idx * r2 + r2_idx
                                U6[row, col] = val3 * val2
        
        # Build V6
        for j3 in range(3):
            for k3 in range(3):
                for r3_idx in range(r3):
                    val3 = V3[j3 * 3 + k3, r3_idx]
                    if val3 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            for r2_idx in range(r2):
                                val2 = V2[j2 * 2 + k2, r2_idx]
                                if val2 == 0:
                                    continue
                                row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                                col = r3_idx * r2 + r2_idx
                                V6[row, col] = val3 * val2
        
        # Build W6
        for k3 in range(3):
            for i3 in range(3):
                for r3_idx in range(r3):
                    val3 = W3[k3 * 3 + i3, r3_idx]
                    if val3 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            for r2_idx in range(r2):
                                val2 = W2[k2 * 2 + i2, r2_idx]
                                if val2 == 0:
                                    continue
                                row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                                col = r3_idx * r2 + r2_idx
                                W6[row, col] = val3 * val2
        
        return U6, V6, W6, rank
    
    # Build components
    U2, V2, W2, r2 = build_2x2x2_strassen()
    U3, V3, W3, r3 = build_3x3x3_schoolbook()
    
    # Verify Strassen decomposition
    def verify_decomposition(U, V, W, n, m, p, rank):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify 2x2x2 Strassen
    if not verify_decomposition(U2, V2, W2, 2, 2, 2, r2):
        # Fall back to schoolbook 2x2x2
        r2 = 8
        U2 = np.zeros((4, r2))
        V2 = np.zeros((4, r2))
        W2 = np.zeros((4, r2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = i * 4 + j * 2 + k
                    U2[i*2+j, col] = 1
                    V2[j*2+k, col] = 1
                    W2[k*2+i, col] = 1
    
    # Combine
    U6, V6, W6, rank = combine_blockwise(U3, V3, W3, r3, U2, V2, W2, r2)
    
    # Verify full 6x6x6 decomposition
    if not verify_decomposition(U6, V6, W6, 6, 6, 6, rank):
        # Fall back to schoolbook 6x6x6
        rank = 216
        U6 = np.zeros((36, rank))
        V6 = np.zeros((36, rank))
        W6 = np.zeros((36, rank))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U6[i*6+j, col] = 1
                    V6[j*6+k, col] = 1
                    W6[k*6+i, col] = 1
                    col += 1
    
    # Convert to lists of lists (integer/half-integer entries)
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    # Ensure entries are integers or half-integers (they should be integers here)
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = int(round(mat[i][j])) if abs(mat[i][j] - round(mat[i][j])) < 1e-10 else mat[i][j]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
