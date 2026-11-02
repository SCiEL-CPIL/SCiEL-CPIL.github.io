import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    # Decompose into 4 blocks of M_{2,3,2}
    # Each block: A(2x3) * B(3x2) = C(2x2)
    # Rank per block: 11 (Strassen-7 on 2x2x2 sub-block + 4 rank-1 for extra column)
    # Total rank: 4 * 11 = 44
    
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        # For block blk, B columns are 2*blk and 2*blk+1
        # C columns (output) are 2*blk and 2*blk+1
        
        # A(i,j): i in {0,1}, j in {0,1,2} -> U index i*3+j
        # B(j,k): j in {0,1,2}, k in {0..7} -> V index j*p+k
        # C(i,k): i in {0,1}, k in {0..7} -> W index k*n+i
        
        # Strassen part for the 2x2 core (A[:,0:2] * B[0:2,:]):
        # m1 = (A[0,0]+A[1,1])(B[0,k0]+B[1,k1])
        U[0*3+0, base+0] = 1   # A[0,0]
        U[1*3+1, base+0] = 1   # A[1,1]
        V[0*p + 2*blk, base+0] = 1     # B[0, 2*blk]
        V[1*p + (2*blk+1), base+0] = 1 # B[1, 2*blk+1]
        W[(2*blk)*n + 0, base+0] = 1       # C[0, 2*blk]
        W[(2*blk+1)*n + 1, base+0] = 1     # C[1, 2*blk+1]
        
        # m2 = (A[1,0]+A[1,1]) * B[0, 2*blk]
        U[1*3+0, base+1] = 1   # A[1,0]
        U[1*3+1, base+1] = 1   # A[1,1]
        V[0*p + 2*blk, base+1] = 1     # B[0, 2*blk]
        W[(2*blk)*n + 1, base+1] = 1       # C[1, 2*blk]
        W[(2*blk+1)*n + 1, base+1] = -1    # C[1, 2*blk+1]
        
        # m3 = A[0,0] * (B[0, 2*blk+1] - B[1, 2*blk+1])
        U[0*3+0, base+2] = 1   # A[0,0]
        V[0*p + (2*blk+1), base+2] = 1   # B[0, 2*blk+1]
        V[1*p + (2*blk+1), base+2] = -1  # B[1, 2*blk+1]
        W[(2*blk+1)*n + 0, base+2] = 1   # C[0, 2*blk+1]
        W[(2*blk+1)*n + 1, base+2] = 1   # C[1, 2*blk+1]
        
        # m4 = A[1,1] * (B[1, 2*blk] - B[0, 2*blk])
        U[1*3+1, base+3] = 1   # A[1,1]
        V[1*p + 2*blk, base+3] = 1     # B[1, 2*blk]
        V[0*p + 2*blk, base+3] = -1    # B[0, 2*blk]
        W[(2*blk)*n + 0, base+3] = 1       # C[0, 2*blk]
        W[(2*blk)*n + 1, base+3] = 1       # C[1, 2*blk]
        
        # m5 = (A[0,0]+A[0,1]) * B[1, 2*blk+1]
        U[0*3+0, base+4] = 1   # A[0,0]
        U[0*3+1, base+4] = 1   # A[0,1]
        V[1*p + (2*blk+1), base+4] = 1  # B[1, 2*blk+1]
        W[(2*blk)*n + 0, base+4] = -1       # C[0, 2*blk]
        W[(2*blk+1)*n + 0, base+4] = 1      # C[0, 2*blk+1]
        
        # m6 = (A[1,0]-A[0,0]) * (B[0, 2*blk] + B[0, 2*blk+1])
        U[1*3+0, base+5] = 1   # A[1,0]
        U[0*3+0, base+5] = -1  # A[0,0]
        V[0*p + 2*blk, base+5] = 1        # B[0, 2*blk]
        V[0*p + (2*blk+1), base+5] = 1    # B[0, 2*blk+1]
        W[(2*blk+1)*n + 1, base+5] = 1    # C[1, 2*blk+1]
        
        # m7 = (A[0,1]-A[1,1]) * (B[1, 2*blk] + B[1, 2*blk+1])
        U[0*3+1, base+6] = 1   # A[0,1]
        U[1*3+1, base+6] = -1  # A[1,1]
        V[1*p + 2*blk, base+6] = 1        # B[1, 2*blk]
        V[1*p + (2*blk+1), base+6] = 1    # B[1, 2*blk+1]
        W[(2*blk)*n + 0, base+6] = 1      # C[0, 2*blk]
        
        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # 4 rank-1 terms (indices 7-10 in block):
        # C[0, 2*blk] += A[0,2]*B[2, 2*blk]
        U[0*3+2, base+7] = 1
        V[2*p + 2*blk, base+7] = 1
        W[(2*blk)*n + 0, base+7] = 1
        
        # C[0, 2*blk+1] += A[0,2]*B[2, 2*blk+1]
        U[0*3+2, base+8] = 1
        V[2*p + (2*blk+1), base+8] = 1
        W[(2*blk+1)*n + 0, base+8] = 1
        
        # C[1, 2*blk] += A[1,2]*B[2, 2*blk]
        U[1*3+2, base+9] = 1
        V[2*p + 2*blk, base+9] = 1
        W[(2*blk)*n + 1, base+9] = 1
        
        # C[1, 2*blk+1] += A[1,2]*B[2, 2*blk+1]
        U[1*3+2, base+10] = 1
        V[2*p + (2*blk+1), base+10] = 1
        W[(2*blk+1)*n + 1, base+10] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
