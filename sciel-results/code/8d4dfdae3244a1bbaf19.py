import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 120.0  # safety margin before 150s kill

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
        
        # Split A(2x3) into A0(2x2) [cols 0,1] and A1(2x1) [col 2]
        # Split B(3x2) into B0(3x2) [rows 0,1] and B1(3x1) [row 2]
        
        # Part 1: A0(2x2) * B0(2x2) = C00(2x2) -- use Strassen rank-7
        # Part 2: A0(2x2) * B1(2x1) = C01(2x1) -- 4 rank-1 terms
        # Part 3: A1(2x1) * B0(2x2) = C10(2x2) -- 4 rank-1 terms
        # Wait, that's 7+4+4=15, too many.
        
        # Better: A(2x3) * B(3x2)
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        # Use Strassen on the "2x2 core" formed by A[:,0:2] and B[0:2,:]:
        # C_core[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] (rank 7 via Strassen)
        # C_extra[i,k] = A[i,2]*B[2,k] (rank 4: one rank-1 for each (i,k))
        # Total: 7 + 4 = 11 per block. 
        
        # Strassen for 2x2 * 2x2:
        # A_s = [[A[i,0] for i in {0,1}], [A[i,1] for i in {0,1}]] (2x2)
        # B_s = [[B[0,k] for k in {0,1}], [B[1,k] for k in {0,1}]] (2x2)
        # C_s[i,k] = A_s[i,0]*B_s[0,k] + A_s[i,1]*B_s[1,k]
        
        # Strassen 7 products for 2x2*2x2:
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B01-B11)
        # m4 = A11(B10-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # In our indexing for block blk:
        # A[i,0] -> U index i*3+0, A[i,1] -> U index i*3+1, A[i,2] -> U index i*3+2
        # B[0,k] -> V index 0*p + k = k, B[1,k] -> V index 1*p+k = p+k, B[2,k] -> V index 2*p+k = 2p+k
        # C[i,k] -> W index k*n + i = 2k+i (for block blk: k is 2*blk or 2*blk+1)
        # Actually W[k*n+i] where k is the global column index.
        # For block blk, output columns are 2*blk and 2*blk+1.
        # W index for C[i, 2*blk] = (2*blk)*n + i = 4*blk + i
        # W index for C[i, 2*blk+1] = (2*blk+1)*n + i = 4*blk + 2 + i
        
        # Strassen part (indices 0-6 in block):
        # m1 = (A00+A11)(B00+B11)
        U[0*3+0, base+0] = 1   # A[0,0]
        U[1*3+1, base+0] = 1   # A[1,1]
        V[0*p + 2*blk, base+0] = 1     # B[0, 2*blk]
        V[1*p + (2*blk+1), base+0] = 1 # B[1, 2*blk+1]
        W[4*blk + 0, base+0] = 1       # C[0, 2*blk]
        W[4*blk + 3, base+0] = 1       # C[1, 2*blk+1]
        
        # m2 = (A10+A11)B00
        U[1*3+0, base+1] = 1   # A[1,0]
        U[1*3+1, base+1] = 1   # A[1,1]
        V[0*p + 2*blk, base+1] = 1     # B[0, 2*blk]
        W[4*blk + 1, base+1] = 1       # C[1, 2*blk]
        W[4*blk + 3, base+1] = -1      # C[1, 2*blk+1]
        
        # m3 = A00(B01-B11)
        U[0*3+0, base+2] = 1   # A[0,0]
        V[0*p + (2*blk+1), base+2] = 1   # B[0, 2*blk+1]
        V[1*p + (2*blk+1), base+2] = -1  # B[1, 2*blk+1]
        W[4*blk + 2, base+2] = 1         # C[0, 2*blk+1]
        W[4*blk + 3, base+2] = 1         # C[1, 2*blk+1]
        
        # m4 = A11(B10-B00)
        U[1*3+1, base+3] = 1   # A[1,1]
        V[1*p + 2*blk, base+3] = 1     # B[1, 2*blk]
        V[0*p + 2*blk, base+3] = -1    # B[0, 2*blk]
        W[4*blk + 0, base+3] = 1       # C[0, 2*blk]
        W[4*blk + 1, base+3] = 1       # C[1, 2*blk]
        
        # m5 = (A00+A01)B11
        U[0*3+0, base+4] = 1   # A[0,0]
        U[0*3+1, base+4] = 1   # A[0,1]
        V[1*p + (2*blk+1), base+4] = 1  # B[1, 2*blk+1]
        W[4*blk + 0, base+4] = -1       # C[0, 2*blk]
        W[4*blk + 2, base+4] = 1        # C[0, 2*blk+1]
        
        # m6 = (A10-A00)(B00+B01)
        U[1*3+0, base+5] = 1   # A[1,0]
        U[0*3+0, base+5] = -1  # A[0,0]
        V[0*p + 2*blk, base+5] = 1        # B[0, 2*blk]
        V[0*p + (2*blk+1), base+5] = 1    # B[0, 2*blk+1]
        W[4*blk + 3, base+5] = 1          # C[1, 2*blk+1]
        
        # m7 = (A01-A11)(B10+B11)
        U[0*3+1, base+6] = 1   # A[0,1]
        U[1*3+1, base+6] = -1  # A[1,1]
        V[1*p + 2*blk, base+6] = 1        # B[1, 2*blk]
        V[1*p + (2*blk+1), base+6] = 1    # B[1, 2*blk+1]
        W[4*blk + 0, base+6] = 1          # C[0, 2*blk]
        
        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # 4 rank-1 terms (indices 7-10 in block):
        # C[0, 2*blk] += A[0,2]*B[2, 2*blk]
        U[0*3+2, base+7] = 1
        V[2*p + 2*blk, base+7] = 1
        W[4*blk + 0, base+7] = 1
        
        # C[0, 2*blk+1] += A[0,2]*B[2, 2*blk+1]
        U[0*3+2, base+8] = 1
        V[2*p + (2*blk+1), base+8] = 1
        W[4*blk + 2, base+8] = 1
        
        # C[1, 2*blk] += A[1,2]*B[2, 2*blk]
        U[1*3+2, base+9] = 1
        V[2*p + 2*blk, base+9] = 1
        W[4*blk + 1, base+9] = 1
        
        # C[1, 2*blk+1] += A[1,2]*B[2, 2*blk+1]
        U[1*3+2, base+10] = 1
        V[2*p + (2*blk+1), base+10] = 1
        W[4*blk + 3, base+10] = 1

    # Time guard: return immediately since construction is deterministic and fast
    if time.time() > _deadline - 5.0:
        # Should never hit here for this deterministic construction, but safety net
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
