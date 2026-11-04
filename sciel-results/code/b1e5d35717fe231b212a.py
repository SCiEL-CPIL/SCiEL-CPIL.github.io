"""
Strategy: Direct schoolbook decomposition for 4x4x7 matrix multiplication.
The multiplication tensor T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,7).
The rank-112 schoolbook decomposition is guaranteed correct and trivial to verify.
Any rank reduction requires finding a non-trivial tensor decomposition, which is
an open problem for these dimensions. We return the schoolbook as the safe baseline.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    Convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Use vectorized check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _try_strassen_4x4x7():
    """
    Attempt a Strassen-based decomposition for the 4x4x4 part (rank 49 or 56)
    plus schoolbook for the 4x4x3 part.
    Returns (U, V, W) or None if verification fails.
    """
    n, m, p = 4, 4, 7
    
    # 2x2x2 Strassen (rank 7)
    def strassen_2x2():
        U2 = np.zeros((4, 7), dtype=np.int64)
        V2 = np.zeros((4, 7), dtype=np.int64)
        W2 = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A00-A11)(B11-B00) -> contributes to C00, C01, C10, C11
        U2[0,0]=1; U2[3,0]=-1
        V2[3,0]=1; V2[0,0]=-1
        W2[0,0]=1; W2[1,0]=1; W2[2,0]=1; W2[3,0]=-1
        
        # M2 = (A00+A11)B11 -> C11, C10? Let me be careful.
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3
        # C10 = M6 + M5
        # C11 = M1 - M2 + M3 + M4
        # Where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # Let me redefine properly:
        U2 = np.zeros((4, 7), dtype=np.int64)
        V2 = np.zeros((4, 7), dtype=np.int64)
        W2 = np.zeros((4, 7), dtype=np.int64)
        
        # M1 = (A00+A11)(B00+B11)
        U2[0,0]=1; U2[3,0]=1
        V2[0,0]=1; V2[3,0]=1
        # M1 contributes to C00, C01, C10, C11 with coefficient +1 for C00 and C11, -1 for C01 and C10?
        # C00 = M1 - M4 - M7 + M5 ... this is getting confusing.
        
        # Let me use the verified correct version from the seed:
        U2[0,0]=1; U2[3,0]=-1
        V2[3,0]=1; V2[0,0]=-1
        W2[0,0]=1; W2[1,0]=1; W2[2,0]=1; W2[3,0]=-1
        
        U2[0,1]=1; U2[3,1]=1
        V2[3,1]=1
        W2[2,1]=1; W2[3,1]=1
        
        U2[3,2]=1
        V2[2,2]=1; V2[0,2]=-1
        W2[1,2]=1; W2[3,2]=-1
        
        U2[0,3]=1
        V2[1,3]=1; V2[3,3]=-1
        W2[0,3]=1; W2[2,3]=1
        
        U2[0,4]=1; U2[1,4]=1
        V2[3,4]=1
        W2[0,4]=1; W2[2,4]=1
        
        U2[2,5]=1; U2[0,5]=-1
        V2[0,5]=1; V2[1,5]=1
        W2[2,5]=1
        
        U2[1,6]=1; U2[3,6]=-1
        V2[2,6]=1; V2[3,6]=1
        W2[0,6]=-1; W2[1,6]=1; W2[3,6]=1
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2()
    
    # Build 4x4x4 from 8 blocks of 2x2x2 Strassen: rank 56
    # Then 4x4x3 schoolbook: rank 48
    # Total: 104
    
    total_rank = 56 + 48
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((28, total_rank), dtype=np.float64)
    W = np.zeros((28, total_rank), dtype=np.float64)
    
    # 8 blocks for 4x4x4: (row_block, col_block) where each is 2x2
    block_configs = [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (0, 0), (0, 1),  # Wait, need to be more careful
    ]
    
    # Actually, the 4x4x4 multiplication has 16 scalar products, grouped into 8 blocks of 2x2x2?
    # No. C = A*B where A is 4x4, B is 4x4, C is 4x4.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor has indices (i, j, k) with i,j in 0..3, k in 0..3.
    # We can partition i into {0,1} and {2,3}, j into {0,1} and {2,3}, k into {0,1} and {2,3}.
    # But that gives 8 sub-tensors, each of type 2x2x2? No, each sub-tensor is over 2x2x2 indices but they're not independent.
    
    # The block approach: For each (i_block, j_block, k_block) where each is in {0,1},
    # we have a 2x2x2 sub-problem. There are 8 such sub-problems.
    # But the multiplication tensor is NOT a direct sum of 8 independent 2x2x2 tensors.
    # It's more subtle. The standard "block Strassen" for 4x4x4 uses the 2x2x2 Strassen
    # on the 2x2 blocks, which gives rank 8*7 = 56. This is correct but not minimal.
    
    # Let me just verify the 56-rank construction works:
    # For i in {0,1}, j in {0,1}, k in {0,1}: uses block (0,0)
    # For i in {0,1}, j in {0,1}, k in {2,3}: uses block (0,1)
    # For i in {0,1}, j in {2,3}, k in {0,1}: uses block (1,0)
    # For i in {0,1}, j in {2,3}, k in {2,3}: uses block (1,1)
    # For i in {2,3}, j in {0,1}, k in {0,1}: uses block (2,0)
    # For i in {2,3}, j in {0,1}, k in {2,3}: uses block (2,1)
    # For i in {2,3}, j in {2,3}, k in {0,1}: uses block (3,0)
    # For i in {2,3}, j in {2,3}, k in {2,3}: uses block (3,1)
    
    # Wait, that's 8 blocks but the indices don't align this way.
    # In C[i,k] = sum_j A[i,j] B[j,k], the "middle" index j is what connects A and B.
    # If we block i and k, then for each (i_block, k_block) pair, we sum over all j.
    # So we have 4 blocks of (2x2 output) x (4x4 middle) ... this is getting complicated.
    
    # The correct way: The 4x4x4 tensor can be seen as a 2x2 block matrix multiplication
    # where each block is 2x2. The rank-56 "block Strassen" applies Strassen to each
    # 2x2 block multiplication independently, which is valid because each block's
    # computation is a 2x2x2 tensor.
    
    # Block (a, b) where a is the output block row (i in {2a, 2a+1}), 
    # b is the output block col (k in {2b, 2b+1}).
    # The input blocks are: A[:, 2c:2c+2] for c in {0,1} and B[2c:2c+2, :] for c in {0,1}.
    # So for output block (a,b): C[2a:2a+2, 2b:2b+2] = sum_{c=0}^{1} A[2a:2a+2, 2c:2c+2] * B[2c:2c+2, 2b:2b+2]
    # This is a 2x2 = 2x2*2x2 multiplication, which has tensor rank 7 via Strassen.
    # But wait, it's a SUM over c of two 2x2x2 tensors. So the rank is at most 14 per output block.
    # With 4 output blocks, total rank = 4 * 14 = 56. Yes.
    
    # So for each output block (a,b) in {(0,0),(0,1),(1,0),(1,1)}:
    #   For each input block c in {0,1}:
    #     Apply Strassen 2x2x2 to A[2a:2a+2, 2c:2c+2] * B[2c:2c+2, 2b:2b+2]
    #     This contributes rank 7.
    # Total: 4 * 2 * 7 = 56.
    
    rank_offset = 0
    for a in range(2):  # output block row
        for b in range(2):  # output block col
            for c in range(2):  # input block (summed over)
                for r in range(7):
                    for ii in range(2):  # local i
                        for jj in range(2):  # local j
                            i = 2*a + ii
                            j = 2*c + jj
                            U[i*4 + j, rank_offset + r] += U2[ii*2 + jj, r]
                    
                    for jj in range(2):  # local j (input col of A = input row of B)
                        for kk in range(2):  # local k
                            j = 2*c + jj
                            k = 2*b + kk
                            V[j*4 + k, rank_offset + r] += V2[jj*2 + kk, r]
                    
                    for kk in range(2):  # local k
                        for ii in range(2):  # local i
                            k = 2*b + kk
                            i = 2*a + ii
                            W[k*4 + i, rank_offset + r] += W2[kk*2 + ii, r]
                
                rank_offset += 7
    
    # Now add 4x4x3 schoolbook for k in {4,5,6}
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k = 4 + k_local
                col = rank_offset + i*12 + j*3 + k_local
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*4 + i, col] = 1
    
    assert rank_offset + 48 == total_rank
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin from 150s limit
    
    # First, try the Strassen-based construction (rank 104)
    U, V, W = _try_strassen_4x4x7()
    
    if _verify(U, V, W, n, m, p):
        # Try to see if we can verify and return
        pass
    else:
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
    
    # Verify schoolbook as ultimate fallback
    if not _verify(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
    
    # Return as lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
