import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use two 4x4x4 Strassen decompositions, one for the first 4 columns of B, one for the last 4.
    # Rank of 4x4x4 Strassen is 49. Total rank = 98.
    
    # First, let's construct the 4x4x4 Strassen decomposition.
    # We can do this by using the 2x2x2 Strassen decomposition on the 2x2 blocks.
    
    # 2x2x2 Strassen decomposition (rank 7)
    # U: 2x2 x 7, V: 2x2 x 7, W: 2x2 x 7
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We need to map this to the U, V, W format.
    # U[i*2+j, c] corresponds to the coefficient of A[i,j] in the c-th product.
    # V[j*2+k, c] corresponds to the coefficient of B[j,k] in the c-th product.
    # W[k*2+i, c] corresponds to the coefficient of the c-th product in C[i,k].
    
    # Let's define the 2x2x2 Strassen U, V, W.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: A00+A11, B00+B11 -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's fill U2, V2, W2 for 2x2x2.
    # c=0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1 # This is not correct for W. W is for C[i,k].
    # W[k*2+i, c]. For 2x2, indices are 0,1,2,3.
    # C00 is W[0*2+0] = W[0]. C01 is W[1*2+0] = W[2]. C10 is W[0*2+1] = W[1]. C11 is W[1*2+1] = W[3].
    # Wait, the convention is W[k*n+i, r]. For n=2, W[k*2+i, r].
    # C[i,k] corresponds to W[k*2+i, r].
    # So C00 -> W[0], C01 -> W[2], C10 -> W[1], C11 -> W[3].
    
    # Let's restart the 2x2x2 Strassen U, V, W construction carefully.
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[2][0] = 1; W2[1][0] = 1; W2[3][0] = 1 # This is wrong. M1 contributes to all C? No.
    # M1 is part of C00, C01, C10, C11? No, Strassen formulas are:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 and C11.
    W2[0][0] = 1 # C00
    W2[3][0] = 1 # C11
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1 # C10
    W2[3][1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1 # C01
    W2[3][2] = 1 # C11
    
    # M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1 # C00
    W2[1][3] = 1 # C10
    
    # M5 = (A00+A10)B11
    U2[0][4] = 1; U2[2][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # C00
    W2[2][4] = 1 # C01
    
    # M6 = (A10-A00)(B00+B10)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1 # C11
    
    # M7 = (A01-A11)(B01+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[1][6] = 1; V2[3][6] = 1
    W2[0][6] = -1 # C00
    
    # Now we have 2x2x2 Strassen U2, V2, W2.
    # We need to construct 4x4x4 Strassen U, V, W.
    # We can do this by applying the 2x2x2 Strassen to the 2x2 blocks of the 4x4 matrices.
    # This is a standard recursive construction.
    
    # Let's define a function to build 4x4x4 from 2x2x2.
    def build_4x4x4(U2, V2, W2):
        rank2 = 7
        rank4 = 4 * rank2 # 28? No, 4x4x4 rank is 49.
        # The 4x4x4 tensor can be decomposed into 4 2x2x2 tensors? No.
        # It's more complex. The 4x4x4 Strassen decomposition is not simply 4 copies of 2x2x2.
        # It uses a different scheme.
        
        # Let's use the known 49-rank decomposition for 4x4x4.
        # It's easier to just use the schoolbook for 4x4x4 and see if we can improve.
        # Schoolbook rank for 4x4x4 is 64.
        # 2 * 64 = 128. Same as seed.
        
        # Let's try to implement the 49-rank 4x4x4 decomposition.
        # This is non-trivial. Let's use a different approach.
        
        # We can use the fact that 4x4x8 can be decomposed into 4 2x2x4 blocks?
        # Or 8 2x2x2 blocks?
        # 8 * 7 = 56. This is better than 98!
        
        # Let's verify this.
        # Split n=4 into 2+2, m=4 into 2+2, p=8 into 4+4.
        # This gives 2*2*2 = 8 blocks of size 2x2x4.
        # Each 2x2x4 block can be decomposed into 2 2x2x2 blocks.
        # So each 2x2x4 block has rank 2*7 = 14.
        # Total rank = 8 * 14 = 112. Still worse than 98.
        
        # What if we split p=8 into 2+2+2+2?
        # Then we have 2*2*4 = 16 blocks of size 2x2x2.
        # Total rank = 16 * 7 = 112. Same.
        
        # So the 2x2x2 Strassen approach gives 112.
        # The 4x4x4 Strassen approach gives 98.
        # So 98 is better.
        
        # Let's implement the 4x4x4 Strassen decomposition.
        # I will use a known 49-rank decomposition for 4x4x4.
        # This is a well-known result.
        
        # Since implementing the 49-rank 4x4x4 from scratch is complex, let's use a different strategy.
        # We can use the schoolbook for 4x4x4 (rank 64) and see if we can prune it.
        # Or we can use the 2x2x2 Strassen and accept rank 112.
        # 112 is worse than 128? No, 112 < 128. So it's an improvement.
        
        # Let's implement the 2x2x2 Strassen based 112-rank decomposition.
        # It's simpler and guaranteed to work.
        
        U4 = [[0]*112 for _ in range(16)]
        V4 = [[0]*112 for _ in range(32)]
        W4 = [[0]*112 for _ in range(32)]
        
        # Map 2x2x2 blocks to 4x4x8.
        # n=4, m=4, p=8.
        # Blocks: i in {0,1}, j in {0,1}, k in {0,1,2,3} (for p=8, split into 4 2s? No, p=8, split into 2 4s? No.)
        # Let's split p=8 into 4 blocks of size 2.
        # So we have 2 (n) * 2 (m) * 4 (p) = 16 blocks of size 2x2x2.
        # Each block uses rank 7. Total rank 112.
        
        col = 0
        for i_block in range(2): # n=4 -> 2 blocks of 2
            for j_block in range(2): # m=4 -> 2 blocks of 2
                for k_block in range(4): # p=8 -> 4 blocks of 2
                    # U2 indices: i in {0,1}, j in {0,1} -> U2[i*2+j, c]
                    # V2 indices: j in {0,1}, k in {0,1} -> V2[j*2+k, c]
                    # W2 indices: k in {0,1}, i in {0,1} -> W2[k*2+i, c]
                    
                    for c in range(7):
                        # U
                        for i in range(2):
                            for j in range(2):
                                if U2[i*2+j][c] != 0:
                                    U4[(i_block*2+i)*4 + (j_block*2+j)][col] += U2[i*2+j][c]
                        
                        # V
                        for j in range(2):
                            for k in range(2):
                                if V2[j*2+k][c] != 0:
                                    V4[(j_block*2+j)*8 + (k_block*2+k)][col] += V2[j*2+k][c]
                        
                        # W
                        for k in range(2):
                            for i in range(2):
                                if W2[k*2+i][c] != 0:
                                    W4[(k_block*2+k)*4 + i_block*2+i][col] += W2[k*2+i][c]
                    
                    col += 7
        
        return {"U": U4, "V": V4, "W": W4}

    # Call the function
    return build_4x4x4(U2, V2, W2)
# EVOLVE-BLOCK-END
