import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 decomposition of the 4x4 matrix multiplication tensor
    using the Strassen algorithm recursively.
    """
    n, m, p = DIMENSIONS
    
    # Recursive Strassen decomposition
    # Returns U, V, W for multiplying an n x m matrix by an m x p matrix
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    def strassen_decomp(n, m, p):
        if n == 1 and m == 1 and p == 1:
            # Rank 1
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            # Rank 7 Strassen
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Indices for A (2x2):
            # 0: a (0,0), 1: b (0,1), 2: c (1,0), 3: d (1,1)
            # Indices for B (2x2):
            # 0: e (0,0), 1: f (0,1), 2: g (1,0), 3: h (1,1)
            # Indices for C (2x2) with W[k*n+i]:
            # k=0, i=0 -> 0: c00
            # k=0, i=1 -> 1: c10 (Wait, W index is k*n+i. n=2. k=0, i=1 -> 1. This is C[1,0]?)
            # Let's check the W index convention again.
            # "W[k*n+i,r]"
            # Standard C[i,k].
            # If W index is k*n+i, then for C[0,0], k=0, i=0 -> 0.
            # For C[0,1], k=1, i=0 -> 2.
            # For C[1,0], k=0, i=1 -> 1.
            # For C[1,1], k=1, i=1 -> 3.
            # So W index 0 -> C[0,0], W index 1 -> C[1,0], W index 2 -> C[0,1], W index 3 -> C[1,1].
            # This is column-major order for C if we view i as row, k as col?
            # C[i,k]. Index = k*n + i.
            # Yes.
            
            # U indices (A): i*m+j. m=2.
            # 0: a (0,0), 1: b (0,1), 2: c (1,0), 3: d (1,1)
            
            # V indices (B): j*p+k. p=2.
            # 0: e (0,0), 1: f (0,1), 2: g (1,0), 3: h (1,1)
            
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # M1: (a+d)(e+h)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1 # C00, C11
            
            # M2: (c+d)e
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1 # C10 (index 2 is k=1, i=0? No. k=1, i=0 -> 1*2+0=2. C[0,1]?)
            # Let's re-verify W index.
            # C[i,k]. W index k*n+i.
            # C[0,0] -> k=0, i=0 -> 0.
            # C[0,1] -> k=1, i=0 -> 2.
            # C[1,0] -> k=0, i=1 -> 1.
            # C[1,1] -> k=1, i=1 -> 3.
            
            # M1 contributes to C00 (idx 0) and C11 (idx 3). Correct.
            # M2: (c+d)e. c=A[1,0], d=A[1,1], e=B[0,0].
            # Contributes to C10?
            # C10 = a*e + b*g. No.
            # C10 = A[1,0]B[0,0] + A[1,1]B[1,0] = c*e + d*g.
            # M2 = (c+d)e = c*e + d*e.
            # This doesn't look like a standard Strassen term for C10.
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Let's map these to indices.
            # A00=0, A01=1, A10=2, A11=3.
            # B00=0, B01=1, B10=2, B11=3.
            # C00=0, C01=2, C10=1, C11=3.
            
            # M1: (A00+A11)(B00+B11)
            U[0][0]=1; U[3][0]=1
            V[0][0]=1; V[3][0]=1
            W[0][0]=1; W[3][0]=1 # C00, C11
            
            # M2: (A10+A11)B00
            U[2][1]=1; U[3][1]=1
            V[0][1]=1
            W[1][1]=1 # C10
            
            # M3: A00(B01-B11)
            U[0][2]=1
            V[1][2]=1; V[3][2]=-1
            W[2][2]=1 # C01
            
            # M4: A11(B10-B00)
            U[3][3]=1
            V[2][3]=1; V[0][3]=-1
            W[0][3]=1; W[1][3]=1 # C00, C10
            
            # M5: (A00+A01)B11
            U[0][4]=1; U[1][4]=1
            V[3][4]=1
            W[0][4]=-1; W[2][4]=1 # C00, C01
            
            # M6: (A10-A00)(B00+B01)
            U[2][5]=1; U[0][5]=-1
            V[0][5]=1; V[1][5]=1
            W[3][5]=1 # C11
            
            # M7: (A01-A11)(B10+B11)
            U[1][6]=1; U[3][6]=-1
            V[2][6]=1; V[3][6]=1
            W[0][6]=1 # C00
            
            return U, V, W
            
        # Recursive step for 4x4
        # Split A, B, C into 2x2 blocks.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # We need to map the indices of the 4x4 problem to the 2x2 subproblems.
        # This is complex to do recursively with the specific flattening indices.
        # It is safer to hardcode the 49 terms or use a library.
        # Since I cannot use a library for the specific task interface, I will
        # implement the 49 terms explicitly or generate them carefully.
        
        # Given the complexity and risk of indexing errors in recursive generation
        # with specific flattening, and the fact that the "inspiration" mentioned
        # recursive generation worked, I will attempt a careful recursive implementation.
        
        # However, for 4x4, the rank is 49.
        # The decomposition is the tensor product of the 2x2 decomposition with itself?
        # No, the tensor product of two rank 7 tensors has rank 49.
        # The 4x4 multiplication tensor is isomorphic to the tensor product of two 2x2 multiplication tensors.
        # T_4 = T_2 \otimes T_2.
        # So the decomposition is the Kronecker product of the rank 7 decomposition of T_2 with itself.
        
        # Let U2, V2, W2 be the rank 7 decomposition of 2x2.
        # Then U4 = U2 \otimes U2?
        # The indices need to be mapped correctly.
        
        # Let's construct U4, V4, W4 from U2, V2, W2.
        # U2 has shape (4, 7). V2 has shape (4, 7). W2 has shape (4, 7).
        # U4 has shape (16, 49).
        # The rows of U4 correspond to (i1, i2, j1, j2) where i1,i2 in 0..1, j1,j2 in 0..1.
        # Row index in 4x4: i*m + j = i*4 + j.
        # i = i1*2 + i2? No.
        # Let's define the block structure.
        # A[i,j] where i = 2*i1 + i2, j = 2*j1 + j2.
        # So A[i,j] is the element at row i, col j.
        # In the 2x2 block decomposition, A is [[A00, A01], [A10, A11]].
        # A00 contains A[0,0], A[0,1], A[1,0], A[1,1].
        # So A[i,j] belongs to block A[i1, j1] where i1 = i//2, j1 = j//2.
        # The local index in the block is i2 = i%2, j2 = j%2.
        
        # The rank 7 decomposition of the block multiplication uses indices for the 2x2 matrix.
        # Let u2[i2, j2, r1] be the entry for block row i1, col j1?
        # No, the decomposition of the tensor product T2 \otimes T2 is:
        # Sum over r1, r2 of (u2[r1] \otimes u2[r2]) \otimes (v2[r1] \otimes v2[r2]) \otimes (w2[r1] \otimes w2[r2]).
        # The combined rank index is r = r1 * 7 + r2.
        
        # U4[i, j, r1*7+r2] = U2[i_local, j_local, r1] * U2[i_block, j_block, r2]?
        # Wait.
        # A[i,j] = A_block[i1,j1][i2,j2].
        # The factor U for the 4x4 case should be:
        # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]?
        # No.
        # The tensor T_4 is T_2 \otimes T_2.
        # The indices of T_4 are (I, J, K) where I=(i1,i2,j1,j2)?
        # Let's use the property:
        # U4[row, r1, r2] = U2[row_block, r1] * U2[row_local, r2].
        # row_block = i1*2 + j1? No.
        # In 2x2, row index is i*2+j.
        # In 4x4, row index is i*4+j.
        # i = 2*i1 + i2.
        # j = 2*j1 + j2.
        # row = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
        
        # The block index for U2 is i1*2 + j1.
        # The local index for U2 is i2*2 + j2.
        
        # So U4[row, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2].
        
        U2, V2, W2 = strassen_decomp(2, 2, 2)
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        row = (2*i1 + i2)*4 + (2*j1 + j2)
                        block_idx = i1*2 + j1
                        local_idx = i2*2 + j2
                        
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                U4[row][r] = U2[block_idx][r1] * U2[local_idx][r2]
                                
        # Similarly for V
        # V indices: j*p + k.
        # j = 2*j1 + j2.
        # k = 2*k1 + k2.
        # row = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2.
        # Block index for V2: j1*2 + k1.
        # Local index for V2: j2*2 + k2.
        
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        row = (2*j1 + j2)*4 + (2*k1 + k2)
                        block_idx = j1*2 + k1
                        local_idx = j2*2 + k2
                        
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                V4[row][r] = V2[block_idx][r1] * V2[local_idx][r2]
                                
        # Similarly for W
        # W indices: k*n + i.
        # k = 2*k1 + k2.
        # i = 2*i1 + i2.
        # row = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2.
        # Block index for W2: k1*2 + i1.
        # Local index for W2: k2*2 + i2.
        
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        row = (2*k1 + k2)*4 + (2*i1 + i2)
                        block_idx = k1*2 + i1
                        local_idx = k2*2 + i2
                        
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                W4[row][r] = W2[block_idx][r1] * W2[local_idx][r2]
                                
        return U4, V4, W4

    U, V, W = strassen_decomp(4, 4, 4)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
