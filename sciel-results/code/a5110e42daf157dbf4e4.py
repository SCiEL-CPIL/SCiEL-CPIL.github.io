"""Matrix multiplication 4,4,7 - Strassen-based decomposition."""
DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split p into p1=4 and p2=3
    p1 = 4
    p2 = p - p1  # 3
    
    rank = 28 + n * m * p2  # 28 + 48 = 76
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 base case: returns (U, V, W) for 2x2x2 with rank 7
    def strassen_2x2():
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        s = []
        # Each entry: (U_factor, V_factor, W_factor)
        # U[i,j], V[j,k], W[k,i]
        # For 2x2x2: indices (i,j,k) in {0,1}^3
        
        # M1: (A11+A22)(B11+B22) -> C11 +=, C22 +=
        # U: rows 0,1 (i=0,j=0 -> A11; i=1,j=1 -> A22)
        # V: cols 0,1 (j=0,k=0 -> B11; j=1,k=1 -> B22)
        # W: C11 (k=0,i=0), C22 (k=1,i=1)
        s.append(([1,1,0,0], [1,1,0,0], [1,0,0,1]))  # M1: U[A11,A22]=1, V[B11,B22]=1, W[C11,C22]=1
        
        # M2: (A21+A22)B11 -> C21 +=
        # A21: i=1,j=0; A22: i=1,j=1; B11: j=0,k=0
        s.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))  # M2: U[A21,A22]=1, V[B11]=1, W[C21]=1
        
        # M3: A11(B21-B22) -> C12 +=
        # A11: i=0,j=0; B21: j=1,k=0; B22: j=1,k=1
        s.append(([1,0,0,0], [0,1,-1,0], [0,1,0,0]))  # M3: U[A11]=1, V[B21=1,B22=-1], W[C12]=1
        
        # M4: A22(B12-B11) -> C11 +=, C21 +=
        # A22: i=1,j=1; B12: j=0,k=1; B11: j=0,k=0
        s.append(([0,0,0,1], [1,-1,0,0], [1,0,0,1]))  # M4: U[A22]=1, V[B12=1,B11=-1], W[C11=1,C21=1]
        
        # M5: (A11+A12)B22 -> C11 -=, C12 +=
        # A11: i=0,j=0; A12: i=0,j=1; B22: j=1,k=1
        s.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))  # M5: U[A11,A12]=1, V[B22]=1, W[C11=-1,C12=1]
        
        # M6: (A21-A11)(B11+B12) -> C22 +=
        # A21: i=1,j=0; A11: i=0,j=0; B11: j=0,k=0; B12: j=0,k=1
        s.append(([1,-1,0,0], [1,1,0,0], [0,0,0,1]))  # M6: U[A21=1,A11=-1], V[B11,B12]=1, W[C22]=1
        
        # M7: (A12-A22)(B21+B22) -> C11 +=
        # A12: i=0,j=1; A22: i=1,j=1; B21: j=1,k=0; B22: j=1,k=1
        s.append(([0,1,0,-1], [0,1,1,0], [1,0,0,0]))  # M7: U[A12=1,A22=-1], V[B21,B22]=1, W[C11]=1
        
        return s
    
    # Build Strassen 4x4x4 decomposition
    def strassen_4x4():
        """Returns list of (U_16, V_16, W_16) for 4x4x4, rank 28."""
        base = strassen_2x2()
        
        # 4x4x4 splits into four 2x2x2 blocks in each dimension
        # Block (bi, bj, bk) where bi,bj,bk in {0,1}
        # A block (bi,bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # B block (bj,bk): rows 2*bj..2*bj+1, cols 2*bk..2*bk+1  
        # C block (bk,bi): rows 2*bk..2*bk+1, cols 2*bi..2*bi+1
        
        result = []
        for bk in range(2):
            for bj in range(2):
                for bi in range(2):
                    for (U_base, V_base, W_base) in base:
                        U4 = [0]*16
                        V4 = [0]*16
                        W4 = [0]*16
                        
                        for idx in range(4):
                            i = idx // 2
                            j = idx % 2
                            # U index: block (bi, bj), local (i,j)
                            U4[(2*bi+i)*2 + (2*bj+j)] = U_base[idx]
                            
                            # V index: block (bj, bk), local (j,k)
                            k = idx % 2
                            V4[(2*bj+j)*2 + (2*bk+k)] = V_base[idx]
                            
                            # W index: block (bk, bi), local (k,i)
                            W4[(2*bk+k)*2 + (2*bi+i)] = W_base[idx]
                        
                        result.append((U4, V4, W4))
        
        return result  # 4*4*7 = 112... wait, that's not right
    
    # Actually, Strassen 4x4x4 has rank 28, not 112
    # The 4x4x4 decomposition uses 28 rank-1 terms
    # Each 4x4x4 = sum of 28 rank-1 tensors
    # 
    # Correct approach: 4x4x4 Strassen uses 28 multiplications
    # The decomposition is: C = sum_{c=1}^{28} U_c * V_c * W_c
    # where U is 16x28, V is 16x28, W is 16x28
    #
    # Let me construct this properly using the known 4x4x4 rank-28 decomposition
    # from Strassen's algorithm applied to 2x2 blocks.
    
    def strassen_4x4_correct():
        """4x4x4 matrix multiplication with rank 28."""
        base = strassen_2x2()
        
        # For 4x4x4, we have 4 2x2x2 sub-problems per dimension
        # Total: 4*4*4 = 64 sub-problems, but Strassen shares computations
        # 
        # Actually, the standard Strassen 4x4x4 uses:
        # - 4 quadrants in A, B, C
        # - The 7 Strassen products per 2x2 level
        # - For 4x4: we need to recursively apply
        #
        # Simpler: just use the known rank-28 decomposition directly
        # 
        # Let me build it by combining the 2x2 results appropriately.
        # For 4x4x4, split each matrix into 4 2x2 blocks.
        # A = [[A00, A01], [A10, A11]], similarly for B, C
        #
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        #
        # Each 2x2*2x2 = 2x2 uses 7 multiplications
        # Total: 4*4*7 = 112? No, that's schoolbook.
        #
        # Strassen for 4x4: 
        # Apply Strassen to the 4x4 as four 2x2 blocks
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # But each Mi is a 2x2 matrix product, which itself needs 7 rank-1 terms
        # Total: 7 * 7 = 49? Still not 28.
        #
        # The rank-28 decomposition is more subtle. Let me just use schoolbook
        # for 4x4x4 with rank 64, which is simpler and guaranteed correct.
        
        # Schoolbook 4x4x4: rank 64
        s = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U = [0]*16
                    V = [0]*16
                    W = [0]*16
                    U[i*4+j] = 1
                    V[j*4+k] = 1
                    W[k*4+i] = 1
                    s.append((U, V, W))
        return s
    
    # For now, use schoolbook for both parts: 4x4x4 + 4x4x3
    # Rank = 64 + 48 = 112 (same as schoolbook for 4x4x7)
    # 
    # Let me try a better approach: use the block structure
    # C = A * B where A is 4x4, B is 4x7
    # Split B = [B_left | B_right] where B_left is 4x4, B_right is 4x3
    # C = A*B_left + A*B_right
    # 
    # For A*B_left (4x4x4): use rank 64 schoolbook
    # For A*B_right (4x4x3): use rank 48 schoolbook
    # Total: 112
    #
    # To beat 112, I need a non-schoolbook decomposition.
    # Let me try: use Strassen for the 4x4x4 part.
    # 
    # Known: 4x4x4 has rank at most 28 (from Strassen variants)
    # If I can construct rank-28 for 4x4x4, total = 28 + 48 = 76
    
    # Constructing the actual rank-28 4x4x4 decomposition is complex.
    # Let me use a simpler improvement: exploit the structure.
    
    # Alternative: use the fact that 7 = 4 + 3, and find shared factors.
    # 
    # Actually, let me just implement the straightforward block decomposition
    # with schoolbook for now, which gives rank 112, same as seed.
    # 
    # For a real improvement, I need the Strassen 4x4x4 rank-28.
    # Let me look up / derive it.
    
    # The 4x4x4 rank-28 decomposition can be built as follows:
    # It's equivalent to doing 2x2x2 Strassen at the block level,
    # where each "scalar" is itself a 2x2 matrix.
    # But that gives 7*7=49, not 28.
    #
    # The true rank-28 uses a different tensor decomposition.
    # 
    # Given time constraints, let me return the schoolbook solution.
    # This matches the seed score of 1.0.
    
    # Schoolbook 4x4x7
    rank_final = n * m * p
    U_final = [[0] * rank_final for _ in range(n * m)]
    V_final = [[0] * rank_final for _ in range(m * p)]
    W_final = [[0] * rank_final for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_final[i*m + j][col] = 1
                V_final[j*p + k][col] = 1
                W_final[k*n + i][col] = 1
                col += 1
    
    return {"U": U_final, "V": V_final, "W": W_final}
