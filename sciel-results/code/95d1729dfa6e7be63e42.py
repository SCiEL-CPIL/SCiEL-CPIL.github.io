import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build a rank-r decomposition for 3x3 x 3x3 matrix multiplication
    # Using a structured approach: decompose into 2x2 Strassen + corrections
    # We'll use a known rank-23 decomposition pattern.
    
    # For 3x3 x 3x3, we use the following approach:
    # Split into 2x2 block (top-left) using Strassen (rank 7)
    # Plus remaining terms for the third row and column
    # This gives rank 7 + 16 = 23
    
    # Let me construct a valid rank-23 decomposition for 3x3x3
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = 1
    
    # Strassen for 2x2x2: rank 7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B11)  -> A22*B11... let me use standard Strassen
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B11) -- no, standard is:
    # M4 = A22(B11-B11) is wrong. Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B11) -- I'll just use the standard 7 terms
    
    # Actually, let me just use a verified approach.
    # I'll construct the full 3x3x3 decomposition explicitly.
    
    # For 3x3 x 3x3, rank 23 decomposition:
    # Use the fact that we can write it as:
    # - 2x2x2 Strassen for indices (0,1)x(0,1)x(0,1): rank 7
    # - Cross terms involving index 2: need 16 more terms
    #   For i=2, j in {0,1}, k in {0,1}: 4 terms
    #   For i in {0,1}, j=2, k in {0,1}: 4 terms  
    #   For i in {0,1}, j in {0,1}, k=2: 4 terms
    #   For i=2, j=2, k in {0,1}: 2 terms
    #   For i in {0,1}, j=2, k=2: 2 terms
    #   For i=2, j in {0,1}, k=2: 2 terms
    #   For i=2, j=2, k=2: 1 term
    # Total cross: 4+4+4+2+2+2+1 = 19... that's too many.
    
    # Let me just use a cleaner approach: schoolbook for 3x3x3 is rank 27.
    # I'll use rank 27 for 3x3x3 and rank 18 for 3x3x2, total = 2*27+18 = 72.
    # That's the same as full schoolbook. Not an improvement.
    
    # Let me try a different strategy: use rank 25 for 3x3x3.
    # A rank-25 decomposition for 3x3x3 exists.
    
    # Actually, the simplest guaranteed improvement: 
    # Use Strassen for the 2x2x2 core (rank 7) and handle the rest.
    # 3x3x3 = 2x2x2 (rank 7) + 3*2*2 + 3*2*2 + 3*3*2 + ... 
    # This gets complicated. Let me just verify the schoolbook works and 
    # try to find a small improvement.
    
    # For 3x3 x 3x2, schoolbook rank is 18. Can we do better?
    # 3x3 x 3x2: this is like multiplying by a 3x2 matrix.
    # We can think of it as two 3x3 x 3x1 multiplications.
    # 3x3 x 3x1: rank 9 (schoolbook). Two of them: rank 18.
    # Can 3x3 x 3x1 be done in less than 9? 
    # T[i,j,k] where k is always the same... actually for 3x3 x 3x1,
    # the output is 3x1, so k only takes one value.
    # T[i,j] = sum over... wait, the tensor is T[(i,j),(j,k),(i,k)]
    # For 3x3 x 3x1, k is fixed (say k=0), so T[(i,j),(j,0),(i,0)] = 1
    # This is just a 3x3 matrix-vector product, which has rank 9 (schoolbook).
    # Can it be done in less? The border rank is 9, so no improvement.
    
    # So the minimum rank for 3x3 x 3x8 is at least 2*27 + 18 = 72 if we use
    # schoolbook for each block. But we can potentially share terms across blocks.
    
    # Key insight: The two 3x3x3 blocks share the same U and V structure 
    # (just different k indices). We can potentially combine them.
    
    # Let me just return the schoolbook decomposition to ensure validity,
    # but structure it to allow for potential optimization.
    
    # Actually, let me try a rank-64 approach:
    # Use a rank-23 decomposition for each 3x3x3 block.
    # 2*23 + 18 = 64.
    
    # I need to construct a valid rank-23 decomposition for 3x3x3.
    # Let me use the following known decomposition:
    
    # The 3x3x3 matrix multiplication tensor has a rank-23 decomposition.
    # I'll construct it using Strassen for the 2x2x2 part and 
    # additional terms for the third row/column.
    
    def rank23_3x3x3():
        """Return (U, V, W) for 3x3 x 3x3 with rank 23."""
        rank = 23
        U = [[0] * rank for _ in range(9)]  # 3*3 = 9
        V = [[0] * rank for _ in range(9)]  # 3*3 = 9
        W = [[0] * rank for _ in range(9)]  # 3*3 = 9
        
        # Strassen 2x2x2 for indices i,j,k in {0,1}
        # 7 terms
        # M1 = (A11+A22)(B11+B22)
        U[0*3+0][0] = 1; U[1*3+1][0] = 1
        V[0*3+0][0] = 1; V[1*3+1][0] = 1
        W[0*3+0][0] = 1; W[1*3+1][0] = 1
        # Wait, Strassen gives C11 = M1 - M4 - M6 + M7, etc.
        # This is getting complex. Let me use a different approach.
        
        # I'll use a verified rank-23 decomposition from literature.
        # For simplicity, let me just use schoolbook (rank 27) for 3x3x3
        # and accept rank 72 total, but structure the code for potential 
        # future optimization.
        
        # Actually, let me try a simpler improvement:
        # For 3x3 x 3x8, we can group the 8 columns differently.
        # Use rank 27 for 3x3x3 and rank 18 for 3x3x2: total 72.
        # No improvement.
        
        # Let me just return schoolbook.
        return None
    
    # Given the complexity, let me just return the schoolbook decomposition
    # which is guaranteed correct. The random search in the original code
    # never finds anything, so we're stuck at rank 72.
    
    return schoolbook()
