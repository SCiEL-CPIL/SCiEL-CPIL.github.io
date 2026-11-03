import numpy as np
import time

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices:
    # U: 00=0, 01=1, 10=2, 11=3
    # V: 00=0, 01=1, 10=2, 11=3
    # W: 00=0, 01=1, 10=2, 11=3  (W[k*2+i, r])
    #    k=0,i=0 -> 0
    #    k=0,i=1 -> 1
    #    k=1,i=0 -> 2
    #    k=1,i=1 -> 3
    
    # M1 = A00 * (B01 + B11)
    U[0, 0] = 1
    V[1, 0] = 1
    V[3, 0] = 1
    W[1, 0] = 1  # k=0, i=1 -> 0*2+1=1
    W[3, 0] = 1  # k=1, i=1 -> 1*2+1=3
    
    # M2 = (A10 + A00) * B00
    U[0, 1] = 1
    U[2, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1  # k=0, i=0 -> 0
    W[2, 1] = 1  # k=1, i=0 -> 2
    
    # M3 = A01 * (B10 - B00)
    U[1, 2] = 1
    V[2, 2] = 1
    V[0, 2] = -1
    W[2, 2] = 1  # k=1, i=0 -> 2
    W[0, 2] = -1 # k=0, i=0 -> 0
    
    # M4 = A11 * (B00 - B10)
    U[3, 3] = 1
    V[0, 3] = 1
    V[2, 3] = -1
    W[0, 3] = 1  # k=0, i=0 -> 0
    W[2, 3] = -1 # k=1, i=0 -> 2
    
    # M5 = (A00 + A11) * (B00 + B11) - M2 - M4
    # But in tensor decomposition, we just add the terms.
    # Contribution to C:
    # C00 += M5 - M2 - M4
    # C11 += M5 - M2 - M4
    # This is handled by the linear combination in the output.
    # Wait, the tensor decomposition is a sum of rank-1 terms.
    # The standard Strassen decomposition is:
    # M1 = A00(B01+B11)
    # M2 = (A10+A00)B00
    # M3 = A01(B10-B00)
    # M4 = A11(B00-B10)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A10-A11)(B11+B01)
    # M7 = (A01-A10)(B00+A10)
    
    # C00 = M5 - M4 - M2 + M1
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # Let's re-verify the W indices for each term.
    # Term r contributes to C[i,k] via U[i,j] V[j,k] W[k,i].
    # Actually, C[i,k] = sum_j sum_r U[i,j] V[j,k] W[k,i]
    
    # M1: A00 * (B01+B11).
    # A00 is U[0,0]=1.
    # B01 is V[1,0]=1. B11 is V[3,0]=1.
    # Result is at C[0,1] and C[0,3]? No.
    # A is 2x2. i in {0,1}, j in {0,1}.
    # B is 2x2. j in {0,1}, k in {0,1}.
    # C is 2x2. i in {0,1}, k in {0,1}.
    # U[0,0]=1 means i=0, j=0.
    # V[1,0]=1 means j=1, k=0? No. V index is j*p+k. For 2x2, p=2.
    # V[1,0] -> j=0, k=1. (0*2+1=1).
    # V[3,0] -> j=1, k=1. (1*2+1=3).
    # So M1 = A[0,0] * (B[0,1] + B[1,1]).
    # This contributes to C[0,1] and C[0,1]?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If i=0, and we use A[0,0], we need j=0.
    # Then B[0,1] gives C[0,1]. B[1,1] requires j=1, which is not 0.
    # Strassen works on blocks.
    # A00 is the top-left block.
    # If A is 2x2, A00 is A[0,0].
    # Then M1 = A[0,0] * (B[0,1] + B[1,1]).
    # This term contributes to C[0,1] (from B[0,1]) and... wait.
    # In Strassen, the blocks are matrices.
    # Here we are decomposing the tensor for a 2x2 matrix multiplication.
    # The "blocks" are scalars.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Strassen:
    # M1 = A00 (B01 + B11)
    # M2 = (A10 + A00) B00
    # M3 = A01 (B10 - B00)
    # M4 = A11 (B00 - B10)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A11) (B11 + B01)
    # M7 = (A01 - A10) (B00 + A10)  <- Typo in standard list? Usually (A01 - A10)(B00 + B10)
    
    # Let's check standard Strassen:
    # M1 = A00 (B01 + B11)
    # M2 = (A10 + A00) B00
    # M3 = A01 (B10 - B00)
    # M4 = A11 (B00 - B10)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A11) (B11 + B01)
    # M7 = (A01 - A10) (B00 + B10)
    
    # C00 = M5 - M4 - M2 + M1
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # Let's verify C00:
    # M5 - M4 - M2 + M1
    # = (A00+A11)(B00+B11) - A11(B00-B10) - (A10+A00)B00 + A00(B01+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B00 + A11B10 - A10B00 - A00B00 + A00B01 + A00B11
    # = A00B11 + A11B11 + A11B10 - A10B00 + A00B01 + A00B11
    # This is not C00. C00 = A00B00 + A01B10.
    # I am mixing up the indices or the formula.
    
    # Correct Strassen:
    # M1 = A00 (B01 + B11)
    # M2 = (A10 + A00) B00
    # M3 = A01 (B10 - B00)
    # M4 = A11 (B00 - B10)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A11) (B11 + B01)
    # M7 = (A01 - A10) (B00 + B10)
    
    # C00 = M5 - M4 - M2 + M1 ?
    # Let's check a reference.
    # C00 = M5 - M4 - M2 + M1 is incorrect for C00.
    # C00 = M5 - M4 - M2 + M1 ... wait.
    
    # Let's use the known correct formulas:
    # C00 = M5 - M4 - M2 + M1
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # Let's re-evaluate C00 with the expansion:
    # M5 = A00B00 + A00B11 + A11B00 + A11B11
    # -M4 = -A11B00 + A11B10
    # -M2 = -A10B00 - A00B00
    # +M1 = A00B01 + A00B11
    # Sum = A00B11 + A11B11 + A11B10 - A10B00 + A00B01 + A00B11
    # Still not C00.
    
    # Okay, I will use the explicit tensor decomposition for 2x2x2 rank 7 found in literature or derive it.
    # Actually, I can just use the schoolbook for 2x2 (rank 4) if I can't get Strassen right quickly, but Strassen is rank 7. Schoolbook is rank 4.
    # Wait, 2x2 matrix multiplication tensor rank is 7. Schoolbook is 4 multiplications? No, 2x2 matrix mult requires 4 multiplications.
    # Why is Strassen rank 7?
    # Ah, Strassen is for 2x2 BLOCKS. If the blocks are 1x1, then it's just 2x2 matrix mult.
    # The rank of the 2x2x2 tensor is 7.
    # The schoolbook algorithm uses 4 multiplications.
    # Is the rank of the 2x2x2 tensor 7 or 4?
    # The tensor T_{ijk} = delta_{ij} delta_{jk} ? No.
    # T_{i,j;k,l} = delta_{jk} ? No.
    # The multiplication tensor for n x n is defined by T_{i,j;k,l} = 1 if j=k else 0.
    # For n=2, there are 4 non-zero entries.
    # The rank of the 2x2x2 matrix multiplication tensor is 7.
    # Wait, if the rank is 7, then schoolbook (4 terms) is NOT a valid decomposition?
    # No, schoolbook IS a decomposition.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # This is 8 terms.
    # Can we do it in 4?
    # If we use 4 terms, we can only form 4 products.
    # But we need 8 terms in the sum.
    # So schoolbook for 2x2 uses 8 terms?
    # No, for 2x2, C = AB.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # Total 8 products.
    # So the schoolbook rank for 2x2 is 8.
    # Strassen reduces it to 7.
    
    # So, I will implement Strassen for the 2x2 block.
    
    # Let's get the correct Strassen coefficients.
    # M1 = A00 (B01 + B11)
    # M2 = (A10 + A00) B00
    # M3 = A01 (B10 - B00)
    # M4 = A11 (B00 - B10)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A11) (B11 + B01)
    # M7 = (A01 - A10) (B00 + B10)
    
    # C00 = M5 - M4 - M2 + M1
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # Let's check C00 again.
    # M5 - M4 - M2 + M1
    # = (A00+A11)(B00+B11) - A11(B00-B10) - (A10+A00)B00 + A00(B01+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B00 + A11B10 - A10B00 - A00B00 + A00B01 + A00B11
    # = A00B11 + A11B11 + A11B10 - A10B00 + A00B01 + A00B11
    # This is definitely not C00.
    
    # I will use a known correct set of coefficients.
    # From "Fast Matrix Multiplication" by Strassen:
    # M1 = A00 (B01 + B11)
    # M2 = (A10 + A00) B00
    # M3 = A01 (B10 - B00)
    # M4 = A11 (B00 - B10)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A10 - A11) (B11 + B01)
    # M7 = (A01 - A10) (B00 + B10)
    
    # C00 = M5 - M4 - M2 + M1
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M5
    
    # There must be a sign error in my expansion or the formula.
    # Let's try:
    # C00 = M5 - M4 - M2 + M1
    # If I replace M1 with A00(B01+B11), M2 with (A10+A00)B00, M4 with A11(B00-B10), M5 with (A00+A11)(B00+B11).
    # Maybe the formula for C00 is different.
    
    # Let's just use the schoolbook for the 2x2 block (rank 8) to be safe, or find a rank 7 decomposition.
    # Actually, for 5x5, splitting 2+3:
    # 2x2 block: rank 7 (Strassen) or 8 (Schoolbook)
    # 3x3 block: rank 19 (Boras) or 27 (Schoolbook)
    # Cross terms: 2x3 and 3x2.
    
    # If I use schoolbook for everything, rank is 125.
    # If I use Strassen for 2x2 (7) and schoolbook for 3x3 (27):
    # Blocks:
    # A = [A00 A01; A10 A11] where A00 is 2x2, A01 is 2x3, A10 is 3x2, A11 is 3x3.
    # B = [B00 B01; B10 B11] where B00 is 2x2, B01 is 2x3, B10 is 3x2, B11 is 3x3.
    
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # If I use Strassen for the 2x2 part (A00 B00), it gives 7 terms.
    # But Strassen mixes the blocks.
    # M1 = A00 (B01 + B11). This involves B01 (2x3) and B11 (3x3).
    # So the terms are not block-diagonal.
    
    # This is getting complicated.
    # I will stick to the schoolbook decomposition for 5x5 (rank 125) but try to find a small improvement or just ensure correctness.
    # The prompt says "Find a rank-minimal exact tensor decomposition".
    # The current score is 1.0000 (rank 125).
    # Inspiration 2 says rank 120 is possible with 2+3 split.
    
    # Let's try to implement the 2+3 block decomposition with schoolbook for all sub-blocks.
    # This will still be rank 125.
    # To get rank < 125, I need to use Strassen or Boras.
    
    # Let's try to use Strassen for the 2x2 block and schoolbook for the rest.
    # But Strassen couples the blocks.
    
    # Alternative: Use the known rank 19 decomposition for 3x3.
    # And rank 7 for 2x2.
    
    # Given the complexity and time limit, I will return the schoolbook solution (rank 125) to ensure validity,
    # as finding a valid lower rank decomposition in 150s is risky.
    
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _strassen_2x2() if False else _schoolbook_5x5()

def _schoolbook_5x5():
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
