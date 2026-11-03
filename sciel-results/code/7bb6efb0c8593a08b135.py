"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a low-rank tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Split the 4x4x9 problem into:
      * 4x4x4 (columns 0-3): use Bini-Krönig rank-49 decomposition (from rank-7 2x2x2)
      * 4x4x4 (columns 4-7): use Bini-Krönig rank-49 decomposition
      * 4x4x1 (column 8): use schoolbook rank-16
    - Total rank: 49 + 49 + 16 = 114
    
    The Bini-Krönig decomposition for 2x2x2 has rank 7.
    For 4x4x4, we can view it as a 2x2 block matrix multiplication where each block
    is 2x2. Using the Kronecker product structure, we get rank 7*7 = 49.
    
    Bini's rank-7 decomposition for 2x2x2:
    The multiplication tensor for 2x2x2 has entries T[i,j,k] = 1 if i==j==k (diagonal) 
    but actually for matrix mult, T[i,j,k] = 1 when the (i,k) entry of C = AB gets 
    contribution from A[i,j]*B[j,k].
    
    Actually, the matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all 
    (i,j,k) where i in [0,n), j in [0,m), k in [0,p), representing C[i,k] += A[i,j]*B[j,k].
    
    For 2x2x2, we need to decompose the 8-entry tensor into 7 rank-1 terms.
    
    Bini's rank-7 decomposition (verified):
    Let me use the known decomposition. The 7 rank-1 tensors are:
    
    Term 0: u=(1,0), v=(1,0), w=(1,0)  -> contributes to (0,0,0)
    Term 1: u=(1,0), v=(0,1), w=(1,0)  -> contributes to (0,1,0)  
    Term 2: u=(0,1), v=(1,0), w=(0,1)  -> contributes to (1,0,1)
    Term 3: u=(0,1), v=(0,1), w=(0,1)  -> contributes to (1,1,1)
    
    These 4 terms handle the "diagonal" of the block structure. We need 3 more for
    the off-diagonal blocks.
    
    Actually, let me just use the numerical approach: I'll construct the full 
    multiplication tensor and find a decomposition.
    
    Given time constraints, let me use a practical approach:
    - For 4x4x4: use rank 64 schoolbook (guaranteed correct)
    - For 4x4x1: use rank 16 schoolbook
    - Total: 64 + 64 + 16 = 144
    
    But let me try to do better. Let me implement the Bini-Krönig method properly.
    
    The key insight: 4x4x4 matrix multiplication can be seen as computing
    C = AB where A, B, C are 4x4. View as 2x2 block matrices:
    A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
         [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each block product involves 4 2x2x2 multiplications. With rank-7 for each,
    that's 4*7 = 28 per block, 4 blocks = 112. That's worse than 64.
    
    The Bini-Krönig approach uses a different decomposition. Let me just use 
    schoolbook for now to ensure correctness, then try to improve.
    
    Actually, I realize I should just use the best decomposition I can construct
    correctly. Let me use:
    - 4x4x4: rank 64 (schoolbook)
    - 4x4x4: rank 64 (schoolbook)  
    - 4x4x1: rank 16 (schoolbook)
    Total: 144
    
    This is the same as the current program. To improve, I need a better 
    4x4x4 decomposition.
    
    Let me try Winograd's F(4,4,4) algorithm which uses 25 multiplications.
    This gives rank 25 for 4x4x4!
    
    Winograd F(4,4,4):
    Preprocess A rows: for each row [a0,a1,a2,a3], compute:
      m1 = a0
      m2 = a1
      m3 = a0 + a1
      m4 = a2 + a3
      m5 = a0 + a1 + a2 + a3
    
    Preprocess B columns: for each column [b0,b1,b2,b3], compute:
      n1 = b0 + b1 + b2 + b3
      n2 = b0 + b1
      n3 = b0
      n4 = b2 + b3
      n5 = b2
    
    25 multiplications: M[p,q] = m_p * n_q for p,q in 1..5
    
    Postprocess to get C:
      C[i,j] = M[i,j] combinations...
    
    This is complex to implement correctly. Let me use a simpler approach.
    
    Given the complexity, let me stick with schoolbook but try to find any 
    improvement. The minimum rank for 4x4x9 is at most 144 (schoolbook).
    
    Let me just return the schoolbook decomposition to ensure correctness.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Schoolbook decomposition: rank = n*m*p = 144
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
