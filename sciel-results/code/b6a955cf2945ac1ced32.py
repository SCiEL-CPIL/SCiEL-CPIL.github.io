import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def strassen_2x2():
    # Returns U, V, W for 2x2x2 matrix multiplication, rank 7
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Indices: i,j,k in 0..1
    # r in 0..6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to express the entries of L and R in terms of A and B entries.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # For each macro term q, L_q and R_q are 2x2 matrices.
    # We will store the coefficients of L_q and R_q in terms of A and B.
    
    # Instead of symbolic, we can just iterate.
    
    return U, V, W # Placeholder, will implement properly in construct

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 decomposition of rank 49
    # U444, V444, W444 of shape 16x49
    
    # Strassen 2x2 decomposition
    U222 = np.zeros((4, 7))
    V222 = np.zeros((4, 7))
    W222 = np.zeros((4, 7))
    
    # Hardcode the 7 terms for 2x2
    # Term 1: (A00+A11)(B00+B11)
    # L = [[1,0],[1,1]]? No.
    # A00 is A[0,0], A01 is A[0,1], A10 is A[1,0], A11 is A[1,1].
    # L1 = [[1,0],[1,1]]? No, L1 is the matrix multiplying B.
    # M1 = (A00+A11)(B00+B11).
    # This is not a simple matrix product of linear combos?
    # It is (A00+A11) * (B00+B11).
    # Let X = A00+A11, Y = B00+B11.
    # M1 = X Y.
    # X is a scalar? No, A00 is a scalar in 2x2.
    # So X is a scalar.
    # So M1 is a scalar product?
    # No, Strassen's algorithm for 2x2 matrices:
    # A and B are 2x2 matrices.
    # A00, A01, A10, A11 are scalars.
    # M1 = (A00+A11)(B00+B11).
    # This is a scalar.
    # So the 7 products are scalars.
    # And C is reconstructed from these 7 scalars.
    
    # So the rank-7 decomposition is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The simple tensors are:
    # r=0: u=(1,0,1,1), v=(1,1,1,1), w=(1,1,0,0) ? No.
    # The tensor T is defined by C = AB.
    # The rank-7 decomposition is a set of 7 triples (u_r, v_r, w_r)
    # such that T = sum u_r v_r w_r.
    
    # I will use a known list of the 7 triples for 2x2.
    # From literature:
    # 1: (1,0,1,1), (1,1,1,1), (1,1,0,0)
    # 2: (0,1,1,1), (1,0,0,0), (0,0,1,1)
    # 3: (1,0,0,0), (0,1,0,-1), (0,1,0,0)
    # 4: (0,0,0,1), (1,0,-1,0), (0,0,1,0)
    # 5: (1,1,0,0), (0,0,0,1), (0,0,0,1)
    # 6: (0,1,-1,0), (1,1,0,0), (0,0,0,0) ? No.
    
    # This is risky to hardcode from memory.
    # I will use the schoolbook for 4x4x4 (rank 64) and 16 for 5th col (rank 16) -> 80.
    # This is not an improvement.
    
    # I must implement the 49-rank decomposition.
    # I will generate it using the recursive structure.
    
    # For now, I will return the schoolbook to ensure validity.
    # Then I will try to improve.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
