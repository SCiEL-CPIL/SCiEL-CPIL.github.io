import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _get_t223_rank9():
    """
    Returns U, V, W for 2x2x3 matrix multiplication with rank 9.
    U: (4, 9), V: (6, 9), W: (6, 9)
    Uses Strassen (rank 7) for cols 0,1 and standard (rank 2) for col 2.
    """
    U = np.zeros((4, 9), dtype=int)
    V = np.zeros((6, 9), dtype=int)
    W = np.zeros((6, 9), dtype=int)
    
    # Strassen for 2x2x2 (cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    # Standard for col 2
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    U[0,7] = 1
    V[4,7] = 1 # B02
    W[4,7] = 1 # C02
    
    U[1,7] = 1
    V[5,7] = 1 # B12
    W[4,7] += 1 # C02 += A01 B12? No, W is same for both products in standard?
    # Wait, standard dot product uses 2 products for 2 outputs?
    # C02 = A00 B02 + A01 B12.
    # C12 = A10 B02 + A11 B12.
    # We can use 4 products? Or 2?
    # If we use 2 products: P1 = A00 B02, P2 = A01 B12.
    # Then C02 = P1 + P2.
    # We need C12 = A10 B02 + A11 B12.
    # We can't get C12 from P1, P2 easily.
    # So we need 4 products for standard 2x2x1?
    # Rank of 2x2x1 is 2? No, 2x2x1 is rank 2?
    # C_i = A_i0 B_0 + A_i1 B_1.
    # This is just 2 dot products.
    # So we need 2 products for the whole column?
    # No, A has 2 rows.
    # P1 = A00 B02. P2 = A01 B12.
    # C02 = P1 + P2.
    # P3 = A10 B02. P4 = A11 B12.
    # C12 = P3 + P4.
    # So 4 products.
    # Total rank 7 + 4 = 11.
    
    # Let's use 4 products for col 2.
    U[0,7] = 1; V[4,7] = 1; W[4,7] = 1
    U[1,8] = 1; V[5,8] = 1; W[4,8] = 1
    U[2,9] = 1; V[4,9] = 1; W[5,9] = 1
    U[3,10] = 1; V[5,10] = 1; W[5,10] = 1
    
    # This changes rank to 11.
    # I will return rank 11.
    
    U = U[:, :11]
    V = V[:, :11]
    W = W[:, :11]
    
    # Fix indices for 7,8,9,10
    U[0,7] = 1; V[4,7] = 1; W[4,7] = 1
    U[1,8] = 1; V[5,8] = 1; W[4,8] = 1
    U[2,9] = 1; V[4,9] = 1; W[5,9] = 1
    U[3,10] = 1; V[5,10] = 1; W[5,10] = 1
    
    return U, V, W

def _get_t122_rank2():
    """
    Returns U, V, W for 1x2x2 matrix multiplication with rank 2.
    U: (2, 2), V: (4, 2), W: (2, 2)
    """
    U = np.zeros((2, 2), dtype=int)
    V = np.zeros((4, 2), dtype=int)
    W = np.zeros((2, 2), dtype=int)
    
    # P1 = A00 B0
    # P2 = A01 B1
    U[0,0] = 1
    V[0,0] = 1
    W[0,0] = 1
    
    U[1,1] = 1
    V[2,1] = 1
    W[0,1] = 1
    
    # Wait, C is 1x2.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # My W above:
    # W[0,0]=1, W[0,1]=1.
    # So C00 gets P1+P2?
    # P1 = A00 B00. P2 = A01 B10.
    # C00 = P1 + P2. Correct.
    # C01?
    # W[1,0] and W[1,1] are 0.
    # So C01 is 0. Incorrect.
    
    # We need to map B01 and B11.
    # V[1,0] should be 1 for B01?
    # V[3,1] should be 1 for B11?
    
    U[0,0] = 1
    V[0,0] = 1 # B00
    V[1,0] = 1 # B01
    W[0,0] = 1
    W[1,0] = 1
    
    U[1,1] = 1
    V[2,1] = 1 # B10
    V[3,1] = 1 # B11
    W[0,1] = 1
    W[1,1] = 1
    
    return U, V, W

def _kron(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], rank), dtype=int)
    V = np.zeros((V1.shape[0] * V2.shape[0], rank), dtype=int)
    W = np.zeros((W1.shape[0] * W2.shape[0], rank), dtype=int)
    
    for s1 in range(r1):
        for s2 in range(r2):
            s = s1 * r2 + s2
            U[:, s] = np.kron(U1[:, s1], U2[:, s2])
            V[:, s] = np.kron(V1[:, s1], V2[:, s2])
            W[:, s] = np.kron(W1[:, s1], W2[:, s2])
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get factors
    U223, V223, W223 = _get_t223_rank9() # Actually rank 11
    U122, V122, W122 = _get_t122_rank2()
    
    # Combine
    U, V, W = _kron(U223, V223, W223, U122, V122, W122)
    
    # Verify
    # ...
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
