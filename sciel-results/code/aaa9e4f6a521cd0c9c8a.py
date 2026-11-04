"""Kronecker product of 3x3 schoolbook and 2x2 Strassen for 6x6 matrix mult."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5 second safety margin
    
    # 3x3 schoolbook decomposition (rank 27)
    n1, m1, p1 = 3, 3, 3
    r1 = n1 * m1 * p1  # 27
    U1 = np.zeros((n1 * m1, r1), dtype=int)
    V1 = np.zeros((m1 * p1, r1), dtype=int)
    W1 = np.zeros((p1 * n1, r1), dtype=int)
    col = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1, col] = 1
                V1[j1 * p1 + k1, col] = 1
                W1[k1 * n1 + i1, col] = 1
                col += 1
    
    # 2x2 Strassen decomposition (rank 7)
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = np.zeros((n2 * m2, r2), dtype=int)
    V2 = np.zeros((m2 * p2, r2), dtype=int)
    W2 = np.zeros((p2 * n2, r2), dtype=int)
    
    # Strassen indices
    # U2 rows: i2*m2 + j2
    # V2 rows: j2*p2 + k2
    # W2 rows: k2*n2 + i2
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B01-B10)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[2, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Kronecker product construction
    n, m, p = 6, 6, 6
    rank = r1 * r2  # 189
    
    # Reshape to 3D for easy Kronecker product of slices
    # U1: (n1*m1, r1) -> (n1, m1, r1)
    U1_3d = U1.reshape(n1, m1, r1)
    V1_3d = V1.reshape(m1, p1, r1)
    W1_3d = W1.reshape(p1, n1, r1)
    
    U2_3d = U2.reshape(n2, m2, r2)
    V2_3d = V2.reshape(m2, p2, r2)
    W2_3d = W2.reshape(p2, n2, r2)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # Kronecker product of the 2D slices
            # U_slice1: (n1, m1), U_slice2: (n2, m2)
            # Result: (n1*n2, m1*m2)
            U_slice = np.kron(U1_3d[:, :, c1], U2_3d[:, :, c2])
            V_slice = np.kron(V1_3d[:, :, c1], V2_3d[:, :, c2])
            W_slice = np.kron(W1_3d[:, :, c1], W2_3d[:, :, c2])
            
            U[:, c] = U_slice.flatten()
            V[:, c] = V_slice.flatten()
            W[:, c] = W_slice.flatten()
            
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
        if elapsed > deadline:
            break

    # Convert to lists of lists as required by interface? 
    # The interface says "factor matrices of shapes... with integer/half-integer entries".
    # It doesn't explicitly say lists of lists, but the seed program returned lists of lists.
    # However, returning NumPy arrays is often accepted if the evaluator handles it.
    # To be safe and match the seed program's return type structure:
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
