import time
import numpy as np

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor for validation
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                target_tensor[u, v, w] = 1.0
    
    def to_half_int_list(matrix):
        scaled = np.rint(matrix * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]
    
    def validate(U, V, W):
        """Check if U, V, W decompose the tensor exactly."""
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, target_tensor, atol=1e-8)
    
    # Schoolbook rank 18
    U_sb = np.zeros((n*m, 18))
    V_sb = np.zeros((m*p, 18))
    W_sb = np.zeros((p*n, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    # Try to find a rank-17 decomposition
    # The known rank of 2x3x3 is 17.
    # Strategy: Use Strassen for the 2x2x2 core and add schoolbook terms for the rest.
    # 
    # Partition:
    # A = [[A00, A01, A02], [A10, A11, A12]]
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    # C = [[C00, C01, C02], [C10, C11, C12]]
    #
    # We can use Strassen on the subproblem involving A[:, 0:2] and B[0:2, :] to compute part of C[:, 0:2].
    # But this is tricky because B has 3 rows.
    #
    # Alternative: Use the explicit rank-17 construction from the literature.
    # One known construction uses 17 scalar multiplications.
    #
    # Let's try to construct it manually.
    # We need to compute:
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C12 = A10*B02 + A11*B12 + A12*B22
    #
    # There are 18 terms. We need to save 1 multiplication.
    #
    # Let's try to use Strassen on the 2x2 submatrix formed by A[0:2, 0:2] and B[0:2, 0:2].
    # This gives us 7 multiplications to compute C[0:2, 0:2] using only A[0:2, 0:2] and B[0:2, 0:2].
    # Then we need to add the contributions from A[0:2, 2] and B[2, 0:2] and B[0:2, 2] and A[0:2, 2]*B[2, 2].
    #
    # The terms not covered by the Strassen block are:
    # C00: A02*B20
    # C01: A02*B21
    # C02: A00*B02 + A01*B12 + A02*B22
    # C10: A12*B20
    # C11: A12*B21
    # C12: A10*B02 + A11*B12 + A12*B22
    #
    # Total remaining terms: 1 + 1 + 3 + 1 + 1 + 3 = 10 terms.
    # Total multiplications: 7 (Strassen) + 10 (schoolbook) = 17.
    #
    # Let's implement this.
    
    rank = 17
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    col = 0
    
    # Strassen on A[0:2, 0:2] and B[0:2, 0:2]
    # p1 = (A00+A11)(B00+B11)
    U[0, col] = 1.0; U[4, col] = 1.0  # A00, A11
    V[0, col] = 1.0; V[4, col] = 1.0  # B00, B11
    W[0, col] = 1.0; W[5, col] = 1.0  # C00, C11
    col += 1
    
    # p2 = (A10+A11)(B00+B01)
    U[3, col] = 1.0; U[4, col] = 1.0  # A10, A11
    V[0, col] = 1.0; V[1, col] = 1.0  # B00, B01
    W[0, col] = -1.0; W[1, col] = 1.0; W[3, col] = -1.0; W[4, col] = 1.0 # C00, C01, C10, C11
    col += 1
    
    # p3 = (A00-A11)(B10+B11)
    U[0, col] = 1.0; U[4, col] = -1.0 # A00, A11
    V[3, col] = 1.0; V[4, col] = 1.0  # B10, B11
    W[0, col] = 1.0; W[5, col] = 1.0  # C00, C11
    col += 1
    
    # p4 = A00(B01-B11)
    U[0, col] = 1.0  # A00
    V[1, col] = 1.0; V[4, col] = -1.0 # B01, B11
    W[0, col] = -1.0; W[1, col] = 1.0 # C00, C01
    col += 1
    
    # p5 = A11(B10-B00)
    U[4, col] = 1.0  # A11
    V[3, col] = 1.0; V[0, col] = -1.0 # B10, B00
    W[0, col] = -1.0; W[3, col] = 1.0 # C00, C10
    col += 1
    
    # p6 = (A00+A01)(B11-B01)
    U[0, col] = 1.0; U[1, col] = 1.0 # A00, A01
    V[4, col] = 1.0; V[1, col] = -1.0 # B11, B01
    W[1, col] = 1.0; W[4, col] = -1.0 # C01, C11
    col += 1
    
    # p7 = (A10+A11)(B00-B10)
    U[3, col] = 1.0; U[4, col] = 1.0 # A10, A11
    V[0, col] = 1.0; V[3, col] = -1.0 # B00, B10
    W[0, col] = -1.0; W[1, col] = 1.0; W[3, col] = 1.0; W[4, col] = -1.0 # C00, C01, C10, C11
    col += 1
    
    # Now add the schoolbook terms for the remaining parts
    # C00: A02*B20
    U[2, col] = 1.0; V[6, col] = 1.0; W[0, col] = 1.0
    col += 1
    
    # C01: A02*B21
    U[2, col] = 1.0; V[7, col] = 1.0; W[1, col] = 1.0
    col += 1
    
    # C02: A00*B02 + A01*B12 + A02*B22
    U[0, col] = 1.0; V[2, col] = 1.0; W[2, col] = 1.0
    col += 1
    
    U[1, col] = 1.0; V[5, col] = 1.0; W[2, col] = 1.0
    col += 1
    
    U[2, col] = 1.0; V[8, col] = 1.0; W[2, col] = 1.0
    col += 1
    
    # C10: A12*B20
    U[5, col] = 1.0; V[6, col] = 1.0; W[3, col] = 1.0
    col += 1
    
    # C11: A12*B21
    U[5, col] = 1.0; V[7, col] = 1.0; W[4, col] = 1.0
    col += 1
    
    # C12: A10*B02 + A11*B12 + A12*B22
    U[3, col] = 1.0; V[2, col] = 1.0; W[5, col] = 1.0
    col += 1
    
    U[4, col] = 1.0; V[5, col] = 1.0; W[5, col] = 1.0
    col += 1
    
    U[5, col] = 1.0; V[8, col] = 1.0; W[5, col] = 1.0
    col += 1
    
    if validate(U, V, W):
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    else:
        # Fallback to schoolbook
        return {"U": to_half_int_list(U_sb), "V": to_half_int_list(V_sb), "W": to_half_int_list(W_sb)}
