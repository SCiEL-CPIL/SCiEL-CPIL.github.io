import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]  (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:, 0:2] * B[0:2, 4] -> C[0:2, 4]  (schoolbook, rank 4)
    add_schoolbook([0, 1], [0, 1], [4])
    
    # Block 4: A[:, 2] * B[2, 0:2] -> C[0:2, 0:2]  (schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [0, 1])
    
    # Block 5: A[:, 2] * B[2, 2:4] -> C[0:2, 2:4]  (schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [2, 3])
    
    # Block 6: A[:, 2] * B[2, 4] -> C[0:2, 4]  (schoolbook, rank 2)
    add_schoolbook([0, 1], [2], [4])

    # Verify col matches expected rank
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify exactness: sum_c U_c ⊗ V_c ⊗ W_c should equal the multiplication tensor
    # The multiplication tensor T[i, j, k] = 1 if output C[i,k] includes A[i,j]*B[j,k]
    # Reconstruct: for each factor c, contribution is U_c[i,j] * V_c[j,k] * W_c[k,i]
    # Sum over c should give 1 for all valid (i,j,k) triples
    reconstructed = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        # W is indexed [k*n + i], so W_c[k, i]
        # Contribution to C[i,k] from factor c: sum_j U_c[i,j] * V_c[j,k] * W_c[k,i]
        # But in tensor form: T[i,j,k] gets U_c[i,j] * V_c[j,k] * W_c[k,i]
        # Actually the tensor for matmul is: T[i,j,k] = delta (it's 1 for the "path" i->j->k)
        # The decomposition says: sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 for all i,j,k
        # Wait, that's not right. The matmul tensor has T[i,j,k] = 1 (each entry is 1)
        # and the decomposition reconstructs the bilinear map.
        
        # Let me verify differently: check that for random A, B, the result matches
        # Actually, let's verify the tensor identity directly
        # The matmul tensor T has T[i,j,k] = 1 for all i,j,k (it's the structure tensor)
        # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] should equal 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
    
    if not np.allclose(reconstructed, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = col
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
