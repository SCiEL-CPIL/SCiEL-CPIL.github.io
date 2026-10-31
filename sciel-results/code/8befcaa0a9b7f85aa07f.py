import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4
    
    # Strassen's 7 multiplications for 2x2
    # We define U2, V2, W2 such that sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] = delta_{i,k} for 2x2?
    # Actually, the tensor is T[i,j, j,k, k,i] = 1 if indices match for multiplication.
    # For 2x2, the tensor is T[i,j, j,k, k,i] = 1 if i==k? No.
    # The multiplication tensor T_mult satisfies T_mult(A,B,C) = tr(ABC) is NOT the one.
    # The one is T_mult(A,B,C) = sum_{i,j,k} A_{ij} B_{jk} C_{ki} is the trace.
    # The tensor for C=AB is T[i,j, j,k, i,k] = 1?
    # Let's stick to the coefficients derived from Strassen.
    
    # Strassen coefficients for 2x2
    # U2[i, j, r]
    # V2[j, k, r]
    # W2[k, i, r]
    
    # r=0: M0 = (A00 + A11)(B00 + B11)
    # A00=1, A11=1. B00=1, B11=1.
    # C00 += 1, C11 += 1.
    # W2[0,0,0]=1, W2[1,1,0]=1.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Index mapping for 2x2: i,j in {0,1} -> i*2+j
    # U2[i*2+j, r]
    # V2[j*2+k, r]
    # W2[k*2+i, r]
    
    # M0 = (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M1 = (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10+=M1, C11-=M1
    
    # M2 = A00(B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01+=M2, C11+=M2
    
    # M3 = A11(B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00+=M3, C10+=M3
    
    # M4 = (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00-=M4, C01+=M4
    
    # M5 = (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11+=M5
    
    # M6 = (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00+=M6
    
    # Verify 2x2 decomposition
    # sum_r U2[i,j,r] V2[j,k,r] W2[k,i,r] should be delta_{i,k}
    # Note: The tensor is T[i,j, j,k, k,i]
    # For i=0, k=0: sum_j T[0,j, j,0, 0,0] = T[0,0,0,0,0,0] + T[0,1,1,0,0,0]
    # This should be 1.
    
    # Construct 4x4 using Kronecker product
    # U = U2 (x) U2
    # Shape: (4*4, 7*7) = (16, 49)
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    # The indices for 4x4 are:
    # U[i*4+j, r] where i,j in {0..3}
    # V[j*4+k, r]
    # W[k*4+i, r]
    
    # Kronecker product of U2 (4x7) and U2 (4x7) gives (16x49).
    # The index ordering of Kronecker product A (m x n) and B (p x q) is:
    # (A (x) B)[i*m + k, j*n + l] = A[i,j] B[k,l]
    # Here m=4, n=7.
    # Index i in U2 is i2*2+j2. Index k in U2 is i2'*2+j2'.
    # So index in U is (i2*2+j2)*4 + (i2'*2+j2') = i2*8 + j2*4 + i2'*2 + j2'.
    # This corresponds to i = i2*2 + i2', j = j2*2 + j2'?
    # Let's check: i = i2*2 + i2'. j = j2*2 + j2'.
    # Then i*4+j = (i2*2+i2')*4 + j2*2+j2' = i2*8 + i2'*4 + j2*2 + j2'.
    # The Kronecker index is i2*8 + j2*4 + i2'*2 + j2'.
    # This is different. i2' and j2 are swapped in position?
    # i2*8 + i2'*4 + j2*2 + j2' vs i2*8 + j2*4 + i2'*2 + j2'.
    # So the Kronecker product U (x) U2 does NOT directly give the correct indexing for matrix multiplication blocks.
    # We need to permute the indices.
    
    # Correct indexing for block multiplication:
    # A is 4x4, blocks A_{I,J} are 2x2.
    # A_{I,J}[i,j] is at index (I*2+i)*4 + (J*2+j) = I*8 + i*4 + J*2 + j.
    # U2 has indices I,J.
    # So we need U[(I*2+i)*4 + (J*2+j), r1*7+r2] = U2[I*2+J, r1] * U2[I'*2+J', r2]?
    # No, U2 depends on the block indices I,J and the inner indices i,j.
    # U2[I*2+J, r] is the coefficient for A_{I,J} in the r-th product.
    # But A_{I,J} is a 2x2 matrix. The product is M_{r1,r2} = (A_{I1,J1} + ...)(B_{J1,K1} + ...).
    # The inner 2x2 multiplication is handled by the second level of Strassen.
    # So U4[I*2+i, J*2+j, r1*7+r2] = U2[I*2+J, r1] * U2[i*2+j, r2]?
    # Wait, U2[i*2+j, r2] is the coefficient for the inner matrix element.
    # Yes, this is the Kronecker product structure.
    # The index in U4 is (I*2+i)*4 + (J*2+j).
    # Let's call this index u.
    # u = I*8 + i*4 + J*2 + j.
    # The Kronecker product U (x) U2 has index u_kron = (I*2+J)*4 + (i*2+j) = I*8 + J*4 + i*2 + j.
    # We need to map u_kron to u.
    # u = I*8 + i*4 + J*2 + j.
    # u_kron = I*8 + J*4 + i*2 + j.
    # This is a permutation of the indices J and i.
    # We can construct U4 directly or permute the Kronecker product.
    
    # Let's construct directly to be safe.
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for I in range(2):
                for J in range(2):
                    for i in range(2):
                        for j in range(2):
                            u_idx = (I*2+i)*4 + (J*2+j)
                            U4[u_idx, r] += U2[I*2+J, r1] * U2[i*2+j, r2]
                            
                            v_idx = (J*2+j)*4 + (K*2+k) # Wait, V depends on J,K and j,k
                            # V2[J*2+K, r1] * V2[j*2+k, r2]
                            # We need to loop K,k as well.
                            
    # This direct construction is O(49 * 16 * 16) which is fine.
    # But we can just use the permutation.
    # Permutation P such that P[u_kron] = u.
    # u_kron = I*8 + J*4 + i*2 + j.
    # u = I*8 + i*4 + J*2 + j.
    # Let's create a permutation vector.
    perm_U = np.zeros(16, dtype=int)
    for I in range(2):
        for J in range(2):
            for i in range(2):
                for j in range(2):
                    u_kron = I*8 + J*4 + i*2 + j
                    u = I*8 + i*4 + J*2 + j
                    perm_U[u] = u_kron
    
    # U4[u, r] = U_kron[perm_U[u], r]
    U4 = U[perm_U, :]
    
    # Similarly for V.
    # V index: j*4+k.
    # Block indices J,K. Inner j,k.
    # v_idx = (J*2+j)*4 + (K*2+k) = J*8 + j*4 + K*2 + k.
    # v_kron = (J*2+K)*4 + (j*2+k) = J*8 + K*4 + j*2 + k.
    # Permutation for V:
    perm_V = np.zeros(16, dtype=int)
    for J in range(2):
        for K in range(2):
            for j in range(2):
                for k in range(2):
                    v_kron = J*8 + K*4 + j*2 + k
                    v = J*8 + j*4 + K*2 + k
                    perm_V[v] = v_kron
    V4 = V[perm_V, :]
    
    # Similarly for W.
    # W index: k*4+i.
    # Block indices K,I. Inner k,i.
    # w_idx = (K*2+k)*4 + (I*2+i) = K*8 + k*4 + I*2 + i.
    # w_kron = (K*2+I)*4 + (k*2+i) = K*8 + I*4 + k*2 + i.
    perm_W = np.zeros(16, dtype=int)
    for K in range(2):
        for I in range(2):
            for k in range(2):
                for i in range(2):
                    w_kron = K*8 + I*4 + k*2 + i
                    w = K*8 + k*4 + I*2 + i
                    perm_W[w] = w_kron
    W4 = W[perm_W, :]
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected = 1.0 if i == k else 0.0
                    actual = 0.0
                    for r in range(49):
                        actual += U_mat[i*4+j, r] * V_mat[j*4+k, r] * W_mat[k*4+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    if verify(U4, V4, W4):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        # Fallback to schoolbook
        U_f = np.zeros((16, 64))
        V_f = np.zeros((16, 64))
        W_f = np.zeros((16, 64))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_f[i*4+j, idx] = 1.0
                    V_f[j*4+k, idx] = 1.0
                    W_f[k*4+i, idx] = 1.0
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

