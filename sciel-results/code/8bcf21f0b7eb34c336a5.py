import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin
    
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Build tensor product decompositions
    # M_{a,b,c} tensor: U shape (a*b, r), V shape (b*c, r), W shape (c*a, r)
    # T[i,j,k] = 1 for all i in [0,a), j in [0,b), k in [0,c)
    
    def schoolbook(a, b, c):
        """Schoolbook decomposition for M_{a,b,c}"""
        rank = a * b * c
        U = np.zeros((a*b, rank))
        V = np.zeros((b*c, rank))
        W = np.zeros((c*a, rank))
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i*b + j, col] = 1
                    V[j*c + k, col] = 1
                    W[k*a + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def tensor_product_decomp(U1, V1, W1, r1, U2, V2, W2, r2):
        """
        Tensor product of two matrix multiplication tensors.
        M_{a1,b1,c1} ⊗ M_{a2,b2,c2} = M_{a1*a2, b1*b2, c1*c2}
        
        The tensor product combines the decompositions.
        """
        a1, b1, c1 = U1.shape[0]//b1, U1.shape[0]//U1.shape[1] if False else None, None
        # We need to know the dimensions. Let me pass them explicitly.
        pass
    
    # Better approach: build up from known small decompositions
    # M_{1,1,1}: rank 1
    # M_{1,1,2}: rank 2 (schoolbook)
    # M_{1,2,1}: rank 2 (schoolbook)  
    # M_{2,1,1}: rank 2 (schoolbook)
    # M_{1,2,2}: rank 4 (schoolbook - 4 non-zero entries in general position)
    
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{3,4,4} = M_{3,2,2} ⊗ M_{1,2,2}
    # M_{3,2,2} = M_{3,1,1} ⊗ M_{1,2,2}
    # M_{3,1,1}: rank 3
    # M_{1,2,2}: rank 4
    # M_{1,1,2}: rank 2
    
    # So rank ≤ 3 * 4 * 4 * 2 = 96
    
    # Let me implement the tensor product properly
    def matmul_tensor_product(U1, V1, W1, r1, a1, b1, c1, U2, V2, W2, r2, a2, b2, c2):
        """
        Compute tensor product of two matrix multiplication tensor decompositions.
        
        M_{a1,b1,c1} ⊗ M_{a2,b2,c2} = M_{a1*a2, b1*b2, c1*c2}
        
        The input index for the product: (i1*a2+i2, j1*b2+j2)
        The inner index: (j1*b2+j2, k1*c2+k2)  
        The output index: (k1*c2+k2, i1*a2+i2)
        
        U_prod[(i1*a2+i2)*(b1*b2) + (j1*b2+j2), c1_idx*r2+c2_idx] 
            = U1[i1*b1+j1, c1_idx] * U2[i2*b2+j2, c2_idx]
        
        V_prod[(j1*b2+j2)*(c1*c2) + (k1*c2+k2), c1_idx*r2+c2_idx]
            = V1[j1*c1+k1, c1_idx] * V2[j2*c2+k2, c2_idx]
        
        W_prod[(k1*c2+k2)*(a1*a2) + (i1*a2+i2), c1_idx*r2+c2_idx]
            = W1[k1*a1+i1, c1_idx] * W2[k2*a2+i2, c2_idx]
        """
        a, b, c = a1*a2, b1*b2, c1*c2
        r = r1 * r2
        U = np.zeros((a*b, r))
        V = np.zeros((b*c, r))
        W = np.zeros((c*a, r))
        
        for c1_idx in range(r1):
            for c2_idx in range(r2):
                c_idx = c1_idx * r2 + c2_idx
                for i1 in range(a1):
                    for j1 in range(b1):
                        for k1 in range(c1):
                            u1_val = U1[i1*b1 + j1, c1_idx]
                            v1_val = V1[j1*c1 + k1, c1_idx]
                            w1_val = W1[k1*a1 + i1, c1_idx]
                            if u1_val == 0 and v1_val == 0 and w1_val == 0:
                                continue
                            for i2 in range(a2):
                                for j2 in range(b2):
                                    for k2 in range(c2):
                                        u2_val = U2[i2*b2 + j2, c2_idx]
                                        v2_val = V2[j2*c2 + k2, c2_idx]
                                        w2_val = W2[k2*a2 + i2, c2_idx]
                                        if u2_val == 0 and v2_val == 0 and w2_val == 0:
                                            continue
                                        
                                        u_val = u1_val * u2_val
                                        v_val = v1_val * v2_val
                                        w_val = w1_val * w2_val
                                        
                                        i = i1 * a2 + i2
                                        j = j1 * b2 + j2
                                        k = k1 * c2 + k2
                                        
                                        U[i*b + j, c_idx] = u_val
                                        V[j*c + k, c_idx] = v_val
                                        W[k*a + i, c_idx] = w_val
        return U, V, W, r
    
    # Base decompositions
    # M_{1,1,1}
    U_111, V_111, W_111, r_111 = schoolbook(1, 1, 1)
    
    # M_{1,1,2}
    U_112, V_112, W_112, r_112 = schoolbook(1, 1, 2)
    
    # M_{1,2,1}
    U_121, V_121, W_121, r_121 = schoolbook(1, 2, 1)
    
    # M_{2,1,1}
    U_211, V_211, W_211, r_211 = schoolbook(2, 1, 1)
    
    # M_{1,2,2}
    U_122, V_122, W_122, r_122 = schoolbook(1, 2, 2)
    
    # M_{3,1,1}
    U_311, V_311, W_311, r_311 = schoolbook(3, 1, 1)
    
    # Build M_{3,2,2} = M_{3,1,1} ⊗ M_{1,2,2}
    U_322, V_322, W_322, r_322 = matmul_tensor_product(
        U_311, V_311, W_311, r_311, 3, 1, 1,
        U_122, V_122, W_122, r_122, 1, 2, 2
    )
    
    # Build M_{3,4,4} = M_{3,2,2} ⊗ M_{1,2,2}
    U_344, V_344, W_344, r_344 = matmul_tensor_product(
        U_322, V_322, W_322, r_322, 3, 2, 2,
        U_122, V_122, W_122, r_122, 1, 2, 2
    )
    
    # Build M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    U, V, W, rank = matmul_tensor_product(
        U_344, V_344, W_344, r_344, 3, 4, 4,
        U_112, V_112, W_112, r_112, 1, 1, 2
    )
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct"""
        nm, mp, pn = n*m, m*p, p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False, (i, j, k, val)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p)
    if not valid:
        print(f"Verification failed: {info}")
        # Fall back to schoolbook
        U, V, W = schoolbook(n, m, p)[:3]
        rank = n * m * p
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return schoolbook as safety
        U, V, W = schoolbook(n, m, p)[:3]
        rank = n * m * p
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
