import time
import numpy as np

def construct():
    start_time = time.time()
    
    def tensor_product(D1, D2):
        """
        D1: M_{n1, m1, p1}
        D2: M_{n2, m2, p2}
        Returns D3: M_{n1*n2, m1*m2, p1*p2}
        """
        n1, m1, p1 = D1['dims']
        n2, m2, p2 = D2['dims']
        R1 = D1['U'].shape[1]
        R2 = D2['U'].shape[1]
        R = R1 * R2
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for r1 in range(R1):
            for r2 in range(R2):
                r = r1 * R2 + r2
                
                # U indices: i = i1*n2 + i2, j = j1*m2 + j2
                # U_new[i*m + j, r] = U1[i1*m1 + j1, r1] * U2[i2*m2 + j2, r2]
                for i1 in range(n1):
                    for j1 in range(m1):
                        u1_idx = i1 * m1 + j1
                        u1_val = D1['U'][u1_idx, r1]
                        if u1_val == 0: continue
                        
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_idx = i2 * m2 + j2
                                u2_val = D2['U'][u2_idx, r2]
                                if u2_val == 0: continue
                                
                                i = i1 * n2 + i2
                                j = j1 * m2 + j2
                                u_idx = i * m + j
                                U[u_idx, r] = u1_val * u2_val
                                
                # V indices: j = j1*m2 + j2, k = k1*p2 + k2
                # V_new[j*p + k, r] = V1[j1*p1 + k1, r1] * V2[j2*p2 + k2, r2]
                for j1 in range(m1):
                    for k1 in range(p1):
                        v1_idx = j1 * p1 + k1
                        v1_val = D1['V'][v1_idx, r1]
                        if v1_val == 0: continue
                        
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2_idx = j2 * p2 + k2
                                v2_val = D2['V'][v2_idx, r2]
                                if v2_val == 0: continue
                                
                                j = j1 * m2 + j2
                                k = k1 * p2 + k2
                                v_idx = j * p + k
                                V[v_idx, r] = v1_val * v2_val
                                
                # W indices: k = k1*p2 + k2, i = i1*n2 + i2
                # W_new[k*n + i, r] = W1[k1*n1 + i1, r1] * W2[k2*n2 + i2, r2]
                for k1 in range(p1):
                    for i1 in range(n1):
                        w1_idx = k1 * n1 + i1
                        w1_val = D1['W'][w1_idx, r1]
                        if w1_val == 0: continue
                        
                        for k2 in range(p2):
                            for i2 in range(n2):
                                w2_idx = k2 * n2 + i2
                                w2_val = D2['W'][w2_idx, r2]
                                if w2_val == 0: continue
                                
                                k = k1 * p2 + k2
                                i = i1 * n2 + i2
                                w_idx = k * n + i
                                W[w_idx, r] = w1_val * w2_val
                                
        return {'dims': (n, m, p), 'U': U, 'V': V, 'W': W}

    def direct_sum(D1, D2):
        """
        D1: M_{n, m1, p}
        D2: M_{n, m2, p}
        Returns D3: M_{n, m1+m2, p}
        """
        n, m1, p = D1['dims']
        _, m2, _ = D2['dims']
        R1 = D1['U'].shape[1]
        R2 = D2['U'].shape[1]
        R = R1 + R2
        
        m = m1 + m2
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Copy D1
        # U1 indices: i*m1 + j. New U indices: i*m + j.
        for i in range(n):
            for j in range(m1):
                U[i * m + j, :R1] = D1['U'][i * m1 + j, :]
        
        # Copy D2
        # U2 indices: i*m2 + j. New U indices: i*m + (m1+j).
        for i in range(n):
            for j in range(m2):
                U[i * m + (m1 + j), R1:] = D2['U'][i * m2 + j, :]
        
        # V1 indices: j*p + k. New V indices: j*p + k.
        for j in range(m1):
            for k in range(p):
                V[j * p + k, :R1] = D1['V'][j * p + k, :]
        
        # V2 indices: j*p + k. New V indices: (m1+j)*p + k.
        for j in range(m2):
            for k in range(p):
                V[(m1 + j) * p + k, R1:] = D2['V'][j * p + k, :]
        
        # W1 indices: k*n + i. New W indices: k*n + i.
        W[:, :R1] = D1['W']
        # W2 indices: k*n + i. New W indices: k*n + i.
        W[:, R1:] = D2['W']
        
        return {'dims': (n, m, p), 'U': U, 'V': V, 'W': W}

    # Base Decompositions
    
    # M_{1,1,2} Rank 2
    D_112 = {
        'dims': (1, 1, 2),
        'U': np.array([[1, 1]]),
        'V': np.array([[1, 0], [0, 1]]),
        'W': np.array([[1, 0], [0, 1]])
    }
    
    # M_{1,2,2} Rank 2
    D_122 = {
        'dims': (1, 2, 2),
        'U': np.array([[1, 0], [0, 1]]),
        'V': np.array([[1, 0], [0, 0], [0, 0], [0, 1]]),
        'W': np.array([[1, 0], [0, 1]])
    }
    
    # M_{3,1,1} Rank 3
    D_311 = {
        'dims': (3, 1, 1),
        'U': np.eye(3),
        'V': np.array([[1, 1, 1]]),
        'W': np.eye(3)
    }
    
    # Build M_{3,2,2} = M_{3,1,1} \otimes M_{1,2,2}
    D_322 = tensor_product(D_311, D_122)
    
    # Build M_{3,2,4} = M_{3,2,2} \otimes M_{1,1,2}
    D_324 = tensor_product(D_322, D_112)
    
    # Build M_{3,4,4} = M_{3,2,4} \oplus M_{3,2,4}
    D_344 = direct_sum(D_324, D_324)
    
    # Build M_{3,4,8} = M_{3,4,4} \otimes M_{1,1,2}
    D_348 = tensor_product(D_344, D_112)
    
    U = D_348['U']
    V = D_348['V']
    W = D_348['W']
    
    # Verify
    n, m, p = 3, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        raise TimeoutError("Too slow")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
