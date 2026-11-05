import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V22 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Tensor product function
    def tensor_product(U1, V1, W1, U2, V2, W2):
        n1, m1, p1 = U1.shape[0] // U1.shape[1] if False else (U1.shape[0], U1.shape[1], W1.shape[0] // U1.shape[0]) # This is wrong, need explicit dims
        # Better to pass dimensions
        pass

    # Let's implement tensor product explicitly for known dims
    # M444 = M222 x M222
    # n=4, m=4, p=4, rank=49
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U444[i * 4 + j, r] = U22[i1 * 2 + j1, r1] * U22[i2 * 2 + j2, r2]
                            
                            # V
                            # j = j1*2+j2, k = k1*2+k2
                            # V index is j*p + k. p=4.
                            # V444 index: (j1*2+j2)*4 + (k1*2+k2)
                            
    # This loop is getting complicated. Let's use Kronecker products.
    # U444 = kron(U22, U22)?
    # U22 is 4x7. kron(U22, U22) is 16x49.
    # Index mapping:
    # U22[i1*2+j1, r1] * U22[i2*2+j2, r2]
    # Kron index: (i1*2+j1)*4 + (i2*2+j2) ? No.
    # Kron(A, B)[i_A * n_B + i_B, j_A * m_B + j_B] = A[i_A, j_A] * B[i_B, j_B]
    # Here A=U22 (4x7), B=U22 (4x7).
    # i_A = i1*2+j1, i_B = i2*2+j2.
    # Result index i = i_A * 4 + i_B = (i1*2+j1)*4 + i2*2+j2 = 8*i1 + 4*j1 + 2*i2 + j2.
    # We want i = i1*2+i2, j = j1*2+j2.
    # Index in U444 is i*4+j = (i1*2+i2)*4 + j1*2+j2 = 8*i1 + 4*i2 + 2*j1 + j2.
    # These are different.
    # So simple kron doesn't work directly with the index ordering.
    
    # We must construct manually or permute.
    # Let's stick to the manual loop for correctness, it's small.
    
    def build_tensor(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
        rank = U1.shape[1] * U2.shape[1]
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r1 in range(U1.shape[1]):
            for r2 in range(U2.shape[1]):
                r = r1 * U2.shape[1] + r2
                for i1 in range(n1):
                    for j1 in range(m1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                i = i1 * n2 + i2
                                j = j1 * m2 + j2
                                U[i * m + j, r] = U1[i1 * m1 + j1, r1] * U2[i2 * m2 + j2, r2]
                                
                                # V
                                # j1, k1 from V1. j2, k2 from V2.
                                # V1 index: j1 * p1 + k1
                                # V2 index: j2 * p2 + k2
                                # V index: j * p + k = (j1*m2+j2) * (p1*p2) + (k1*p2+k2)
                                # This is not separable in a simple kron way for V and W if we just loop.
                                # We need to loop over k1, k2 as well.
                                
        # This is too slow/complex.
        # Let's use the property that U, V, W are sparse.
        # But for 49 terms, it's fine.
        
        # Let's just compute U444, V444, W444 using explicit loops.
        
        U444 = np.zeros((16, 49))
        V444 = np.zeros((16, 49))
        W444 = np.zeros((16, 49))
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U444[i * 4 + j, r] = U22[i1 * 2 + j1, r1] * U22[i2 * 2 + j2, r2]
                                
                                for k1 in range(2):
                                    for k2 in range(2):
                                        k = k1 * 2 + k2
                                        V444[j * 4 + k, r] = V22[j1 * 2 + k1, r1] * V22[j2 * 2 + k2, r2]
                                        W444[k * 4 + i, r] = W22[k1 * 2 + i1, r1] * W22[k2 * 2 + i2, r2]
                                        
        # M446 from M444 x M112
        # M112: n=1, m=1, p=2. Rank 1.
        U112 = np.array([[1]])
        V112 = np.array([[1], [1]]) # j=0, k=0,1
        W112 = np.array([[1], [1]]) # k=0,1, i=0
        
        U446 = np.zeros((16, 49))
        V446 = np.zeros((24, 49)) # m=4, p=6 -> 24
        W446 = np.zeros((24, 49)) # p=6, n=4 -> 24
        
        # M444 (4,4,4) x M112 (1,1,2) -> (4,4,8)
        # We only need p=6.
        # k = k1 * 2 + k2. k1 in 0..3, k2 in 0..1.
        # We keep k in 0..5.
        
        for r1 in range(49):
            for r2 in range(1):
                r = r1 # rank 49
                for i1 in range(4):
                    for j1 in range(4):
                        for i2 in range(1):
                            for j2 in range(1):
                                i = i1
                                j = j1
                                U446[i * 4 + j, r] = U444[i1 * 4 + j1, r1] * U112[i2 * 1 + j2, r2]
                                
                                for k1 in range(4):
                                    for k2 in range(2):
                                        k = k1 * 2 + k2
                                        if k < 6:
                                            V446[j * 6 + k, r] = V444[j1 * 4 + k1, r1] * V112[j2 * 2 + k2, r2]
                                            W446[k * 4 + i, r] = W444[k1 * 4 + i1, r1] * W112[k2 * 1 + i2, r2]
                                            
        # M416: n=4, m=1, p=6. Rank 1.
        U416 = np.ones((4, 1))
        V416 = np.ones((6, 1))
        W416 = np.ones((24, 1))
        
        # Combine M446 and M416 into M456
        # n=4, m=5, p=6. Rank 50.
        U = np.zeros((20, 50))
        V = np.zeros((30, 50))
        W = np.zeros((24, 50))
        
        # Copy M446
        U[:16, :49] = U446
        V[:24, :49] = V446
        W[:, :49] = W446
        
        # Copy M416
        # U rows 16..19 (j=4)
        U[16:20, 49] = U416
        # V rows 24..29 (j=4)
        V[24:30, 49] = V416
        # W
        W[:, 49] = W416
        
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
