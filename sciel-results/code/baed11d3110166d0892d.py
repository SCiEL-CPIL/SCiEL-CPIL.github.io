import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = 4, 4, 4
    rank = 49

    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 coefficients
    # U2: (4, 7) where row is (i,k) flattened 0..3, col is s
    # V2: (4, 7) where row is (k,j) flattened 0..3, col is s
    # W2: (4, 7) where row is (j,i) flattened 0..3, col is s
    
    # Note: Indices for 2x2 block:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (1,0)
    # 3: (1,1)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # Row 0: (0,0)
        [0, 0, 0, 0, 1, 0, 1],  # Row 1: (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # Row 2: (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # Row 3: (1,1)
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # Row 0: (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # Row 1: (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # Row 2: (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # Row 3: (1,1)
    ]
    
    # W2 indices correspond to C indices (j, i) flattened.
    # C = [[C00, C01], [C10, C11]]
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Flattened C indices:
    # 0: (0,0) -> C00
    # 1: (0,1) -> C01
    # 2: (1,0) -> C10
    # 3: (1,1) -> C11
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # Row 0: C00 (M0, M3, -M4, M6)
        [0, 0, 1, 0, 1, 0, 0],  # Row 1: C01 (M2, M4)
        [0, 1, 0, 1, 0, 0, 0],  # Row 2: C10 (M1, M3)
        [1, -1, 1, 0, 0, 1, 0]  # Row 3: C11 (M0, -M1, M2, M5)
    ]
    
    # Construct 4x4 decomposition via Kronecker product
    # Rank index r = s * 7 + t
    # U[i*4+j, r] where i,j are 4x4 indices.
    # i = I * 2 + i', j = J * 2 + j'
    # U[i*4+j, s*7+t] = U2[I*2+J, s] * U2[i'*2+j', t]
    
    # V[k*4+l, r] where k,l are 4x4 indices.
    # k = K * 2 + k', l = L * 2 + l'
    # V[k*4+l, s*7+t] = V2[K*2+L, s] * V2[k'*2+l', t]
    
    # W[j*4+i, r] (Trace convention: W[k*n+i] where k is col of A/row of C, i is row of A/col of C)
    # Wait, let's map carefully.
    # A is n x m (4x4). B is m x p (4x4). C is p x n (4x4).
    # U index: (i, k) from A. i in 0..3, k in 0..3.
    # V index: (k, j) from B. k in 0..3, j in 0..3.
    # W index: (j, i) from C. j in 0..3, i in 0..3.
    
    # In the Kronecker structure:
    # i = I * 2 + i'
    # k = K * 2 + k'  (Wait, in A, the second index is the column. In B, the first index is the row. They must match for multiplication.)
    # j = J * 2 + j'
    
    # Let's re-verify the block structure indices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # A element (i, k):
    # i = I*2 + i', k = K*2 + k'
    # If I=0, K=0, it's in A00.
    
    # B element (k, j):
    # k = K*2 + k', j = J*2 + j'
    # If K=0, J=0, it's in B00.
    
    # C element (j, i):
    # j = J*2 + j', i = I*2 + i'
    # If J=0, I=0, it's in C00.
    
    # The Strassen decomposition acts on the blocks.
    # Outer Strassen (s) acts on the 2x2 block structure (I,K), (K,J), (J,I).
    # Inner Strassen (t) acts on the 2x2 elements within the blocks (i',k'), (k',j'), (j',i').
    
    # U2 for outer: U2_outer[I*2+K, s]
    # U2 for inner: U2_inner[i'*2+k', t]
    # So U[i*4+k, s*7+t] = U2[I*2+K, s] * U2[i'*2+k', t]
    
    # V2 for outer: V2_outer[K*2+J, s]
    # V2 for inner: V2_inner[k'*2+j', t]
    # So V[k*4+j, s*7+t] = V2[K*2+J, s] * V2[k'*2+j', t]
    
    # W2 for outer: W2_outer[J*2+I, s]
    # W2 for inner: W2_inner[j'*2+i', t]
    # So W[j*4+i, s*7+t] = W2[J*2+I, s] * W2[j'*2+i', t]
    
    # Note: The U2, V2, W2 arrays defined above are for the 2x2 case.
    # They work for both outer and inner levels.
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Loop over all indices i, k for U
            for i in range(4):
                for k in range(4):
                    I = i // 2
                    i_prime = i % 2
                    K = k // 2
                    k_prime = k % 2
                    
                    # Outer coeff
                    u_outer = U2[I * 2 + K][s]
                    if u_outer == 0:
                        continue
                    # Inner coeff
                    u_inner = U2[i_prime * 2 + k_prime][t]
                    if u_inner == 0:
                        continue
                    
                    U[i * 4 + k][r] = u_outer * u_inner
            
            # Loop over all indices k, j for V
            for k in range(4):
                for j in range(4):
                    K = k // 2
                    k_prime = k % 2
                    J = j // 2
                    j_prime = j % 2
                    
                    v_outer = V2[K * 2 + J][s]
                    if v_outer == 0:
                        continue
                    v_inner = V2[k_prime * 2 + j_prime][t]
                    if v_inner == 0:
                        continue
                    
                    V[k * 4 + j][r] = v_outer * v_inner
            
            # Loop over all indices j, i for W
            for j in range(4):
                for i in range(4):
                    J = j // 2
                    j_prime = j % 2
                    I = i // 2
                    i_prime = i % 2
                    
                    w_outer = W2[J * 2 + I][s]
                    if w_outer == 0:
                        continue
                    w_inner = W2[j_prime * 2 + i_prime][t]
                    if w_inner == 0:
                        continue
                    
                    W[j * 4 + i][r] = w_outer * w_inner
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {
        "U": U,
        "V": V,
        "W": W
    }
