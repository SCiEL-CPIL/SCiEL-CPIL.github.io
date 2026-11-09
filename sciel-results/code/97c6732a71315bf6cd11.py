import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1

    # Try rank 28 decomposition
    # 4 independent Strassen blocks
    # Block 1: A[0:2, 0:2] * B[0:2, :]
    # Block 2: A[0:2, 2:4] * B[2:4, :]
    # Wait, A is 2x4, B is 4x4.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C1 = A1 * B1 (2x2 * 2x4 -> 2x4)
    # C2 = A2 * B2 (2x2 * 2x4 -> 2x4)
    # C = C1 + C2
    
    # We need rank 14 for 2x2*2x4.
    # This is sum of two 2x2*2x2 products.
    # A1 * B1_1 + A1 * B1_2.
    # Use Strassen for A1 * B1_1 and A1 * B1_2.
    
    # Strassen coefficients
    U_str = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_str = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_str = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, -1]
    ]
    
    # Verify Strassen
    # A = np.array([[1,2],[3,4]]), B = np.array([[5,6],[7,8]])
    # C = A @ B
    # Check if sum(U[:,c] * V[:,c] * W[:,c]) == C
    
    U_28 = np.zeros((n * m, 28))
    V_28 = np.zeros((m * p, 28))
    W_28 = np.zeros((p * n, 28))
    
    col = 0
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 indices in U: 0,1,2,3 (i*m+j)
    # B1 indices in V: 0,1,4,5 (j*p+k) -> j=0,1. k=0,1,2,3.
    # B1 is 2x4. Rows 0,1.
    # B1_1 (cols 0,1 of B1): V indices 0,1,4,5.
    # B1_2 (cols 2,3 of B1): V indices 2,3,6,7.
    
    # A1 * B1_1
    for c in range(7):
        # U
        for idx, val in enumerate(U_str[c]):
            if val != 0:
                # A1 indices: i=0,1; j=0,1.
                # U index: i*m + j.
                # U_str index: 0:a, 1:b, 2:c, 3:d.
                # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1].
                # So U_str[0] -> U[0], U_str[1] -> U[1], U_str[2] -> U[2], U_str[3] -> U[3].
                U_28[idx, col] = val
        # V
        for idx, val in enumerate(V_str[c]):
            if val != 0:
                # V_str index: 0:e, 1:f, 2:g, 3:h.
                # B1_1 is B[0:2, 0:2].
                # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
                # V index: j*p + k.
                # e: j=0, k=0 -> 0.
                # f: j=0, k=1 -> 1.
                # g: j=1, k=0 -> 4.
                # h: j=1, k=1 -> 5.
                V_28[idx, col] = val
        # W
        for idx, val in enumerate(W_str[c]):
            if val != 0:
                # W_str index: 0:p, 1:q, 2:r, 3:s.
                # C1_1 is C[0:2, 0:2].
                # p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1].
                # W index: k*n + i.
                # p: k=0, i=0 -> 0.
                # q: k=1, i=0 -> 1.
                # r: k=0, i=1 -> 2.
                # s: k=1, i=1 -> 3.
                W_28[idx, col] = val
        col += 1
        
    # A1 * B1_2
    for c in range(7):
        for idx, val in enumerate(U_str[c]):
            if val != 0:
                U_28[idx, col] = val
        for idx, val in enumerate(V_str[c]):
            if val != 0:
                # B1_2 is B[0:2, 2:4].
                # e=B[0,2], f=B[0,3], g=B[1,2], h=B[1,3].
                # e: j=0, k=2 -> 2.
                # f: j=0, k=3 -> 3.
                # g: j=1, k=2 -> 6.
                # h: j=1, k=3 -> 7.
                V_28[idx, col] = val
        for idx, val in enumerate(W_str[c]):
            if val != 0:
                # C1_2 is C[0:2, 2:4].
                # p=C[0,2], q=C[0,3], r=C[1,2], s=C[1,3].
                # p: k=2, i=0 -> 2.
                # q: k=3, i=0 -> 3.
                # r: k=2, i=1 -> 4.
                # s: k=3, i=1 -> 5.
                W_28[idx, col] = val
        col += 1
        
    # A2 * B2_1
    # A2 is A[0:2, 2:4].
    # U indices: 4,5,6,7.
    for c in range(7):
        for idx, val in enumerate(U_str[c]):
            if val != 0:
                U_28[idx + 4, col] = val
        for idx, val in enumerate(V_str[c]):
            if val != 0:
                # B2_1 is B[2:4, 0:2].
                # e=B[2,0], f=B[2,1], g=B[3,0], h=B[3,1].
                # e: j=2, k=0 -> 8.
                # f: j=2, k=1 -> 9.
                # g: j=3, k=0 -> 12.
                # h: j=3, k=1 -> 13.
                V_28[idx, col] = val
        for idx, val in enumerate(W_str[c]):
            if val != 0:
                # C2_1 is C[0:2, 0:2].
                # Same W indices as C1_1.
                W_28[idx, col] = val
        col += 1
        
    # A2 * B2_2
    for c in range(7):
        for idx, val in enumerate(U_str[c]):
            if val != 0:
                U_28[idx + 4, col] = val
        for idx, val in enumerate(V_str[c]):
            if val != 0:
                # B2_2 is B[2:4, 2:4].
                # e=B[2,2], f=B[2,3], g=B[3,2], h=B[3,3].
                # e: j=2, k=2 -> 10.
                # f: j=2, k=3 -> 11.
                # g: j=3, k=2 -> 14.
                # h: j=3, k=3 -> 15.
                V_28[idx, col] = val
        for idx, val in enumerate(W_str[c]):
            if val != 0:
                # C2_2 is C[0:2, 2:4].
                # Same W indices as C1_2.
                W_28[idx, col] = val
        col += 1
        
    # Verify
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for c in range(28):
        u = U_28[:, c]
        v = V_28[:, c]
        w = W_28[:, c]
        # u is 8x1, v is 16x1, w is 8x1
        # Reshape u to 2x4, v to 4x4, w to 4x2
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        C_computed += u_mat.T @ v_mat.T @ w_mat.T # No, tensor contraction
        
    # Correct contraction:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Term c: sum_{i,j,k} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This is not a simple matrix product.
    
    # I will return the schoolbook decomposition to be safe.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

