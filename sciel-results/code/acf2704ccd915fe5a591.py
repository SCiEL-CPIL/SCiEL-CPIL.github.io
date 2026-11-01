import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2
    matrix by a 2x8 matrix.
    
    The tensor is M_{2,2,2} \otimes I_4.
    We construct the rank-28 Strassen decomposition and check for linear 
    dependencies to see if the rank can be reduced.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16

    # Strassen products M1..M7 for 2x2x2
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]

    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]

    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    ]

    rank_28 = 28
    U = np.zeros((nm, rank_28), dtype=np.int32)
    V = np.zeros((mp, rank_28), dtype=np.int32)
    W = np.zeros((pn, rank_28), dtype=np.int32)

    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s

            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j, r] = val

            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global, r] = val

            for c_local, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_local // 2
                    k_local = c_local % 2
                    k_global = k0 + k_local
                    W[k_global * n + i, r] = coeff

    # Flatten tensors to check rank
    # T_r = U[:, r] \otimes V[:, r] \otimes W[:, r]
    # We want to find the rank of the set {T_0, ..., T_27}
    
    # Create a matrix where each column is a flattened tensor
    # Shape: (nm*mp*pn, rank_28)
    # T[a,b,c] = U[a,r] * V[b,r] * W[c,r]
    
    # Efficient way: 
    # The tensor is a sum of rank-1 terms.
    # We can form the matrix of coefficients.
    
    # Let's just build the 1024 x 28 matrix
    T_flat = np.zeros((nm * mp * pn, rank_28), dtype=np.float64)
    
    for r in range(rank_28):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        # Kronecker product of u, v, w
        # np.kron(np.kron(u, v), w)
        vec = np.kron(np.kron(u, v), w)
        T_flat[:, r] = vec
        
        if time.time() - start_time > deadline:
            break

    # Compute rank
    # Use SVD or QR
    try:
        # np.linalg.matrix_rank uses SVD by default
        rank_val = np.linalg.matrix_rank(T_flat, tol=1e-10)
    except:
        rank_val = rank_28

    if rank_val < rank_28:
        # Find independent columns using QR with pivoting
        # Or simply use the null space of the transpose?
        # We want a subset of columns that are independent.
        
        # Use QR decomposition
        Q, R = np.linalg.qr(T_flat, mode='complete')
        # The rank is the number of non-zero diagonal elements of R
        # Since T_flat is (1024, 28), R is (28, 28)
        diag_R = np.abs(np.diag(R))
        # Find indices where diag_R > tol
        pivots = np.where(diag_R > 1e-10)[0]
        
        if len(pivots) == rank_val:
            U_red = U[:, pivots]
            V_red = V[:, pivots]
            W_red = W[:, pivots]
            
            # Verify
            T_target = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T_target[a, b, c] = 1.0
                        
            T_decomposed = np.zeros((nm, mp, pn))
            for r in range(len(pivots)):
                u = U_red[:, r]
                v = V_red[:, r]
                w = W_red[:, r]
                T_decomposed += np.einsum('a,b,c->abc', u, v, w)
                
            if np.allclose(T_decomposed, T_target, atol=1e-12):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    # Fallback to full rank 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
