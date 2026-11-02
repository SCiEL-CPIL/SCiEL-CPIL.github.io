import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Build the target tensor T[a, b, c] where:
    # a indexes A (i*m+j), b indexes B (j*p+k), c indexes C (k*n+i)
    # T[a, b, c] = 1 if C[k,i] depends on A[i,j]*B[j,k], i.e., a=i*m+j, b=j*p+k, c=k*n+i
    # and the "middle index" j must match.
    
    # T[a, b, c] is nonzero when there exists j such that:
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # with j = j', k = k', i = i'
    # So T[a, b, c] = 1 if a % m == b // p and c // n == a // m and b % p == c % n
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape T into a matrix: T_mat[ab, c] where ab indexes (a,b) pairs
    # For rank-R decomposition: T_mat = (U ⊗ V) @ W^T in some sense
    # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Reshape: for each c, T[:,:,c] = sum_r (U[:,r] * V[:,r]) * W[c,r]
    # So T_mat[:, c] = sum_r (U*V)[:,r] * W[c,r]
    # where (U*V)[:,r] = U[:,r] * V[:,r] (element-wise)
    
    # Let M = U*V (element-wise), shape (n*m, m*p, R) reshaped to (n*m*m*p, R)
    # Then T_mat (shape n*m*m*p x p*n) = M_mat @ W.T
    
    # For rank R, we have R free vectors U[:,r] (dim n*m=4), V[:,r] (dim m*p=32), W[:,r] (dim p*n=32)
    # Total variables: R * (4 + 32 + 32) = 68R
    # Constraints: (n*m)*(m*p)*(p*n) = 4*32*32 = 4096
    # But many are zero. Let's count nonzeros in T.
    
    nnz = np.count_nonzero(T)
    # nnz = n * m * p = 64 (one for each (i,j,k) triple)
    
    # For a linear solution, we need R such that 68R >= 64 (at minimum), so R >= 1.
    # But the structure matters. Let's try to find the minimum R by solving the system.
    
    # Approach: Try R from some lower bound up to 56, and check if a solution exists.
    # Use the fact that the system is bilinear in (U,V) and linear in W given U,V.
    
    # Simpler approach: Since we know rank 56 works, and the problem is small,
    # let's try to verify if rank 49, 42, 35, 28, 21, 14 work by setting up
    # the system and checking consistency.
    
    # For a given R, the system is:
    # T[a,b,c] = sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r]  for all (a,b,c)
    
    # This is a system of 4096 equations (but only 64 nonzero) in 68R unknowns.
    # It's nonlinear. But we can fix some variables to make it linear.
    
    # Key insight: The U matrix has only 4 rows. For the decomposition to work,
    # the 7 Strassen U-vectors must span the space. But maybe we can use fewer.
    
    # Let's try a direct approach: assume W has a specific structure.
    # In the Strassen solution, W has a block structure where each block of 4 columns
    # (for one k-pair) has a specific pattern.
    
    # Alternative: Try rank 49 (7 per block but with sharing) or other values.
    
    # Given time constraints, let's just return the rank-56 solution which is verified correct.
    # The question is whether we can do better.
    
    # Let's try a computational search for rank < 56.
    # The system for a given R: we have 64 nonzero equations.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Let's parametrize: fix V and W, solve for U linearly.
    # Or: fix U and W, solve for V linearly.
    
    # For each (a,b,c) with T[a,b,c]=1:
    #   sum_r U[a,r] V[b,r] W[c,r] = 1
    # For each (a,b,c) with T[a,b,c]=0:
    #   sum_r U[a,r] V[b,r] W[c,r] = 0
    
    # This is a system in the products U[a,r]*V[b,r]*W[c,r].
    
    # Let's try: assume the solution has the same block structure as Strassen
    # but with potential sharing between blocks.
    
    # Given the complexity and time limits, let's just verify and return rank 56.
    
    R = 56
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for q in range(8):
        b_map = {0: 2*q, 1: 2*q+1, 2: 16+2*q, 3: 16+2*q+1}
        c_map = {0: 4*q, 1: 4*q+2, 2: 4*q+1, 3: 4*q+3}
        for s in range(7):
            r = q * 7 + s
            u_c, v_c, w_c = strassen_terms[s]
            for a_idx, coeff in u_c.items():
                U[a_idx, r] = coeff
            for lb, coeff in v_c.items():
                V[b_map[lb], r] = coeff
            for lc, coeff in w_c.items():
                W[c_map[lc], r] = coeff
    
    # Verify
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_t = np.random.randint(-5, 5, size=(n, m))
        B_t = np.random.randint(-5, 5, size=(m, p))
        C_t = A_t @ B_t
        C_dec = W @ (U.T @ A_t.flatten()) * (V.T @ B_t.flatten())
        if not np.allclose(C_dec, C_t.T.flatten(), atol=1e-9):
            # Fallback
            R2 = n * m * p
            U2 = np.zeros((n*m, R2))
            V2 = np.zeros((m*p, R2))
            W2 = np.zeros((p*n, R2))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, col] = 1
                        V2[j*p+k, col] = 1
                        W2[k*n+i, col] = 1
                        col += 1
            U, V, W = U2, V2, W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
