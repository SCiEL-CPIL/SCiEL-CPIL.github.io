import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    # U index: i*m + j  (i in [0,n), j in [0,m))
    # V index: j*p + k  (j in [0,m), k in [0,p))
    # W index: k*n + i  (k in [0,p), i in [0,n))
    
    # We'll build a list of terms. Each term is (u_indices, v_indices, w_indices)
    # where each is a list of (flat_index, coeff).
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Part 1: 2x2x2 Winograd for C[0:2, 0:2] = A[0:2,0:2] * B[0:2,0:2]
    # A block rows 0-1, cols 0-1. B block rows 0-1, cols 0-1.
    # Winograd 7 multiplications:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. Each entry is a list of (flat_idx, coeff)."""
        terms.append((u_entries, v_entries, w_entries))
    
    # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] and A[1,1] -> flat: 0*4+0=0, 1*4+1=5
    # V: B[0,0] and B[1,1] -> flat: 0*5+0=0, 1*5+1=6
    # W: C[0,0] and C[1,1] -> flat: 0*4+0=0, 1*4+1=5
    add_term(
        [(0, 1), (5, 1)],
        [(0, 1), (6, 1)],
        [(0, 1), (5, 1)]
    )
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    # U: A[1,0]=1*4+0=4, A[1,1]=1*4+1=5
    # V: B[0,0]=0*5+0=0
    # W: C[1,0]=1*4+0=4
    add_term(
        [(4, 1), (5, 1)],
        [(0, 1)],
        [(4, 1)]
    )
    
    # M3: A[0,0] * (B[1,0]-B[1,1])
    # U: A[0,0]=0
    # V: B[1,0]=1*5+0=5, B[1,1]=1*5+1=6
    # W: C[0,1]=0*4+1=1
    add_term(
        [(0, 1)],
        [(5, 1), (6, -1)],
        [(1, 1)]
    )
    
    # M4: A[1,1] * (B[0,1]-B[0,0])
    # U: A[1,1]=5
    # V: B[0,1]=0*5+1=1, B[0,0]=0
    # W: C[0,0]=0, C[1,0]=4
    # C[0,0] gets +M4, C[1,0] gets +M4
    add_term(
        [(5, 1)],
        [(1, 1), (0, -1)],
        [(0, 1), (4, 1)]
    )
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    # U: A[0,0]=0, A[0,1]=0*4+1=1
    # V: B[1,1]=6
    # W: C[0,0] gets -M5, C[0,1] gets +M5
    add_term(
        [(0, 1), (1, 1)],
        [(6, 1)],
        [(0, -1), (1, 1)]
    )
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # U: A[1,0]=4, A[0,0]=0
    # V: B[0,0]=0, B[0,1]=1
    # W: C[1,1] gets +M6
    add_term(
        [(4, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(5, 1)]
    )
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # U: A[0,1]=1, A[1,1]=5
    # V: B[1,0]=5, B[1,1]=6
    # W: C[0,0] gets +M7
    add_term(
        [(1, 1), (5, -1)],
        [(5, 1), (6, 1)],
        [(0, 1)]
    )
    
    # Part 2: Schoolbook for C[2:4, 0:2] = A[2:4,0:4] * B[0:2,0:2]
    # 8 terms: for each i in {2,3}, j in {0,1,2,3}, k in {0,1}
    for i in range(2, 4):
        for j in range(4):
            for k in range(2):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # Part 3: Schoolbook for C[0:4, 2:5] = A[0:4,0:4] * B[0:4,2:5]
    # 12 terms: for each i in {0,1,2,3}, j in {0,1,2,3}, k in {2,3,4}
    for i in range(4):
        for j in range(4):
            for k in range(2, 5):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    rank = len(terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, coeff in u_entries:
            U[idx, r] = coeff
        for idx, coeff in v_entries:
            V[idx, r] = coeff
        for idx, coeff in w_entries:
            W[idx, r] = coeff
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 for all valid (i,j,k)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Let's verify with random matrices
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute via tensor decomposition
    # C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
