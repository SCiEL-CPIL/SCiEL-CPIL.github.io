"""Matrix multiplication 2x5x5 rank decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (10, rank)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    U = []
    V = []
    W = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add a rank-1 term. u_coeffs: dict (i,j)->val, v_coeffs: dict (j,k)->val, w_coeffs: dict (k,i)->val"""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for (i, j), val in u_coeffs.items():
            u[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            v[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            w[k * n + i] = val
        U.append(u)
        V.append(v)
        W.append(w)
    
    def add_strassen_2x2x2(i_vals, j_vals, k_vals):
        """Add Strassen 2×2×2 block.
        i_vals = (i0, i1): row indices for A and C
        j_vals = (j0, j1): col indices for A, row indices for B
        k_vals = (k0, k1): col indices for B, col indices for C
        """
        i0, i1 = i_vals
        j0, j1 = j_vals
        k0, k1 = k_vals
        
        # Term 1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        # W: C[i0,k0] += 1, C[i1,k1] += 1
        add_term(
            {(i0,j0):1, (i1,j1):1},
            {(j0,k0):1, (j1,k1):1},
            {(k0,i0):1, (k1,i1):1}
        )
        
        # Term 2: (A[i1,j0]+A[i1,j1])B[j0,k0]
        # W: C[i1,k0] += 1
        add_term(
            {(i1,j0):1, (i1,j1):1},
            {(j0,k0):1},
            {(k0,i1):1}
        )
        
        # Term 3: A[i0,j0](B[j1,k0]-B[j0,k0])
        # W: C[i0,k1] += 1
        add_term(
            {(i0,j0):1},
            {(j1,k0):1, (j0,k0):-1},
            {(k1,i0):1}
        )
        
        # Term 4: A[i1,j1](B[j0,k1]-B[j1,k1])
        # W: C[i0,k0] += -1, C[i0,k1] += 1
        add_term(
            {(i1,j1):1},
            {(j0,k1):1, (j1,k1):-1},
            {(k0,i0):-1, (k1,i0):1}
        )
        
        # Term 5: (A[i0,j0]+A[i0,j1])B[j1,k1]
        # W: C[i0,k0] += 1, C[i1,k0] += 1
        add_term(
            {(i0,j0):1, (i0,j1):1},
            {(j1,k1):1},
            {(k0,i0):1, (k0,i1):1}
        )
        
        # Term 6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j1,k0])
        # W: C[i1,k0] += 1  (need to verify)
        add_term(
            {(i1,j0):1, (i0,j0):-1},
            {(j0,k0):1, (j1,k0):1},
            {(k0,i1):1}
        )
        
        # Term 7: (A[i0,j1]-A[i1,j1])(B[j0,k1]+B[j1,k1])
        # W: C[i0,k0] += -1, C[i1,k1] += 1  (need to verify)
        add_term(
            {(i0,j1):1, (i1,j1):-1},
            {(j0,k1):1, (j1,k1):1},
            {(k0,i0):-1, (k1,i1):1}
        )
    
    def add_schoolbook_2x2x1(i_vals, j_vals, k_val):
        """Add schoolbook 2×2×1 block.
        i_vals = (i0, i1), j_vals = (j0, j1), k_val = k
        """
        i0, i1 = i_vals
        j0, j1 = j_vals
        k = k_val
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term(
                    {(i,j):1},
                    {(j,k):1},
                    {(k,i):1}
                )
    
    def add_schoolbook_2x1x5(i_vals, j_val, k_vals):
        """Add schoolbook 2×1×5 block.
        i_vals = (i0, i1), j_val = j, k_vals = (k0,...,k4)
        """
        i0, i1 = i_vals
        j = j_val
        for i in [i0, i1]:
            for k in k_vals:
                add_term(
                    {(i,j):1},
                    {(j,k):1},
                    {(k,i):1}
                )
    
    # Group 1: j in {0,1}
    # k in {0,1}: Strassen
    add_strassen_2x2x2((0,1), (0,1), (0,1))
    # k in {2,3}: Strassen
    add_strassen_2x2x2((0,1), (0,1), (2,3))
    # k = 4: schoolbook 2x2x1
    add_schoolbook_2x2x1((0,1), (0,1), 4)
    
    # Group 2: j in {2,3}
    # k in {0,1}: Strassen
    add_strassen_2x2x2((0,1), (2,3), (0,1))
    # k in {2,3}: Strassen
    add_strassen_2x2x2((0,1), (2,3), (2,3))
    # k = 4: schoolbook 2x2x1
    add_schoolbook_2x2x1((0,1), (2,3), 4)
    
    # Group 3: j = 4
    # schoolbook 2x1x5
    add_schoolbook_2x1x5((0,1), 4, (0,1,2,3,4))
    
    rank = len(U)
    
    # Convert to list of lists
    U_out = [row[:] for row in zip(*U)] if U else []
    V_out = [row[:] for row in zip(*V)] if V else []
    W_out = [row[:] for row in zip(*W)] if W else []
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
