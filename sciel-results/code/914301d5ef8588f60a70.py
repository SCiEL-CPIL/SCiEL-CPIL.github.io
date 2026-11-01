# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 10.0 # 10 seconds is plenty for this
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Note: The tensor rank of 2x2 matrix multiplication is proven to be 7 (Strassen).
    # Attempts to find a rank-6 decomposition using Heisenberg Lie algebra structure constants
    # fail to satisfy the Jacobi identity constraints required for exact bilinear multiplication.
    # Thus, rank 7 is the minimal exact rank.
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen's Algorithm for 2x2 multiplication
    # A is 2x2, B is 2x2
    # Indices:
    # U row index: i*m + j  (i in 0..1, j in 0..1)
    # V row index: j*p + k  (j in 0..1, k in 0..1)
    # W row index: k*n + i  (k in 0..1, i in 0..1) -> Note: W depends on k and i, NOT j.
    # The sum is over j and r.
    # Result term for C[i, k] is sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    
    # Strassen Factors:
    # u, v, w are vectors of coefficients for A, B, and output C[i,k] respectively.
    
    factors = [
        # (u_coords, u_vals, v_coords, v_vals, w_coords, w_vals)
        # u_coords: list of (i, j), v_coords: list of (j, k), w_coords: list of (k, i)
        # M1
        ([ (0,0), (1,1) ], [1, 1], 
         [ (0,0), (1,1) ], [1, 1], 
         [ (0,0), (1,1) ], [1, 1]),
        # M2
        ([ (1,0), (1,1) ], [1, 1], 
         [ (0,0) ], [1], 
         [ (0,1), (1,1) ], [1, -1]),
        # M3
        ([ (0,0) ], [1], 
         [ (0,1), (1,1) ], [1, -1], 
         [ (1,0), (1,1) ], [1, 1]),
        # M4
        ([ (1,1) ], [1], 
         [ (1,0), (0,0) ], [1, -1], 
         [ (0,0), (0,1) ], [1, 1]),
        # M5
        ([ (0,0), (0,1) ], [1, 1], 
         [ (1,1) ], [1], 
         [ (0,0), (1,0) ], [-1, 1]),
        # M6
        ([ (1,0), (0,0) ], [1, -1], 
         [ (0,0), (0,1) ], [1, 1], 
         [ (1,1) ], [1]),
        # M7
        ([ (0,1), (1,1) ], [1, -1], 
         [ (1,0), (1,1) ], [1, 1], 
         [ (0,0) ], [1]),
    ]
    
    for r, (u_c, u_v, v_c, v_v, w_c, w_v) in enumerate(factors):
        for (i, j), val in zip(u_c, u_v):
            U[i*m + j, r] = val
        for (j, k), val in zip(v_c, v_v):
            V[j*p + k, r] = val
        for (k, i), val in zip(w_c, w_v):
            W[k*n + i, r] = val
            
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > deadline - 0.1:
         return {"U": U_list, "V": V_list, "W": W_list}
         
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
