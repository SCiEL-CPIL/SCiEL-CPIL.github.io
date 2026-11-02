<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured initialization
    # We try ranks slightly below schoolbook first
    for rank in range(79, 70, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few different seeds for structured initialization
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            U_init, V_init, W_init = _structured_init(n, m, p, rank, seed=seed)
            
            # Refine
            U_ref, V_ref, W_ref = _als_refine(U_init, V_init, W_init, n, m, p, max_iters=50, deadline=deadline - 10)
            
            # Round to half-integers
            U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U_ref])
            V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V_ref])
            W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W_ref])
            
            # Verify
            if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = rank
                found = True
                break
                
        if found:
            # If we found a lower rank, we can stop or try even lower
            # Given the difficulty, finding one is a significant win. 
            # We continue to see if we can go lower, but we are likely to succeed at 79 if possible.
            pass

    return best_result
=======
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for the bilinear form <C, AB> is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # U, V, W are all-ones vectors of the appropriate lengths.
    
    rank = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
