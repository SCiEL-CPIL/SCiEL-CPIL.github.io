<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Default: schoolbook rank 50
    best = _schoolbook()
    best_rank = 50
    
    # Try randomized search for smaller ranks
    for r in range(5, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Check for Rank 1 solution (All-Ones Tensor)
    # U, V, W all filled with 1s
    U_1 = [[1] * 1 for _ in range(n * m)]
    V_1 = [[1] * 1 for _ in range(m * p)]
    W_1 = [[1] * 1 for _ in range(p * n)]
    
    if _verify(U_1, V_1, W_1, n, m, p):
        return {"U": U_1, "V": V_1, "W": W_1}
    
    # Default: schoolbook rank 50
    best = _schoolbook()
    best_rank = 50
    
    # Try randomized search for smaller ranks
    for r in range(2, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, deadline)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
>>>>>>> REPLACE
