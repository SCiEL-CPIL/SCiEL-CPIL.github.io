<<<<<<< SEARCH
def _try_rank_with_search(rank, n, m, p, max_attempts, deadline, seed=42):
    """
    Try to find a valid decomposition of given rank using random search.
    Returns (U, V, W) or None.
    """
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random sparse patterns with entries in {-1, 0, 1}
        # U: 9 x rank, V: 12 x rank
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        
        # For U: each row has 2-4 nonzeros from {-1, 1}
        for i in range(n * m):
            num_nonzero = rng.randint(2, min(4, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                U[i, c] = 1 if rng.random() < 0.5 else -1
        
        # For V: each row has 2-4 nonzeros from {-1, 1}
        for i in range(m * p):
            num_nonzero = rng.randint(2, min(4, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                V[i, c] = 1 if rng.random() < 0.5 else -1
        
        # Try to solve for W over F2 (use absolute values mod 2)
        U_mod2 = (U + 2) % 2  # Convert -1,0,1 to 1,0,1
        V_mod2 = (V + 2) % 2
        
        W = _solve_w_over_f2(U_mod2, V_mod2, n, m, p, rank)
        if W is not None:
            # The binary solution works over F2, which implies it works over Integers (0/1)
            # Verify with the binary U, V, W
            if _verify_decomposition(U_mod2, V_mod2, W, n, m, p):
                return U_mod2, V_mod2, W
    
    return None
=======
def _try_rank_with_search(rank, n, m, p, max_attempts, deadline, seed=42):
    """
    Try to find a valid decomposition of given rank using random search.
    Returns (U, V, W) or None.
    """
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        
        # Generate structured sparse patterns
        # Each column of U and V should have some structure
        for t in range(rank):
            # For U column t: pick a subset of rows
            nz_u = rng.randint(2, min(5, n * m))
            cols_u = rng.sample(range(n * m), nz_u)
            for idx in cols_u:
                U[idx, t] = 1 if rng.random() < 0.5 else -1
            
            # For V column t: pick a subset of rows
            nz_v = rng.randint(2, min(5, m * p))
            cols_v = rng.sample(range(m * p), nz_v)
            for idx in cols_v:
                V[idx, t] = 1 if rng.random() < 0.5 else -1
        
        # Try direct integer solve
        W = _solve_w_direct(U, V, n, m, p, rank, deadline)
        if W is not None:
            return U, V, W
    
    return None
>>>>>>> REPLACE

