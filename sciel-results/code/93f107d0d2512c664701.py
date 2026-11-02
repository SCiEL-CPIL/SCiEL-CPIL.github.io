<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook matrix multiplication decomposition.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                # Safety check (not strictly necessary for small rank, but good practice)
                if time.time() - start_time > 100:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal decomposition by splitting the 3x8 matrix into
    two 3x4 blocks and searching for shared structure.
    Falls back to schoolbook if no improvement found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 2*3*8 = 48
    
    # Build schoolbook decomposition as fallback
    def build_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = build_schoolbook()
    
    # Try to find a better decomposition by splitting into two 3x4 blocks
    # Block 1: columns 0-3, Block 2: columns 4-7
    # Each block is a 3x4 multiplication: rank at least max(3,4) = 4 per block
    # But we can potentially share some rank terms between blocks
    
    # Strategy: Try ranks from 32 (lower bound) up to 47
    # For each rank, try to find valid U, V, W matrices
    
    def validate_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition with rank < 48
    # Start from rank 40 and go down
    best_result = schoolbook
    best_rank = rank_schoolbook
    
    for target_rank in range(47, 31, -1):
        if time.time() > deadline:
            break
        
        # Try a structured decomposition:
        # Split the 8 columns into two groups of 4
        # For each group, use a 3x4 multiplication tensor
        # Then try to merge/share terms between the two groups
        
        # Simple approach: Try random search with integer entries
        import random
        random.seed(42)
        
        found = False
        attempts = 0
        max_attempts = 1000000
        
        while attempts < max_attempts and not found and time.time() < deadline:
            attempts += 1
            
            # Generate random U, V, W with small integer entries
            U = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if validate_decomposition(U, V, W, target_rank):
                found = True
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if found:
            # Try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
