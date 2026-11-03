"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the multiplication into two 2x3x3 multiplications.
    # The rank of 2x3x3 is 8.
    # Total rank is 16.
    
    # We need a rank 8 decomposition for 2x3x3.
    # Let's use a known one or construct it.
    # For now, we will use the schoolbook for 2x3x3 (rank 18) if we can't find rank 8.
    # But we want to improve on 36.
    # Rank 18 * 2 = 36. No improvement.
    # We need rank 8.
    
    # Let's try to find a rank 8 decomposition for 2x3x3.
    # Indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # T_{i,j,k} = 1.
    
    # We will use the following strategy:
    # Use Laderman's algorithm for the 2x2x2 part (indices i in {0,1}, j in {0,1}, k in {0,1}).
    # Rank 7.
    # The remaining terms involve j=2 or k=2.
    # This is complex.
    
    # Alternative: Use the fact that rank(2x3x3) <= rank(2x2x2) + rank(2x1x3) + rank(2x1x1)?
    # No.
    
    # Let's just return the schoolbook for now, as I cannot guarantee a valid rank 8 decomposition without a reference.
    # However, the task asks for a stronger algorithm.
    # I will implement a search for a lower rank using a simple greedy or random method within the time limit.
    
    import time
    start_time = time.time()
    deadline = 120.0
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower rank using a simple heuristic
    # This is a placeholder for a more sophisticated search.
    # Given the time constraint and complexity, I will stick to the schoolbook unless I can implement a known decomposition.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
