<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # Use 5 independent 2x2x2 Strassen blocks: rank = 5 * 7 = 35
    num_blocks = p // 2  # 5
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
=======
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    t0 = time.time()
    
    # Try rank-27: 2x2x4 (rank 13) + 2x2x2 (rank 7) + 2x2x2 (rank 7)
    result_27 = _build_2x2x4_rank13()
    
    if result_27 is not None:
        U_13, V_13, W_13 = result_27
        # Build the full 2x2x10 decomposition
        # First 4 columns use the rank-13 block, last 6 columns use two rank-7 blocks
        rank = 13 + 7 + 7  # 27
        
        U = [[0] * rank for _ in range(n * m)]   # 4 x 27
        V = [[0] * rank for _ in range(m * p)]   # 20 x 27
        W = [[0] * rank for _ in range(p * n)]   # 20 x 27
        
        # Fill in the rank-13 part for columns 0-3
        for r in range(13):
            for i in range(n * m):
                U[i][r] = U_13[i, r]
            for j in range(m * 4):
                V[j][r] = V_13[j, r]
            for k in range(4 * n):
                W[k][r] = W_13[k, r]
        
        # Fill in two rank-7 Strassen blocks for columns 4-5 and 6-7... 
        # Wait, p=10, so columns 4-9 (6 columns = 3 blocks of 2)
        # Actually: 4 columns in first block, then 6 more = 3 blocks of 2
        # But we only have 7+7=14 more terms, so 2 blocks of 2x2x2
        # That's 4 columns, total 8. We need 10.
        # So: 2x2x4 (rank 13) + 2x2x2 (rank 7) + 2x2x2 (rank 7) = 13+7+7 = 27
        # Covers 4 + 2 + 2 = 8 columns. We need 10.
        # Hmm, that only covers 8 columns. We need 10.
        # So we need: 2x2x4 (rank 13) + 2x2x6? Or 2x2x4 + 2x2x2 + 2x2x4?
        # 13 + 7 + 13 = 33, covers 4+2+4=10. That works!
        # Or: 2x2x2 (rank 7) * 5 = 35 (current)
        # Or: 2x2x4 (rank 13) + 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x2 (rank 7) = 13+7+7+7=34, covers 4+2+2+2=10
        # Or: 2x2x4 (rank 13) + 2x2x4 (rank 13) + 2x2x2 (rank 7) = 13+13+7=33, covers 4+4+2=10
        
        # Let's use: 2x2x4 (rank 13) + 2x2x4 (rank 13) + 2x2x2 (rank 7) = 33
        # Or try to get 2x2x6 with rank 19? R(2,2,6) might be 19.
        # For now, let's just use the safe 35 and return early if we can't do better.
        
        pass  # Fall through to 35-rank approach
    
    # Safe fallback: 5 independent 2x2x2 Strassen blocks: rank = 5 * 7 = 35
    num_blocks = p // 2  # 5
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
>>>>>>> REPLACE

Actually, I realize the rank-13 approach for 2×2×4 is complex and may not yield a valid decomposition easily. Let me take a different approach: try to find a rank-27 decomposition for the full 2×2×10 tensor using numerical optimization, with a time budget.

Let me reconsider. The current program scores 1.1429 = 40/35. To improve, I need rank < 35. The theoretical minimum for 2×2×10 is at least 2×10=20 (trivial lower bound) and at most 35 (Strassen blocks). Known results suggest R(2,2,n) = 3n+1 for small n, so R(2,2,10) might be 31.

Let me try a direct numerical approach to find a rank-31 or rank-27 decomposition:

