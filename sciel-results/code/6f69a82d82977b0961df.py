"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-30 tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Strategy:
    - Partition the 9 output columns into 3 pairs + 3 singles.
    - For each pair of columns (k1, k2), use Strassen's rank-7 algorithm on the
      2x2 * 2x2 sub-problem (treating the two columns as a 2x2 block).
    - For each single column, use the rank-4 schoolbook algorithm.
    - Total rank: 3*7 + 3*4 = 21 + 12 = 33.
    
    Wait, that's 33. Let me reconsider.
    
    Actually, let me think about this differently. The current approach uses:
    - 4 pairs: 4*7 = 28
    - 1 single: 4
    - Total: 32
    
    Can we do better? Let's try:
    - 3 pairs: 3*7 = 21
    - 3 singles: 3*4 = 12
    - Total: 33
    
    That's worse. So the 4 pairs + 1 single is better.
    
    Let me verify the Strassen decomposition is correct by checking the index conventions.
    
    Actually, let me re-examine the Strassen formula. For 2x2 * 2x2:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = A*B = [[c00, c01], [c10, c11]]
    
    Standard Strassen:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b01)
    M7 = (a01-a11)*(b10+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    In the current code, for a block with columns k1, k2:
    - c00 corresponds to output at (i=0, k=k1) -> W index k1*n+0
    - c01 corresponds to output at (i=0, k=k2) -> W index k2*n+0
    - c10 corresponds to output at (i=1, k=k1) -> W index k1*n+1
    - c11 corresponds to output at (i=1, k=k2) -> W index k2*n+1
    
    Let me check the W assignments in the current code:
    - M1: W[w_00][c] += 1, W[w_03][c] += 1  -> c00 and c11 get +M1
      But c00 should be M1+M4-M5+M7 and c11 should be M1-M2+M3+M6. OK.
    - M2: W[w_01][c] += 1, W[w_03][c] -= 1  -> c10 gets +M2, c11 gets -M2. OK.
    - M3: W[w_02][c] += 1, W[w_03][c] += 1  -> c01 gets +M3, c11 gets +M3. OK.
    - M4: W[w_00][c] += 1, W[w_01][c] += 1  -> c00 gets +M4, c10 gets +M4. OK.
    - M5: W[w_00][c] -= 1, W[w_02][c] += 1  -> c00 gets -M5, c01 gets +M5. OK.
    - M6: W[w_03][c] += 1  -> c11 gets +M6. OK.
    - M7: W[w_00][c] += 1  -> c00 gets +M7. OK.
    
    This looks correct. The current code should produce a valid rank-32 decomposition.
    
    Let me just rewrite it cleanly to ensure it's correct and distinct.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32
    
    def strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2 starting at column start_col."""
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00, u_01, u_10, u_11 = 0, 1, 2, 3
        
        c = start_col + 0
        U[u_00][c] += 1; U[u_11][c] += 1
        V[v_00][c] += 1; V[v_11][c] += 1
        W[w_00][c] += 1; W[w_03][c] += 1
        
        c = start_col + 1
        U[u_10][c] += 1; U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1; W[w_03][c] -= 1
        
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1; V[v_11][c] -= 1
        W[w_02][c] += 1; W[w_03][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1; V[v_00][c] -= 1
        W[w_00][c] += 1; W[w_01][c] += 1
        
        c = start_col + 4
        U[u_00][c] += 1; U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1; W[w_02][c] += 1
        
        c = start_col + 5
        U[u_10][c] += 1; U[u_00][c] -= 1
        V[v_00][c] += 1; V[v_01][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 6
        U[u_01][c] += 1; U[u_11][c] -= 1
        V[v_10][c] += 1; V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        """Add 4 schoolbook products for a single column k starting at column start_col."""
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        w_0 = k * n + 0
        w_1 = k * n + 1
        u_00, u_01, u_10, u_11 = 0, 1, 2, 3
        
        c = start_col + 0
        U[u_00][c] += 1; V[v_0][c] += 1; W[w_0][c] += 1
        
        c = start_col + 1
        U[u_01][c] += 1; V[v_1][c] += 1; W[w_0][c] += 1
        
        c = start_col + 2
        U[u_10][c] += 1; V[v_0][c] += 1; W[w_1][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1; V[v_1][c] += 1; W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
