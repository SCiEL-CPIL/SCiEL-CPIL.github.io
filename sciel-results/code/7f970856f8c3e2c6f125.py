"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100.0
    n, m, p = DIMENSIONS
    
    # Structural approach: Use 2 blocks of 4 columns each
    # Each block: A (2x3) * B_block (3x4) -> C_block (2x4)
    # For each block, use a rank-18 decomposition (schoolbook is 24, but we can do better)
    # Actually, let's use a different strategy: exploit the rectangular shape
    
    # Alternative: use the fact that 2x3x8 can be decomposed using
    # rank = rank(2x3x2) * 4 if we can find a good 2x3x2 decomposition
    
    # Let's try to find a rank-10 or rank-9 decomposition for 2x3x2
    # and see if we can extend it
    
    # For now, use a hybrid approach:
    # - First 2 columns: use rank-11 (Strassen-like)
    # - Next 2 columns: use rank-11
    # - Next 2 columns: use rank-11  
    # - Last 2 columns: use rank-11
    # Total: 44 (same as before)
    
    # Instead, let's try a completely different structure:
    # Decompose B as B = [B1, B2, B3, B4] where each Bi is 3x2
    # But allow cross-talk between blocks using shared U factors
    
    # New idea: use a rank-42 decomposition by sharing 2 terms across blocks
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use 3 blocks of 11 terms + 1 block of 9 terms = 42
    # Or use a more clever sharing
    
    # Actually, let's try a rank-40 approach:
    # Use 4 blocks, but share 1 term per block with the next block
    
    # This is getting complex. Let me just return the known rank-44 solution
    # but with a cleaner implementation
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for q in range(4):
        v_b00 = 0 * 8 + 2 * q
        v_b01 = 0 * 8 + 2 * q + 1
        v_b10 = 1 * 8 + 2 * q
        v_b11 = 1 * 8 + 2 * q + 1
        v_b20 = 2 * 8 + 2 * q
        v_b21 = 2 * 8 + 2 * q + 1
        
        w_c00 = 4 * q
        w_c01 = 4 * q + 2
        w_c10 = 4 * q + 1
        w_c11 = 4 * q + 3
        
        # 11 terms per block
        U[0][col] = 1; U[4][col] = 1
        V[v_b00][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1; W[w_c11][col] = 1
        col += 1
        
        U[3][col] = 1; U[4][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1; W[w_c11][col] = -1
        col += 1
        
        U[0][col] = 1
        V[v_b01][col] = 1; V[v_b11][col] = -1
        W[w_c01][col] = 1; W[w_c11][col] = 1
        col += 1
        
        U[4][col] = 1
        V[v_b10][col] = 1; V[v_b00][col] = -1
        W[w_c00][col] = 1; W[w_c10][col] = 1
        col += 1
        
        U[0][col] = 1; U[1][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1; W[w_c01][col] = 1
        col += 1
        
        U[3][col] = 1; U[0][col] = -1
        V[v_b00][col] = 1; V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        U[1][col] = 1; U[4][col] = -1
        V[v_b10][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
        
        U[2][col] = 1
        V[v_b20][col] = 1
        W[w_c00][col] = 1
        col += 1
        
        U[2][col] = 1
        V[v_b21][col] = 1
        W[w_c01][col] = 1
        col += 1
        
        U[5][col] = 1
        V[v_b20][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        U[5][col] = 1
        V[v_b21][col] = 1
        W[w_c11][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
