"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    # U: 4 rows (2x2 A)
    # V: 16 rows (2x8 B)
    # W: 16 rows (8x2 C, index k*n+i)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for one 2x2x2 block
    # u: coeffs for A (a, b, c, d) -> indices 0,1,2,3
    # v: coeffs for B (e, f, g, h) -> local indices 0,1,2,3
    # w: coeffs for C (c00, c01, c10, c11) -> local indices 0,1,2,3
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # p1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # p2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]), # p3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]), # p4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]), # p5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]), # p6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # p7
    ]
    
    col = 0
    for t in range(4):
        if time.time() > deadline:
            break
            
        # Map local B indices (e,f,g,h) to global V indices
        v_map = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
        
        # Map local C indices (c00, c01, c10, c11) to global W indices
        w_map = [4*t, 4*t+2, 4*t+1, 4*t+3]
        
        for u, v, w in terms:
            # Update U
            for i in range(4):
                if u[i] != 0:
                    U[i][col] = u[i]
            
            # Update V
            for i in range(4):
                if v[i] != 0:
                    V[v_map[i]][col] = v[i]
            
            # Update W
            for i in range(4):
                if w[i] != 0:
                    W[w_map[i]][col] = w[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
