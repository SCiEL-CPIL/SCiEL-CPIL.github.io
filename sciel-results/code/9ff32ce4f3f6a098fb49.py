"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    # Rank 39 solution
    # 5 blocks of 2x2 (Strassen rank 7) + 1 block of 2x1 (Standard rank 4)
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(b_start):
        nonlocal col
        # M1 = (A00+A11)(B00+B10)
        U[0][col] += 1; U[3][col] += 1
        V[b_start][col] += 1; V[11+b_start][col] += 1
        W[2*b_start][col] = 1; W[2*b_start+1][col] = 1
        W[2*b_start+2][col] = -1; W[2*b_start+3][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[b_start][col] += 1
        W[2*b_start+2][col] = 1; W[2*b_start+3][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[b_start+1][col] += 1; V[11+b_start+1][col] -= 1
        W[2*b_start+1][col] = 1; W[2*b_start+3][col] = -1
        col += 1
        
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[11+b_start][col] += 1; V[b_start][col] -= 1
        W[2*b_start][col] = -1; W[2*b_start+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B10
        U[0][col] += 1; U[1][col] += 1
        V[11+b_start][col] += 1
        W[2*b_start][col] = 1; W[2*b_start+2][col] = -1
        col += 1
        
        # M6 = (A10+A11)(B11-B01)
        U[2][col] += 1; U[3][col] += 1
        V[11+b_start+1][col] += 1; V[b_start+1][col] -= 1
        W[2*b_start+2][col] = 1; W[2*b_start+3][col] = -1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[b_start+1][col] += 1; V[11+b_start+1][col] += 1
        W[2*b_start][col] = -1; W[2*b_start+1][col] = 1
        col += 1

    for b in range(5):
        if time.time() > deadline:
            break
        add_strassen_block(2*b)
        
    # Last block: column 10
    k = 10
    if time.time() > deadline:
        # Return what we have if deadline exceeded? 
        # But we must return valid U,V,W of size rank.
        # If we break early, col < rank.
        # We should just complete it, it's fast.
        pass

    U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
    U[1][col] = 1; V[11+k][col] = 1; W[2*k][col] = 1; col += 1
    U[2][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
    U[3][col] = 1; V[11+k][col] = 1; W[2*k+1][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
