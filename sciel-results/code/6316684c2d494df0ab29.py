"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen Terms (r=0 to 6)
    # M1
    U[u_idx(0,0)][0] += 1; U[u_idx(1,1)][0] += 1
    V[v_idx(0,0)][0] += 1; V[v_idx(1,1)][0] += 1
    W[w_idx(0,0)][0] += 1; W[w_idx(1,0)][0] += 1; W[w_idx(0,1)][0] += -1; W[w_idx(1,1)][0] += 1
    
    # M2
    U[u_idx(1,0)][1] += 1; U[u_idx(1,1)][1] += 1
    V[v_idx(0,0)][1] += 1
    W[w_idx(0,1)][1] += 1; W[w_idx(1,1)][1] += -1
    
    # M3
    U[u_idx(0,0)][2] += 1
    V[v_idx(0,1)][2] += 1; V[v_idx(1,1)][2] += -1
    W[w_idx(1,0)][2] += 1; W[w_idx(1,1)][2] += 1
    
    # M4
    U[u_idx(1,1)][3] += 1
    V[v_idx(1,0)][3] += 1; V[v_idx(0,0)][3] += -1
    W[w_idx(0,0)][3] += 1; W[w_idx(0,1)][3] += 1
    
    # M5
    U[u_idx(0,0)][4] += 1; U[u_idx(0,1)][4] += 1
    V[v_idx(1,1)][4] += 1
    W[w_idx(0,0)][4] += -1; W[w_idx(1,1)][4] += 1
    
    # M6
    U[u_idx(1,0)][5] += 1; U[u_idx(0,0)][5] += -1
    V[v_idx(0,0)][5] += 1; V[v_idx(0,1)][5] += 1
    W[w_idx(1,1)][5] += 1
    
    # M7
    U[u_idx(0,1)][6] += 1; U[u_idx(1,1)][6] += -1
    V[v_idx(1,0)][6] += 1; V[v_idx(1,1)][6] += 1
    W[w_idx(0,0)][6] += 1; W[w_idx(1,1)][6] += -1

    # Schoolbook j in {0,1}, k=2 (r=7 to 10)
    U[u_idx(0,0)][7] += 1
    V[v_idx(0,2)][7] += 1
    W[w_idx(2,0)][7] += 1
    
    U[u_idx(0,1)][8] += 1
    V[v_idx(1,2)][8] += 1
    W[w_idx(2,0)][8] += 1
    
    U[u_idx(1,0)][9] += 1
    V[v_idx(0,2)][9] += 1
    W[w_idx(2,1)][9] += 1
    
    U[u_idx(1,1)][10] += 1
    V[v_idx(1,2)][10] += 1
    W[w_idx(2,1)][10] += 1

    # Schoolbook j=2, k in {0,1,2} (r=11 to 16)
    U[u_idx(0,2)][11] += 1
    V[v_idx(2,0)][11] += 1
    W[w_idx(0,0)][11] += 1
    
    U[u_idx(0,2)][12] += 1
    V[v_idx(2,1)][12] += 1
    W[w_idx(1,0)][12] += 1
    
    U[u_idx(0,2)][13] += 1
    V[v_idx(2,2)][13] += 1
    W[w_idx(2,0)][13] += 1
    
    U[u_idx(1,2)][14] += 1
    V[v_idx(2,0)][14] += 1
    W[w_idx(0,1)][14] += 1
    
    U[u_idx(1,2)][15] += 1
    V[v_idx(2,1)][15] += 1
    W[w_idx(1,1)][15] += 1
    
    U[u_idx(1,2)][16] += 1
    V[v_idx(2,2)][16] += 1
    W[w_idx(2,1)][16] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
