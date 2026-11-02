"""Rewritten program for matrix multiplication 2x3x3 with rank 17."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = 17
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    # Note: W index is k*n + i. k in 0..2, i in 0..1.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get U index: i*m + j
    def u_idx(i, j): return i * m + j
    # Helper to get V index: j*p + k
    def v_idx(j, k): return j * p + k
    # Helper to get W index: k*n + i
    def w_idx(k, i): return k * n + i
    
    # Strassen for top-left 2x2 (i=0,1; j=0,1; k=0,1)
    # A elements: a0=A(0,0), a1=A(0,1), a2=A(1,0), a3=A(1,1)
    # B elements: b0=B(0,0), b1=B(0,1), b2=B(1,0), b3=B(1,1)
    # C elements: C(0,0), C(0,1), C(1,0), C(1,1)
    
    # M1 = (a0+a3)(b0+b3)
    col = 0
    U[u_idx(0,0)][col] += 1
    U[u_idx(1,1)][col] += 1
    V[v_idx(0,0)][col] += 1
    V[v_idx(1,1)][col] += 1
    W[w_idx(0,0)][col] += 1  # C(0,0)
    W[w_idx(1,0)][col] -= 1  # C(1,0)
    W[w_idx(0,1)][col] += 1  # C(0,1)
    W[w_idx(1,1)][col] -= 1  # C(1,1)
    col += 1
    
    # M2 = (a2+a3)b0
    U[u_idx(1,0)][col] += 1
    U[u_idx(1,1)][col] += 1
    V[v_idx(0,0)][col] += 1
    W[w_idx(0,0)][col] -= 1  # C(0,0)
    W[w_idx(1,0)][col] += 1  # C(1,0)
    col += 1
    
    # M3 = a0(b1-b3)
    U[u_idx(0,0)][col] += 1
    V[v_idx(0,1)][col] += 1
    V[v_idx(1,1)][col] -= 1
    W[w_idx(0,1)][col] += 1  # C(0,1)
    W[w_idx(1,1)][col] += 1  # C(1,1)
    col += 1
    
    # M4 = a3(b2+b3)
    U[u_idx(1,1)][col] += 1
    V[v_idx(1,0)][col] += 1
    V[v_idx(1,1)][col] += 1
    W[w_idx(0,0)][col] += 1  # C(0,0)
    W[w_idx(1,0)][col] += 1  # C(1,0)
    W[w_idx(0,1)][col] -= 1  # C(0,1)
    W[w_idx(1,1)][col] += 1  # C(1,1)
    col += 1
    
    # M5 = (a0+a1)b3
    U[u_idx(0,0)][col] += 1
    U[u_idx(0,1)][col] += 1
    V[v_idx(1,1)][col] += 1
    W[w_idx(0,0)][col] -= 1  # C(0,0)
    W[w_idx(0,1)][col] += 1  # C(0,1)
    col += 1
    
    # M6 = (a2-a0)(b0+b1)
    U[u_idx(1,0)][col] += 1
    U[u_idx(0,0)][col] -= 1
    V[v_idx(0,0)][col] += 1
    V[v_idx(0,1)][col] += 1
    W[w_idx(1,1)][col] += 1  # C(1,1)
    col += 1
    
    # M7 = (a1-a3)(b2+b3)
    U[u_idx(0,1)][col] += 1
    U[u_idx(1,1)][col] -= 1
    V[v_idx(1,0)][col] += 1
    V[v_idx(1,1)][col] += 1
    W[w_idx(0,0)][col] += 1  # C(0,0)
    W[w_idx(1,1)][col] -= 1  # C(1,1)
    col += 1
    
    # Schoolbook for remaining terms
    # Terms involving A(0,2) [index 2] and B row 2 [index 2]
    # M8 = A(0,2) B(2,0) -> C(0,0)
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,0)][col] += 1
    W[w_idx(0,0)][col] += 1
    col += 1
    
    # M9 = A(0,2) B(2,1) -> C(0,1)
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,1)][col] += 1
    W[w_idx(0,1)][col] += 1
    col += 1
    
    # M10 = A(0,2) B(2,2) -> C(0,2)
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,2)][col] += 1
    W[w_idx(2,0)][col] += 1
    col += 1
    
    # Terms involving A(1,2) [index 5] and B row 2 [index 2]
    # M11 = A(1,2) B(2,0) -> C(1,0)
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,0)][col] += 1
    W[w_idx(0,1)][col] += 1
    col += 1
    
    # M12 = A(1,2) B(2,1) -> C(1,1)
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,1)][col] += 1
    W[w_idx(1,1)][col] += 1
    col += 1
    
    # M13 = A(1,2) B(2,2) -> C(1,2)
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,2)][col] += 1
    W[w_idx(2,1)][col] += 1
    col += 1
    
    # Terms involving B col 2 [index 2] and A row 0/1, B row 0/1
    # M14 = A(0,0) B(0,2) -> C(0,2)
    U[u_idx(0,0)][col] += 1
    V[v_idx(0,2)][col] += 1
    W[w_idx(2,0)][col] += 1
    col += 1
    
    # M15 = A(0,1) B(1,2) -> C(0,2)
    U[u_idx(0,1)][col] += 1
    V[v_idx(1,2)][col] += 1
    W[w_idx(2,0)][col] += 1
    col += 1
    
    # M16 = A(1,0) B(0,2) -> C(1,2)
    U[u_idx(1,0)][col] += 1
    V[v_idx(0,2)][col] += 1
    W[w_idx(2,1)][col] += 1
    col += 1
    
    # M17 = A(1,1) B(1,2) -> C(1,2)
    U[u_idx(1,1)][col] += 1
    V[v_idx(1,2)][col] += 1
    W[w_idx(2,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
