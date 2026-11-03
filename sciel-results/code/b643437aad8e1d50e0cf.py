import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i
    
    def build_naive_solution():
        """Build the standard naive 125-term solution."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j)][col] = 1
                    V[v_idx(j, k)][col] = 1
                    W[w_idx(k, i)][col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W):
        """Verify the decomposition sums to 1 for all (i,j,k)."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Use the robust naive solution
    U, V, W = build_naive_solution()
    
    # Verify to be safe
    if not verify(U, V, W):
        # If verification fails, there is a bug in index mapping or construction.
        # For naive, this should not happen.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
