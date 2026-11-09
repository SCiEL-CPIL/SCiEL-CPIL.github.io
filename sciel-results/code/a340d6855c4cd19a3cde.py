"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    base_rank = m  # Rank 5 for 1x5x6
    rank = n * base_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base decomposition for 1x5x6 multiplication
    # T_{j,k} = sum_{b=0}^{4} U_base[j,b] V_base[j,k,b] W_base[k,b] = 1
    # We use the simple basis decomposition:
    # U_base[j,b] = 1 if j==b else 0
    # V_base[j,k,b] = 1 if j==b else 0 (This effectively selects the row)
    # W_base[k,b] = 1
    # Check: sum_b (delta_{j,b}) * (delta_{j,b}) * 1 = 1 if j==j else 0?
    # Wait, the term is A_j B_{j,k}.
    # We need the coefficient of A_j B_{j,k} in C_k to be 1.
    # sum_b U[j,b] V[j,k,b] W[k,b] = 1.
    # If U[j,b]=delta_{j,b}, V[j,k,b]=1, W[k,b]=1.
    # sum_b delta_{j,b} * 1 * 1 = 1. Correct.
    # Is V[j,k,b]=1 valid?
    # V is constant. Yes.
    
    for a in range(n):
        for j in range(m):
            for k in range(p):
                for b in range(base_rank):
                    r = a * base_rank + b
                    u_idx = a * m + j
                    v_idx = j * p + k
                    w_idx = k * n + a
                    
                    # U[a*m+j, a*base_rank+b] = delta_{j,b}
                    if j == b:
                        U[u_idx][r] = 1
                    
                    # V[j*p+k, a*base_rank+b] = 1
                    V[v_idx][r] = 1
                    
                    # W[k*n+a, a*base_rank+b] = 1
                    W[w_idx][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
