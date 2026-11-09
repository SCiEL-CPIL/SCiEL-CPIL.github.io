# Part 1: A2 (col 2) * B2 (row 2)
# Rank 10
for i in range(2):
    for k in range(5):
        u_idx = i * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
