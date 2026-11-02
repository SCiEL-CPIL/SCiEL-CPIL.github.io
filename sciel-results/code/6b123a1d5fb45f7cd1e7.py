"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a bilinear decomposition for (2,5,5) matrix multiplication.
    We construct the exact rank-n*m*p schoolbook decomposition directly by
    enumerating the basis of the input/output spaces. The decomposition uses
    U, V of shape (n*m, rank), (m*p, rank) and W of shape (p*n, rank) such that
    for each c in range(rank):
        sum_{a,b} U[a,c] * (A_{row_a, col_b} B_{row_b, col_k}) * W[k*n+row_a, c]
    yields the correct (i,k)-entry.
    """
    import itertools
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    # Schoolbook basis enumeration. Each (i,j,k) contributes a rank-1 tensor.
    # We store the matrices as Python lists of lists to be JSON-serializable.
    rank = 0
    U_rows = [[0] * rank for _ in range(N)]
    V_rows = [[0] * rank for _ in range(M)]
    W_rows = [[0] * rank for _ in range(P)]

    # For each (i,j,k) we add a new rank-1 term.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = rank
                # Extend matrices by one column
                for r in U_rows:
                    r.append(0)
                for r in V_rows:
                    r.append(0)
                for r in W_rows:
                    r.append(0)

                U_rows[i * m + j][idx] = 1
                V_rows[j * p + k][idx] = 1
                W_rows[k * n + i][idx] = 1
                rank += 1

    return {"U": U_rows, "V": V_rows, "W": W_rows}

# EVOLVE-BLOCK-END
