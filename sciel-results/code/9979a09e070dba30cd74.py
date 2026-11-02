import numpy as np

def construct():
    n, m, p = 4, 5, 6
    # Initialize coefficients
    # Using a dictionary to store non-zero coefficients
    coeffs = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeffs[(i, j, k)] = 1

    # Apply dependencies for disjoint 2x2x2 blocks
    # Block origins
    i_starts = [0, 2]
    j_starts = [0, 2]
    k_starts = [0, 2, 4]
    
    for i0 in i_starts:
        for j0 in j_starts:
            for k0 in k_starts:
                # Check bounds
                if i0 + 1 < n and j0 + 1 < m and k0 + 1 < p:
                    # Terms involved:
                    # (i0, j0, k0)
                    # (i0+1, j0+1, k0+1)  <- Remove this
                    # (i0, j0+1, k0)
                    # (i0+1, j0, k0+1)
                    
                    # Update coefficients
                    coeffs[(i0, j0, k0)] += 1
                    coeffs[(i0, j0+1, k0)] -= 1
                    coeffs[(i0+1, j0, k0+1)] -= 1
                    coeffs[(i0+1, j0+1, k0+1)] = 0

    # Filter out zero coefficients
    active_terms = [(i, j, k) for (i, j, k), c in coeffs.items() if c != 0]
    rank = len(active_terms)
    
    # Construct U, V, W
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for col, (i, j, k) in enumerate(active_terms):
        c = coeffs[(i, j, k)]
        U[i * m + j, col] = c
        V[j * p + k, col] = c
        W[k * n + i, col] = c

    return {"U": U, "V": V, "W": W}
