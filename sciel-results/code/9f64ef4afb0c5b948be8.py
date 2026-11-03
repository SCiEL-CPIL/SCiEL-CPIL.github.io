import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # T[alpha, beta, gamma]
    # alpha: 0..8 (i*m+j)
    # beta: 0..11 (j*p+k)
    # gamma: 0..11 (k*n+i)
    
    # Precompute the target tensor as a sparse dict or dense array
    # Dense is 9x12x12 = 1296. Small enough.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    # We want to find U, V, W of rank R such that
    # sum_s U[alpha, s] V[beta, s] W[gamma, s] == T[alpha, beta, gamma]
    
    best_rank = n * m * p
    best_solution = None

    # Try ranks from low to high
    # Start lower than seed (16) up to schoolbook
    for rank in range(20, n * m * p + 1):
        found = False
        # Multiple trials
        for trial in range(100):
            np.random.seed(trial * 7919 + rank)
            # Initialize with random integers/half-integers
            # Values in {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            W = np.random.choice(vals, size=(p * n, rank))
            
            # Local Search
            # Objective: Sum of squared errors over all alpha, beta, gamma
            def get_error(U, V, W):
                # Compute tensor product sum
                # Reshape U to (nm, 1, 1, rank)
                # V to (1, mp, 1, rank)
                # W to (1, 1, pn, rank)
                # Sum over rank
                # Result shape (nm, mp, pn)
                U_exp = U[:, np.newaxis, np.newaxis, :]
                V_exp = V[np.newaxis, :, np.newaxis, :]
                W_exp = W[np.newaxis, np.newaxis, :, :]
                prod = U_exp * V_exp * W_exp
                res = np.sum(prod, axis=3)
                return np.sum((res - T) ** 2)

            err = get_error(U, V, W)
            
            # Simple hill climbing
            for it in range(500):
                if err < 1e-10:
                    break
                # Pick a random matrix and element to change
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    r, c = np.random.randint(0, n * m), np.random.randint(0, rank)
                    old_val = U[r, c]
                    U[r, c] = np.random.choice(vals)
                elif mat_idx == 1:
                    r, c = np.random.randint(0, m * p), np.random.randint(0, rank)
                    old_val = V[r, c]
                    V[r, c] = np.random.choice(vals)
                else:
                    r, c = np.random.randint(0, p * n), np.random.randint(0, rank)
                    old_val = W[r, c]
                    W[r, c] = np.random.choice(vals)
                
                new_err = get_error(U, V, W)
                if new_err < err:
                    err = new_err
                else:
                    # Revert
                    if mat_idx == 0: U[r, c] = old_val
                    elif mat_idx == 1: V[r, c] = old_val
                    else: W[r, c] = old_val

            if err < 1e-10:
                # Verify exactness
                U_exp = U[:, np.newaxis, np.newaxis, :]
                V_exp = V[np.newaxis, :, np.newaxis, :]
                W_exp = W[np.newaxis, np.newaxis, :, :]
                res = np.sum(U_exp * V_exp * W_exp, axis=3)
                if np.allclose(res, T, atol=1e-9):
                    best_rank = rank
                    best_solution = (U, V, W)
                    found = True
                    break
        
        if found:
            break

    if best_solution is not None:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
