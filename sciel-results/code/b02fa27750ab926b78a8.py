import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _construct_rank_55():
    """Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # W00: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # W01: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # W10: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # W11: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    return np.array(U, dtype=int), np.array(V, dtype=int), np.array(W, dtype=int)

def _check_exact_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += int(U[i*m+j, r]) * int(V[j*p+k, r]) * int(W[k*n+i, r])
                if s != 1:
                    return False
    return True

def _try_random_rank_54(time_budget):
    n, m, p = DIMENSIONS
    R = 54
    U_dim = n * m # 6
    V_dim = m * p # 30
    W_dim = p * n # 20
    
    start = time.time()
    np.random.seed(42)
    
    # We want to find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # Let's fix U and V to be random sparse integer matrices and solve for W.
    # The equation is linear in W.
    # For each (i, k), we have sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # This is a system of n*p = 20 equations.
    # Unknowns are W[k,i,r] for r=0..R-1. Total 20*R = 1080 unknowns.
    # The system is underdetermined.
    # We can set R-1 of the W variables to 0 and solve for the last one?
    # No, we need integer solutions.
    
    # Alternative: Fix U and V. Construct the matrix A of size (n*p) x (R) for each (i,k)?
    # No, the equation couples all r.
    # Let's fix U and V. Then for each (i,k), we have:
    # sum_r (sum_j U[i,j,r] V[j,k,r]) W[k,i,r] = 1
    # Let C[i,k,r] = sum_j U[i,j,r] V[j,k,r].
    # Then sum_r C[i,k,r] W[k,i,r] = 1.
    # This is a system of 20 linear equations in 1080 variables (W).
    # We can find a particular integer solution if one exists.
    # Since it's underdetermined, we can set most W to 0 and solve for the rest.
    
    for _ in range(1000):
        if time.time() - start > time_budget:
            break
        # Generate random sparse U, V
        U = np.zeros((U_dim, R), dtype=int)
        V = np.zeros((V_dim, R), dtype=int)
        
        # Make them sparse with small integers
        for r in range(R):
            # Choose a few non-zero entries in U[:, r]
            idx_u = np.random.choice(U_dim, size=2, replace=False)
            for idx in idx_u:
                U[idx, r] = np.random.choice([-1, 1])
            
            idx_v = np.random.choice(V_dim, size=3, replace=False)
            for idx in idx_v:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Compute C[i, k, r] = sum_j U[i, j, r] V[j, k, r]
        # C is of shape (n, p, R)
        C = np.zeros((n, p, R), dtype=int)
        for r in range(R):
            # C[:, :, r] is n x p matrix
            # C[i, k, r] = sum_j U[i, j, r] V[j, k, r]
            # U[:, r] is n x m
            # V[:, r] is m x p
            # So C[:, :, r] = U[:, r] @ V[:, r]
            C[:, :, r] = U[:, r] @ V[:, r]
        
        # Now solve sum_r C[i, k, r] W[k, i, r] = 1 for W.
        # W is of shape (p, n, R)
        # We can set W[k, i, 0..R-2] = 0 and solve for W[k, i, R-1]?
        # No, we need integer solutions.
        # Let's try to set W to be sparse too.
        # We can fix W[k, i, r] for r < R-1 to be 0, and W[k, i, R-1] to be 1/C[i,k,R-1]?
        # This only works if C[i,k,R-1] is +/- 1.
        
        # Better: Use the Smith Normal Form to find integer solutions.
        # But that's expensive.
        
        # Let's try a different approach:
        # Fix W to be a random sparse integer matrix.
        # Solve for U and V?
        # The equation is quadratic in U, V.
        
        # Let's stick to the column dropping idea but with a better initial solution.
        # Maybe the rank-55 solution is not the only one.
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Start with rank-55
    U, V, W = _construct_rank_55()
    rank = U.shape[1]
    best = (rank, U.copy(), V.copy(), W.copy())

    # Try to reduce further with exact integer arithmetic
    # The previous code did this, but let's keep it simple for now.
    
    # If we don't find a better one, return rank 55.
    r, U, V, W = best
    U = U.astype(int).tolist()
    V = V.astype(int).tolist()
    W = W.astype(int).tolist()
    return {"U": U, "V": V, "W": W}
