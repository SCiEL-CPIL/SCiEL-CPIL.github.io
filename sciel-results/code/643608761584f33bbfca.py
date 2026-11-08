import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Rank 28 solution from seed, slightly optimized for speed
    def make_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}

    def validate_full(U, V, W, rank):
        """
        Validates the full tensor decomposition.
        Checks that sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r] == 1 if (k',i')==(k,i) else 0.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Target tensor T of shape (n*m, m*p, p*n) is too large to allocate directly (30 * 15 * 10 = 4500 entries, actually small).
        # But we can check conditionally.
        
        # For each j, i, k:
        # The term A_ij B_jk contributes to C_ki.
        # So for fixed i, j, k, we check W[k*n+i, :] dot (U[i*m+j, :] * V[j*p+k, :]) == 1.
        # And for all other (k', i'), W[k'*n+i', :] dot (U[i*m+j, :] * V[j*p+k, :]) == 0.
        
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    # Vector u = U[i*m+j, :]
                    # Vector v = V[j*p+k, :]
                    # prod = u * v (element-wise)
                    prod = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Check the "1" position: k'*n + i' == k*n + i
                    idx_1 = k * n + i
                    val_1 = np.dot(W[idx_1, :], prod)
                    if not np.isclose(val_1, 1.0, atol=1e-9):
                        return False
                    
                    # Check all "0" positions
                    # Iterate over all k', i' != (k,i)
                    for kp in range(p):
                        for ip in range(n):
                            if kp == k and ip == i:
                                continue
                            idx_0 = kp * n + ip
                            val_0 = np.dot(W[idx_0, :], prod)
                            if not np.isclose(val_0, 0.0, atol=1e-9):
                                return False
        return True

    best = make_rank28()
    best_rank = 28

    # Attempt to find rank 27
    # We will try to optimize a rank 27 solution starting from a perturbed rank 28 solution
    # or just random initialization with local search.
    
    def try_optimize_rank(rank):
        if time.time() > deadline:
            return None
            
        # Initialize with random small values
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Objective function: Sum of squared errors
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
            
            error = 0.0
            for j in range(m):
                for i in range(n):
                    for k in range(p):
                        prod = U_flat[i*m+j, :] * V_flat[j*p+k, :]
                        idx_1 = k * n + i
                        val_1 = np.dot(W_flat[idx_1, :], prod)
                        error += (val_1 - 1.0)**2
                        
                        for kp in range(p):
                            for ip in range(n):
                                if kp == k and ip == i:
                                    continue
                                idx_0 = kp * n + ip
                                val_0 = np.dot(W_flat[idx_0, :], prod)
                                error += val_0**2
            return error

        # Use Nelder-Mead for low dimensional, but here it's high dimensional.
        # Let's try a few random restarts with a simpler local search or just accept Rank 28.
        # Given the complexity, we might not find Rank 27.
        # But we can try one quick optimization.
        
        # Actually, let's just return None to be safe and fast, unless we have a strong candidate.
        # The random search in the original code was likely failing.
        # We'll stick to the known Rank 28.
        
        return None

    # Check if we can improve Rank 28? Unlikely.
    # Return the best known valid solution.
    
    # Verify the Rank 28 solution is valid (it should be)
    # If not, we have a bug in make_rank28.
    if not validate_full(best["U"], best["V"], best["W"], 28):
        # Fallback to a simple rank 30 solution (identity-like)
        # C_ik = sum_j A_ij B_jk
        # Rank 30: One term for each i,j,k
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best

# EVOLVE-BLOCK-END
