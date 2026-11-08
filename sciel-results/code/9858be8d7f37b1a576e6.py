import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    def make_rank28():
        rank = 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need T_{i,j,k} = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U_arr[i*m+j, rr] * V_arr[j*p+k, rr] * W_arr[k*n+i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_reduce_rank(current_U, current_V, current_W, current_rank, target_rank, time_limit):
        """
        Attempt to reduce rank by removing columns that are linear combinations
        of others or by finding a lower rank representation.
        This is a heuristic search.
        """
        start = time.time()
        # We will try to find a rank-target decomposition.
        # Strategy: Random restarts with local search? Or algebraic elimination?
        # Given the small size, we can try to set some columns to zero and see if we can solve for the rest.
        # However, the space is discrete (integers/half-integers) in the ideal case, but the problem allows reals?
        # "integer/half-integer entries" is specified in the task description.
        # So we are restricted to Z + 0.5Z.
        
        # Let's try a simple approach: 
        # 1. Take the current rank-R solution.
        # 2. Try to merge two terms or eliminate one term.
        # This is hard to do generically.
        
        # Alternative: Random search for rank < 28.
        # Since 28 is close to 30 (naive), maybe 27 or 26 is possible?
        # The Strassen algorithm for 2x2x2 has rank 7.
        # Naive is 8.
        # Here we have 2x3x5.
        # Rank lower bound?
        
        # Let's do a limited random search for rank 27, 26, etc.
        # Values: -1, -0.5, 0, 0.5, 1
        vals = [-1, -0.5, 0, 0.5, 1]
        
        for r in range(target_rank, current_rank):
            if time.time() - start > time_limit:
                break
            
            # Try a few random instances
            for _ in range(50):
                if time.time() - start > time_limit:
                    break
                U_new = np.random.choice(vals, size=(nm, r))
                V_new = np.random.choice(vals, size=(mp, r))
                W_new = np.random.choice(vals, size=(pn, r))
                
                if validate(U_new.tolist(), V_new.tolist(), W_new.tolist(), r):
                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        
        return None

    best = make_rank28()
    best_rank = 28
    
    # Attempt to reduce rank
    # Use 10 seconds for search
    candidate = try_reduce_rank(best["U"], best["V"], best["W"], best_rank, 20, 10.0)
    if candidate is not None:
        # Verify candidate
        if validate(candidate["U"], candidate["V"], candidate["W"], len(candidate["U"][0])):
            best = candidate
            best_rank = len(candidate["U"][0])
    
    return best
# EVOLVE-BLOCK-END
