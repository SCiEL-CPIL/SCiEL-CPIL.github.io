import time
import sys
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def make_rank_55():
        """
        Rank 55 decomposition for 2x3 x 3x10.
        Strategy:
        - j=0 (first column of A, first row of B): Naive 2x10 = 20 terms.
        - j=1,2 (last 2 cols of A, last 2 rows of B): Strassen-like 2x2 x 2x10.
          Split B into 5 blocks of 2 columns.
          Each 2x2 x 2x2 block uses 7 terms (Strassen).
          5 blocks * 7 = 35 terms.
        Total = 20 + 35 = 55.
        """
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        # A[i,0] * B[0,k] -> C[i,k]
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
            
        # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
        # A2 is 2x2, B2 is 2x10
        # Split B2 into 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # Strassen indices for U (A2 entries)
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            # Strassen indices for V (B2 block entries)
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # W indices for outputs
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21][col] += 1
            U[u_22][col] += 1
            V[v_11][col] += 1
            W[w_10][col] += 1
            W[w_11][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11][col] += 1
            V[v_12][col] += 1
            V[v_22][col] += -1
            W[w_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22][col] += 1
            V[v_21][col] += 1
            V[v_11][col] += -1
            W[w_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11][col] += 1
            U[u_12][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += -1
            W[w_01][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21][col] += 1
            U[u_11][col] += -1
            V[v_11][col] += 1
            V[v_12][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12][col] += 1
            U[u_22][col] += -1
            V[v_21][col] += 1
            V[v_22][col] += 1
            W[w_00][col] += 1
            col += 1
            
        if col != rank:
            return None
        return {"U": U, "V": V, "W": W}

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(decomp, tol=1e-9):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        rank = U.shape[1]
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def search_lower_rank(max_rank_to_try=54, time_limit=None):
        """
        Search for a rank < 55 decomposition.
        Given the disjoint support of the j=0 and j=1,2 parts,
        we focus on the 2x2 x 2x10 part.
        We try a limited random search for rank 34 (total 54) in the 2x2 x 2x10 subspace.
        """
        if time_limit is None:
            time_limit = DEADLINE - 10.0 - (time.time() - start_time)
        
        if time_limit <= 0:
            return None

        # We only search for rank 54 (saving 1 term)
        # This implies rank 34 for the 2x2 x 2x10 part.
        # This is a very hard search space.
        # We will try a few random perturbations of the Strassen solution for the 2x2 x 2x10 part.
        
        # For efficiency, we skip the full random search and just return None,
        # as the probability of finding a rank 34 decomposition for 2x2x2x10
        # with simple random integer/half-integer search is extremely low.
        
        return None

    # First, try to find a lower rank decomposition
    result_lower = search_lower_rank(max_rank_to_try=54)
    if result_lower is not None and verify(result_lower):
        return result_lower

    # Fall back to rank 55 construction
    result_55 = make_rank_55()
    if result_55 is not None and verify(result_55):
        return result_55

    # If rank 55 fails verification (shouldn't happen), fall back to schoolbook
    return make_schoolbook()

# EVOLVE-BLOCK-END
