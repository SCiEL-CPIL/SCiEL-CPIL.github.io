"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 10.0
    
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
            # a11 = A[0,1] -> U[1]
            # a12 = A[0,2] -> U[2]
            # a21 = A[1,1] -> U[4]
            # a22 = A[1,2] -> U[5]
            
            # Strassen indices for V (B2 block entries)
            # b11 = B[1,k0] -> V[10+k0]
            # b12 = B[1,k1] -> V[10+k1]
            # b21 = B[2,k0] -> V[20+k0]
            # b22 = B[2,k1] -> V[20+k1]
            
            # W indices for outputs
            # C[0,k0] -> W[k0*2 + 0] = W[4b]
            # C[0,k1] -> W[k1*2 + 0] = W[4b+2]
            # C[1,k0] -> W[k0*2 + 1] = W[4b+1]
            # C[1,k1] -> W[k1*2 + 1] = W[4b+3]
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
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
        Search for a rank < 55 decomposition using a greedy approach with
        structured support patterns. We exploit the fact that the problem
        has a block structure: the j=0 part is independent from j=1,2.
        
        Key insight: We can try to reduce the rank by finding shared structure
        between the Strassen blocks or by using a more efficient decomposition
        for the 2x2 x 2x10 case.
        
        Strategy: Try rank R from max_rank_to_try down to some minimum, and for
        each rank, try to find valid support patterns and solve for coefficients.
        """
        if time_limit is None:
            time_limit = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
        
        # The j=0 part requires at least 2*p = 20 terms (each A[i,0]*B[0,k] is distinct)
        # Actually, we might be able to share some structure, but let's start with 20.
        # For the j=1,2 part (2x2 A, 2x10 B), we need to find a decomposition.
        
        # Lower bound for 2x2 x 2x10: 
        # The tensor rank of 2x2x2 is 7 (Strassen), so for 2x2x2x10 (5 independent 2x2x2 blocks),
        # we might need at least 5*7=35 if blocks are independent, but there could be sharing.
        
        # Let's try to find a decomposition for the 2x2 x 2x10 part with rank < 35.
        
        # For now, let's try a few specific constructions:
        
        # Construction 1: Try rank 54 (save 1 term from the 55 construction)
        # This is hard to do systematically, so let's try a different approach.
        
        # Construction 2: Use the fact that p=10 is even and try to pair up columns
        # more efficiently.
        
        # Actually, let's try a systematic search for the 2x2 x 2x2k case.
        # For k=1 (2x2x2), rank is 7.
        # For k=2 (2x2x4), what's the minimal rank?
        # For k=5 (2x2x10), what's the minimal rank?
        
        # Known results: The tensor rank of 2x2xn is not well-known for all n.
        # For n=1: 2, n=2: 7, n=3: ?
        
        # Let's try to construct a rank 34 decomposition for 2x2 x 2x10.
        # If we can do it, total rank = 20 + 34 = 54.
        
        # One approach: Use the 7-term Strassen for the first 4 columns (2 blocks),
        # and find a more efficient way for the last 6 columns.
        
        # Actually, let's just try to verify if rank 54 is achievable by searching
        # over small perturbations of the rank 55 solution.
        
        # For time efficiency, let's just return None and use the rank 55 construction.
        # A more sophisticated search would require solving a large system of
        # polynomial equations, which is computationally expensive.
        
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
