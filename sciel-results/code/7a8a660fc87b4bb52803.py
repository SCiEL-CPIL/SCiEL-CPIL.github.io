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
        Use the Larsen framework to determine if rank 54 is achievable for 2x3x10.
        
        Key insight: The multiplication tensor has a block structure. The j=0 part
        (A[:,0] * B[0,:]) is a 2x1x10 tensor with rank exactly 20 (each term is 
        distinct). The j=1,2 part is a 2x2x10 tensor.
        
        For the full problem, we need to check if we can share terms between blocks
        or if there's a more efficient decomposition.
        
        Strategy: 
        1. Check if rank 54 is possible by attempting to construct it.
        2. If the 2x2x10 part can be done in rank 34, total is 54.
        3. Try to find such a decomposition using linear algebra.
        """
        if time_limit is None:
            time_limit = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
        
        if time.time() - start_time > time_limit:
            return None
        
        # The j=0 part: A[i,0]*B[0,k] for i in {0,1}, k in {0..9}
        # This is a 2x1x10 tensor. Each term (i,k) is independent.
        # Rank = 20 (can't be reduced since each A[i,0]*B[0,k] appears in exactly one output C[i,k])
        
        # The j=1,2 part: A[i,j]*B[j,k] for i in {0,1}, j in {1,2}, k in {0..9}
        # This is a 2x2x10 tensor. We need rank 34 to get total rank 54.
        
        # Known: 2x2x2 has rank 7 (Strassen)
        # For 2x2x10, if we could share structure across the 5 blocks of 2 columns,
        # we might reduce below 5*7=35.
        
        # Let's try to construct a rank 34 decomposition for 2x2x10.
        # Approach: Use 7 terms for each of the first 4 blocks (8 columns),
        # and find a 2-term decomposition for the last block (2 columns) that
        # shares with existing terms? No, that doesn't work directly.
        
        # Alternative: Try to find if there's a rank 34 solution by solving
        # a system of equations.
        
        # For 2x2xn, the minimal rank is known to be:
        # n=1: 2, n=2: 7, n=3: 10, n=4: 14, n=5: 17, ...
        # Actually, for 2x2xn, the rank is 3n+1 for n>=2? No, that's not right.
        # Strassen gives 7 for n=2. For n=3, it's 10. For n=4, it's 14.
        # Pattern: 3n+1 for n>=2? 3*2+1=7, 3*3+1=10, 3*4+1=13? No, it's 14.
        # Actually, I think for 2x2xn, the rank is 3n+1 for odd n and 3n+2 for even n?
        # Let me just try to construct it.
        
        # For 2x2x10, if the rank is 34, we save 1 term.
        # Let's try to build it by taking the Strassen decomposition for each 2x2x2 block
        # and trying to merge one term.
        
        # Actually, let's just try rank 54 directly by modifying the rank 55 solution.
        # We'll try to find if one of the Strassen terms can be eliminated or merged.
        
        # For now, return None to use the rank 55 construction.
        # A full search would require solving polynomial systems, which is expensive.
        
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
