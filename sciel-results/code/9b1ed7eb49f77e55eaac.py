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
        Attempt to find a rank-54 decomposition for 2x3x10.
        
        The j=0 part (2x1x10) has rank exactly 20.
        The j=1,2 part (2x2x10) needs rank 34 for total rank 54.
        
        We try to construct a rank-34 decomposition for 2x2x10 by:
        1. Using Strassen (7 terms) for 4 blocks = 28 terms
        2. Finding a 6-term decomposition for the last 2x2x2 block that shares
           structure with existing terms, OR
        3. Using a more efficient global decomposition.
        
        Known results: 2x2xn matrix multiplication has rank 3n+1 for n>=2.
        For n=10: rank = 31. So 2x2x10 can be done in rank 31!
        Total would be 20 + 31 = 51.
        
        Let's try to construct a rank-31 decomposition for 2x2x10.
        """
        if time_limit is None:
            time_limit = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
        
        if time.time() - start_time > time_limit:
            return None
        
        # For 2x2xn, the minimal rank is 3n+1 for n >= 2.
        # Reference: Landsberg, "Tensor geometry" and subsequent work.
        # For n=10: rank = 31.
        
        # Construct a rank-31 decomposition for 2x2x10.
        # The decomposition uses the fact that 2x2xn can be decomposed using
        # 3n+1 terms via a recursive/block construction.
        
        # One known construction:
        # For 2x2xn, use the following pattern with n+1 "diagonal" terms and
        # 2n "off-diagonal" terms, total 3n+1.
        
        # Actually, let me use a different approach: build it block by block
        # using Strassen for the first 4 blocks (28 terms) and then try to
        # handle the last block more efficiently by sharing with existing terms.
        
        # Alternatively, use the known optimal construction for 2x2xn.
        
        # Let me try a direct construction for 2x2x10 with rank 31.
        
        def make_2x2x10_rank31():
            """
            Construct a rank-31 decomposition for 2x2x10 matrix multiplication.
            A is 2x2, B is 2x10, C is 2x10.
            C[i,k] = sum_j A[i,j] * B[j,k]
            
            We need 31 terms such that:
            sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,j,k}
            
            where U is 4x31, V is 20x31, W is 20x31.
            """
            rank = 31
            U = [[0] * rank for _ in range(4)]  # 2x2 = 4 entries
            V = [[0] * rank for _ in range(20)]  # 2x10 = 20 entries
            W = [[0] * rank for _ in range(20)]  # 10x2 = 20 entries
            
            col = 0
            
            # Use a construction based on the fact that 2x2xn has rank 3n+1.
            # One approach: use n "main" terms and 2n "cross" terms.
            
            # Let me try a different construction:
            # For each k from 0 to 9, we have B[0,k] and B[1,k].
            # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
            # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
            
            # This is like 10 independent 2x1x1 problems, but we can share.
            
            # Known construction for 2x2xn with rank 3n+1:
            # Terms 0 to n-1: "diagonal" terms
            # Terms n to 3n: "off-diagonal" terms
            # Term 3n: "correction" term
            
            # Let me use a specific construction.
            # For 2x2xn, one construction uses:
            # For k=0,...,n-1:
            #   Term k: U[0,0]=1, V[0,k]=1, W[k,0]=1  (A[0,0]*B[0,k]->C[0,k])
            #   Term k+n: U[0,1]=1, V[1,k]=1, W[k,0]=1  (A[0,1]*B[1,k]->C[0,k])
            #   Term k+2n: U[1,0]=1, V[0,k]=1, W[k,1]=1  (A[1,0]*B[0,k]->C[1,k])
            #   Term k+3n-1: ... this doesn't work directly.
            
            # Let me try a different approach: use the fact that we can group
            # columns and use Strassen-like decompositions.
            
            # For 2x2x10, split into 5 blocks of 2 columns each.
            # Each 2x2x2 block uses 7 terms (Strassen).
            # Total: 5*7 = 35 terms.
            
            # To get rank 31, we need to save 4 terms.
            # This is non-trivial.
            
            # Let me just return None for now and use the rank-55 construction.
            # A proper rank-31 construction requires more sophisticated algebra.
            
            return None
        
        # Try to construct rank-31 for 2x2x10
        result_2x2 = make_2x2x10_rank31()
        if result_2x2 is not None:
            # Combine with the j=0 part (20 terms) to get rank 51
            rank_total = 20 + 31
            U_full = [[0] * rank_total for _ in range(n * m)]
            V_full = [[0] * rank_total for _ in range(m * p)]
            W_full = [[0] * rank_total for _ in range(p * n)]
            
            # Add j=0 part (20 terms)
            col = 0
            for i in range(n):
                for k in range(p):
                    u_idx = i * m + 0
                    v_idx = 0 * p + k
                    w_idx = k * n + i
                    U_full[u_idx][col] = 1
                    V_full[v_idx][col] = 1
                    W_full[w_idx][col] = 1
                    col += 1
            
            # Add j=1,2 part (31 terms)
            U_2x2 = result_2x2["U"]
            V_2x2 = result_2x2["V"]
            W_2x2 = result_2x2["W"]
            for c in range(31):
                for i in range(2):
                    for j in range(2):
                        U_full[(i) * m + (j + 1)][col + c] = U_2x2[i * 2 + j][c]
                for j in range(2):
                    for k in range(10):
                        V_full[(j + 1) * p + k][col + c] = V_2x2[j * 10 + k][c]
                for k in range(10):
                    for i in range(2):
                        W_full[k * n + i][col + c] = W_2x2[k * 2 + i][c]
            
            decomp = {"U": U_full, "V": V_full, "W": W_full}
            if verify(decomp):
                return decomp
        
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
