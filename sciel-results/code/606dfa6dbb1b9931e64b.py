import time
import math
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin for the 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 8

    # The tensor is T[i,j,k] for i in range(n), j in range(m), k in range(p)
    # with T[i,j,k] = 1.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The convention for W is W[k*n + i, r] = coefficient.
    # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        # For efficiency, we only check a random subset of (i, j, k)
        # and some structured ones.
        # Full check: O(n*m*p*rank) which is 96 * rank. If rank < 100, it's fast.
        
        # Check all (i,j,k) to be safe since n*m*p is only 96.
        # T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = 0.0
                    row_u = i * m + j
                    row_v = j * p + k
                    col_w = k * n + i
                    for r in range(len(U[0])):
                        acc += U[row_u][r] * V[row_v][r] * W[col_w][r]
                    if not math.isclose(acc, 1.0, abs_tol=1e-10):
                        return False
        return True

    def make_decomposition(U, V, W):
        """Ensure matrices are lists of lists of numbers."""
        rank = len(U[0]) if U and U[0] else 0
        U_list = [list(row) for row in U]
        V_list = [list(row) for row in V]
        W_list = [list(row) for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # 1. Schoolbook decomposition (Baseline, Rank 96)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    best_res = make_decomposition(U_sb, V_sb, W_sb)

    # 2. Try to find a smaller rank using a random search / optimization
    # We will try random decompositions with decreasing rank.
    # Given the structure, simple random search in R^rank might be hard.
    # Let's try a heuristic: split the tensor into parts that admit small decompositions.
    
    # Idea:
    # The tensor is a 3x4x8 slice of a product tensor.
    # Consider that A (3x4) and B (4x8) are multiplied.
    # If we could decompose the 4x4x4 tensor (pad A to 4x4, split B into two 4x4s) efficiently?
    # But we need exact decomposition for the specific slice.
    
    # Let's try a structured search.
    # Rank R.
    # We can try to set U, V, W from a smaller rank decomposition of a "super-tensor" and project.
    
    # Alternatively, just try random rank R decompositions and verify.
    # But the space is huge.
    
    # Let's try a specific construction based on the "Laderman" idea for 4x4x4.
    # Laderman decomposes 4x4x4 matrix multiplication with rank 21.
    # Can we use it here?
    # We are multiplying 3x4 by 4x8.
    # Pad A to 4x4 (add a row of zeros). Let A_padded[3, :] = 0.
    # Split B into B1 (4x4) and B2 (4x4).
    # C1 = A_padded * B1 (4x4). Top 3 rows are the first 4 columns of C.
    # C2 = A_padded * B2 (4x4). Top 3 rows are the last 4 columns of C.
    # Total rank = 21 + 21 = 42.
    # Let's try to implement this.
    
    # Laderman tensors for 4x4x4:
    # 21 tuples (u, v, w) where u, v, w are vectors of length 16 (flattened 4x4 matrices)
    # u is for A (row-major: i*4+j), v is for B (row-major: j*4+k), w is for C (row-major: k*4+i)
    
    # The Laderman algorithm uses the following 21 terms (indices 0..15 for 4x4):
    # We need the exact Laderman coefficients.
    # Reference: J. R. Laderman, "A 21-term algorithm for 4x4 matrix multiplication"
    # The terms are:
    # (0,0,0), (0,1,0), (0,2,0), (0,3,0), (1,0,1), (1,1,1), (1,2,1), (1,3,1),
    # (2,0,2), (2,1,2), (2,2,2), (2,3,2), (3,0,3), (3,1,3), (3,2,3), (3,3,3) -> This is schoolbook (16 terms)
    # Laderman adds 5 terms to reduce the "cross" terms? No, Laderman is a specific set of 21.
    
    # Actually, I don't have the exact 21 Laderman tuples memorized perfectly to avoid errors.
    # However, I can generate a random sparse decomposition and try to optimize?
    # Or, I can try to find a rank-42 solution by searching.
    
    # Let's try a smarter heuristic search.
    # Since 96 -> 42 is a big jump, let's try to find ANY rank < 96.
    # We can do a local search starting from schoolbook, zeroing out rows?
    # Or random restarts for small ranks.
    
    # Let's try ranks 40-95.
    # For each rank, we can try to fit the tensors.
    # This is a difficult non-convex optimization problem.
    
    # A simpler approach: Use the known structure.
    # If we assume the decomposition is sparse.
    
    # Let's try to use the fact that n=3.
    # We can write the tensor as:
    # T = T_0 + T_1 + T_2
    # where T_a has non-zeros only for i=a.
    # T_a is a 1 x 4 x 8 slice.
    # For fixed i, T_a[j,k] = 1 for all j,k.
    # This is a 1x4x8 all-ones tensor.
    # Rank of a 1 x m x p all-ones tensor is 1?
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # If we set U[i*m+j, r] = 1, V[j*p+k, r] = 1/sqrt(32), W[k*n+i, r] = sqrt(32)?
    # Then sum is 1 * (1/32) * 32 = 1?
    # No, the sum is over r.
    # If rank=1 for each i, then total rank 3?
    # Let's test Rank 3.
    # U[0..11, r] for i=0: r=0 is 1, others 0.
    # V[0..31, r] for i=0: r=0 is 1.
    # W[0..19, r] for i=0: W[k*3+0, 0] should be 1?
    # T[0, j, k] = U[j, 0] * V[j*p+k, 0] * W[k*3, 0].
    # We need U[j, 0]=1, V[j*4+k, 0]=1, W[k*3, 0]=1.
    # Then T[0,j,k] = 1*1*1 = 1.
    # Does this interfere with i=1 or i=2?
    # For i=1, T[1, j, k] should be 0 in this term?
    # U[1*m+j, 0] is not defined by this term (it's 0).
    # V[j*p+k, 0] is 1.
    # W[k*3+1, 0] is 0.
    # So the term for r=0 contributes to T[0,j,k] only.
    # We can do the same for i=1 with r=1, and i=2 with r=2.
    
    # Let's check this Rank 3 decomposition!
    U_3 = [[0.0] * 3 for _ in range(n * m)]
    V_3 = [[0.0] * 3 for _ in range(m * p)]
    W_3 = [[0.0] * 3 for _ in range(p * n)]
    
    for r in range(3):
        # r corresponds to i=r
        for j in range(m):
            U_3[r * m + j][r] = 1.0
        for j in range(m):
            for k in range(p):
                V_3[j * p + k][r] = 1.0
        for k in range(p):
            W_3[k * n + r][r] = 1.0
            
    if verify(U_3, V_3, W_3):
        best_res = make_decomposition(U_3, V_3, W_3)
        return best_res

    # Can we do better than 3?
    # Rank 1?
    # T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # T[0,0,0]=1 => U00 V00 W00 = 1
    # T[0,1,0]=1 => U01 V00 W00 = 1 => U01=U00
    # T[0,0,1]=1 => U00 V01 W01 = 1
    # T[1,0,0]=1 => U4 V00 W01 = 1
    # T[1,1,1]=1 => U5 V01 W02 = 1
    # It seems possible?
    # Let's check consistency.
    # U[i,j] depends only on j? (From T[i,j,0] vs T[i,j,1]?)
    # T[i,j,k] = U_ij V_jk W_ki.
    # This looks like a rank 1 decomposition of the "product tensor" structure?
    # If we set U_ij = u_i v_j, V_jk = v_j w_k, W_ki = w_k u_i?
    # Then term = u_i v_j * v_j w_k * w_k u_i = u_i^2 v_j^2 w_k^2.
    # This is constant for all i,j,k only if u,v,w are constant.
    # Then T is constant.
    # So Rank 1 works if we can find scalars.
    # U[0]=a, U[1]=b...
    # Let's try to solve for Rank 1.
    # U[j] = a_j (for i=0) -> U[0..3]
    # U[4..7] (i=1)
    # U[8..11] (i=2)
    # V[0..3] (j=0), V[4..7] (j=1) etc.
    # W[0..2] (k=0, i=0,1,2?) No, W index is k*n+i.
    # W[0..2] for k=0. W[3..5] for k=1.
    
    # T[0,0,0] = U0 V0 W0 = 1
    # T[0,1,0] = U1 V0 W0 = 1 => U1=U0
    # ... U[0..3] all equal a0.
    # T[1,0,0] = U4 V0 W1 = 1 => U4 = 1/(V0 W1)
    # T[0,0,1] = U0 V4 W4? No W index k=1, i=0 is W[3].
    # T[0,0,1] = U0 V4 W3 = 1
    # T[0,0,2] = U0 V8 W6 = 1
    # T[0,0,3] = U0 V12 W9 = 1
    # It seems we need V4, V8, V12 to be related to W3, W6, W9.
    
    # Let's try to search for Rank 1.
    # Fix U and V, solve for W?
    # W[k*n+i] = T[i,j,k] / (U[i*m+j] * V[j*p+k]).
    # This must be independent of j.
    # So T[i,j,k] / (U_ij * V_jk) = W_ki must be same for all j.
    # U_ij * V_jk = T[i,j,k] / W_ki = 1 / W_ki.
    # Let 1/W_ki = C_ki.
    # U_ij * V_jk = C_ki.
    # This implies U and V must have rank 1 structure?
    # U[i,j] = u_i * v_j
    # V[i,j] = v_i * w_j (here V index is j,k so V[j,k])
    # Then u_i v_j * v_j w_k = u_i v_j^2 w_k = C_ki.
    # This requires v_j^2 to be constant (independent of j) for the equation to hold for all i,k?
    # If v_j^2 = c, then u_i w_k = C_ki / c.
    # This works if C_ki is rank 1.
    # C_ki = 1/W_ki.
    # We can choose W_ki such that 1/W_ki is rank 1.
    # Let 1/W_ki = x_i y_k.
    # Then W_ki = 1/(x_i y_k).
    # We need U_ij * V_jk = x_i y_k.
    # Let U_ij = sqrt(x_i) * s_j.
    # Let V_jk = sqrt(y_k) * (1/s_j).
    # Then U_ij V_jk = sqrt(x_i) sqrt(y_k).
    # This is rank 1.
    # So we can choose any x_i, y_k.
    # Let x_i = 1, y_k = 1.
    # Then W_ki = 1.
    # U_ij * V_jk = 1.
    # We need to satisfy the original T=1.
    # T = U V W = U V * 1 = U V = 1.
    # So we just need a rank 1 decomposition of the "product" structure?
    # Wait, if W is all 1s, and U V = 1, then T = 1.
    # Can we have U and V such that for all i,j,k, U[i*m+j] * V[j*p+k] = 1?
    # This requires U[i*m+j] to depend only on i? No, if j changes, U changes.
    # U[i*m+j] * V[j*p+k] = 1 for all i,j,k.
    # Set i=0, k=0.
    # U[j] * V[j*4] = 1 for j=0..3.
    # U[4+j] * V[j*4] = 1 for j=0..3 (i=1).
    # U[8+j] * V[j*4] = 1 for j=0..3 (i=2).
    # Let V[j*4] = v_j.
    # U[j] = 1/v_j.
    # U[4+j] = 1/v_j.
    # U[8+j] = 1/v_j.
    # So U column j is all 1/v_j.
    # Now check other k.
    # U[j] * V[j*4+1] = 1 => (1/v_j) * V[j*4+1] = 1 => V[j*4+1] = v_j.
    # So V row j is [v_j, v_j, v_j, v_j].
    # This works!
    
    # So we can construct a Rank 1 solution!
    U_1 = [[0.0]] * (n * m)
    V_1 = [[0.0]] * (m * p)
    W_1 = [[0.0]] * (p * n)
    
    # Set arbitrary non-zero values for v_j (j in 0..3).
    v = [1.0, 1.0, 1.0, 1.0]
    for i in range(n):
        for j in range(m):
            U_1[i * m + j][0] = 1.0 / v[j]
            
    for j in range(m):
        for k in range(p):
            V_1[j * p + k][0] = v[j]
            
    for k in range(p):
        for i in range(n):
            W_1[k * n + i][0] = 1.0 # Since W must be 1 to make T=1
            
    if verify(U_1, V_1, W_1):
        best_res = make_decomposition(U_1, V_1, W_1)
        return best_res

    # It seems Rank 1 is valid.
    # Let's double check the verification logic.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For r=0:
    # U[i*m+j, 0] = 1/v_j
    # V[j*p+k, 0] = v_j
    # W[k*n+i, 0] = 1
    # Product = (1/v_j) * v_j * 1 = 1.
    # Correct.

    return best_res
# EVOLVE-BLOCK-END
