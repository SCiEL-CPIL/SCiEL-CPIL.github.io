"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _max_error(U, V, W, n, m, p):
    """Compute the maximum absolute error over all (i,j,k)."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _try_snap_and_check(U, V, W, n, m, p):
    """Try to snap to half-integers and check exactness."""
    for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
        U_snapped, u_ok = _snap_to_half_integer(U, tol=tol)
        V_snapped, v_ok = _snap_to_half_integer(V, tol=tol)
        W_snapped, w_ok = _snap_to_half_integer(W, tol=tol)
        if u_ok and v_ok and w_ok:
            if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped
    return None

def _als_iterate_fast(U, V, W, n, m, p, max_iter=500, tol=1e-14):
    """Faster ALS with vectorized updates."""
    rank = U.shape[1]
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U: for each (i,j), solve least squares
        for idx in range(n * m):
            i, j = divmod(idx, m)
            A = V[j*p:(j+1)*p, :] * W[(0*n+i):(p*n+i):n, :]
            b = np.ones(p)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        
        # Update V
        for idx in range(m * p):
            j, k = divmod(idx, p)
            A = U[j::m, :] * W[k*n:(k+1)*n, :]
            b = np.ones(n)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V[idx, :] = sol
            except:
                pass
        
        # Update W
        for idx in range(p * n):
            k, i = divmod(idx, n)
            A = U[i*m:(i+1)*m, :] * V[k::p, :]
            b = np.ones(m)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W[idx, :] = sol
            except:
                pass
        
        # Check convergence
        diff = (np.max(np.abs(U - U_old)) + 
                np.max(np.abs(V - V_old)) + 
                np.max(np.abs(W - W_old)))
        if diff < tol:
            break
    
    return U, V, W

def _strassen_2x2_init(n, m, p, rank, seed=0):
    """
    Strassen-inspired initialization for 3x4x4.
    
    Split the 4x4 matrix into 2x2 blocks.
    A is 3x4: split into A_top (3x2) and A_bot (3x2)
    B is 4x4: split into B_tl (2x2), B_tr (2x2), B_bl (2x2), B_br (2x2)
    
    C = A * B = [A_top * B_tl + A_bot * B_bl, A_top * B_tr + A_bot * B_br]
    
    For each of the two 2x2 block multiplications, use Strassen (7 mults).
    For the remaining 3x2 * 2x2 part, use schoolbook (6 mults per 2x2 block).
    
    Total: 2*7 (Strassen for two 2x2 blocks) + 3*2 (schoolbook for 3x2*2x2) 
           = 14 + 6 = 20... but we need to be more careful.
    
    Actually let me think differently. The 3x4 * 4x4 can be viewed as:
    - 3 rows of A, each row is 1x4
    - 4 columns of B, each column is 4x1
    
    Alternative: split 4 into 2+2.
    A = [A1 | A2] where A1, A2 are 3x2
    B = [[B11, B12], [B21, B22]] where each is 2x2
    
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each term A1*B11 is 3x2 * 2x2 = 3x2, needing 6 scalar mults schoolbook.
    So total schoolbook: 4 * 6 = 24.
    
    With Strassen on the 2x2 blocks:
    A1*B11: A1 is 3x2, B11 is 2x2. We can't directly apply Strassen since A1 isn't square.
    
    Let me try a different decomposition. Think of it as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1} and {2,3}.
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    For the first sum: A[i,0]*B[0,k] + A[i,1]*B[1,k]
    For the second sum: A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    This is just 4 scalar mults per (i,k) pair, so 3*4*4 = 48 total. Same as schoolbook.
    
    Let me try the Strassen approach more carefully.
    View the problem as multiplying a 3x4 by 4x4.
    
    Pad A to 4x4 (add a zero row). Then it's 4x4 * 4x4.
    Strassen gives rank 7 for 4x4*4x4.
    But we need 3x4 * 4x4, which is a sub-problem.
    
    Actually, let me try a block approach:
    Split the 4x4 B into four 2x2 blocks.
    A is 3x4, split into A_left (3x2) and A_right (3x2).
    
    C_left = A_left * B_tl + A_right * B_bl  (3x2 result)
    C_right = A_left * B_tr + A_right * B_br  (3x2 result)
    
    For A_left * B_tl: this is 3x2 * 2x2.
    We can write this as: for each row i of A_left (which is 1x2), multiply by B_tl (2x2).
    Row i * B_tl = [A[i,0]*B[0,0] + A[i,1]*B[1,0], A[i,0]*B[0,1] + A[i,1]*B[1,1]]
    
    That's 4 mults per row, 3 rows = 12 mults for A_left * B_tl.
    Similarly A_right * B_bl = 12 mults.
    A_left * B_tr = 12 mults.
    A_right * B_br = 12 mults.
    Total: 48. No savings.
    
    Let me try Strassen on 2x2 blocks properly.
    If I had 2x2 * 2x2, Strassen uses 7 mults instead of 8.
    
    For 3x4 * 4x4, let me think of it as three 1x4 * 4x4 products.
    Each 1x4 * 4x4 = 1x4, needing 16 mults schoolbook.
    
    Or: split into 1x2 * 2x4 blocks.
    1x2 * 2x4: can I use Strassen-like tricks?
    
    Let me try a different approach: use the known result that 3x4x4 has rank at most 44.
    The cell finding says "Using Strassen's algorithm on 2x2 blocks for the 2x4 part and naive for 1x4 part reduces rank from 48 to 44."
    
    So: split the 3 rows into 2+1.
    For the 2x4 * 4x4 part: split into 2x2 * 2x4 blocks.
    2x2 * 2x4: use Strassen on the 2x2 part? 
    Actually, 2x2 * 2x4 can be seen as two 2x2 * 2x2 multiplications.
    Each 2x2 * 2x2 uses Strassen rank 7.
    So 2x2 * 2x4 uses 14 mults.
    Two such blocks (for columns 0-1 and 2-3): 28 mults.
    
    For the 1x4 * 4x4 part: schoolbook = 16 mults.
    
    Total: 28 + 16 = 44. That matches!
    
    So the decomposition is:
    - Rows 0,1 of A (2x4) times B (4x4):
      - Split B columns into [0,1] and [2,3]
      - A[0:2, 0:4] * B[0:4, 0:2]: This is 2x4 * 4x2
        Split A[0:2, 0:4] into A[0:2,0:2] and A[0:2,2:4] (each 2x2)
        Split B[0:4, 0:2] into B[0:2,0:2] and B[2:4,0:2] (each 2x2)
        Result = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        Each 2x2*2x2 uses Strassen (7 mults), so 14 mults.
      - Similarly A[0:2, 0:4] * B[0:4, 2:4]: 14 mults.
      Total for 2x4 * 4x4: 28 mults.
    
    - Row 2 of A (1x4) times B (4x4): schoolbook = 16 mults.
    
    Total: 28 + 16 = 44.
    """
    rng = np.random.RandomState(seed)
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    
    # Strassen 2x2 * 2x2: 7 products
    # Given matrices A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    def strassen_2x2(U, V, W, col, a_rows, b_cols, c_rows, c_cols, n, m, p):
        """
        Apply Strassen to multiply 2x2 block.
        a_rows: indices of the 2 rows in U (i values for A)
        b_cols: indices of the 2 cols in V (k values for B output)
        c_rows: indices of the 2 rows in W (i values for C)  
        c_cols: indices of the 2 cols in W (k values for C)
        
        U index: i*m + j where i is row of A, j is col of A (0..m-1)
        V index: j*p + k where j is row of B, k is col of B
        W index: k*n + i where k is col of C, i is row of C
        
        For 2x2 * 2x2:
        A block: rows a_rows[0], a_rows[1], cols j1, j2 (in A's column space)
        B block: rows j1, j2 (in B's row space), cols b_cols[0], b_cols[1]
        C block: rows c_rows[0], c_rows[1], cols c_cols[0], c_cols[1]
        
        Actually this is getting complex. Let me just directly construct the rank-7 Strassen
        decomposition for each 2x2*2x2 sub-problem.
        """
        pass
    
    # Let me construct it more directly.
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the matmul tensor).
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # For Strassen on a 2x2*2x2 sub-problem:
    # Consider A[i,j] for i in {i1,i2}, j in {j1,j2}
    # B[j,k] for j in {j1,j2}, k in {k1,k2}
    # C[i,k] for i in {i1,i2}, k in {k1,k2}
    #
    # The contribution to C[i,k] from this sub-problem is:
    # sum_{j in {j1,j2}} A[i,j] * B[j,k]
    #
    # Strassen decomposes this 2x2*2x2 into 7 rank-1 terms.
    # Let me define the 7 products:
    # p1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # p2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
    # p3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
    # p4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
    # p5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
    # p6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
    # p7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
    #
    # C[i1,k1] = p1 + p4 - p5 + p7
    # C[i1,k2] = p3 + p5
    # C[i2,k1] = p2 + p4
    # C[i2,k2] = p1 - p2 + p3 + p6
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2, U, V, W, col):
        """Add 7 Strassen terms for 2x2*2x2 sub-problem."""
        # p1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] += 1; V[j1*p+k1, col] += 1; W[k1*n+i1, col] += 1
        U[i1*m+j1, col] += 0; V[j1*p+k2, col] += 1; W[k2*n+i1, col] += 1
        U[i2*m+j2, col] += 1; V[j1*p+k1, col] += 1; W[k1*n+i2, col] += 1
        U[i2*m+j2, col] += 0; V[j2*p+k2, col] += 1; W[k2*n+i2, col] += 1
        # Wait, this isn't right. Each rank-1 term has specific U,V,W vectors.
        # Let me redo this properly.
        # 
        # p1: U has 1 at (i1,j1) and (i2,j2); V has 1 at (j1,k1) and (j2,k2); W has 1 at (k1,i1) and (k2,i2)
        # No wait, that's not a rank-1 term. A rank-1 term is u * v * w where u, v, w are vectors.
        # 
        # For p1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2]):
        # This is NOT a single rank-1 term in the tensor decomposition sense.
        # 
        # I'm confusing Strassen's algorithm with tensor rank decomposition.
        # In Strassen, we compute intermediate products, but each "product" like
        # (a+d)(e+h) is itself a product of two linear combinations, which in the
        # tensor decomposition corresponds to ONE rank-1 term where:
        # U vector has entries for the linear combination on A
        # V vector has entries for the linear combination on B  
        # W vector distributes to C
        #
        # So p1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # U: 1 at (i1,j1), 1 at (i2,j2), 0 elsewhere
        # V: 1 at (j1,k1), 1 at (j2,k2), 0 elsewhere
        # W: contributes to C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
        #    C[i1,k1] += 1, C[i1,k2] += 1, C[i2,k1] += 1, C[i2,k2] += 1
        #    So W: 1 at (k1,i1), 1 at (k2,i1), 1 at (k1,i2), 1 at (k2,i2)
        
        # Actually no. Let me think about this more carefully.
        #
        # The matmul tensor T has T[a,b,c] = 1 where a=(i,j), b=(j,k), c=(k,i).
        # A rank-1 decomposition term r contributes:
        #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        #
        # For the sub-problem where i in {i1,i2}, j in {j1,j2}, k in {k1,k2}:
        # We need: for each (i,j,k) in this sub-cube, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        #
        # Strassen says we can do this with 7 terms instead of 8.
        #
        # Term 1: U has 1 at (i1,j1) and (i2,j2)
        #          V has 1 at (j1,k1) and (j2,k2)
        #          W has 1 at (k1,i1), (k2,i1), (k1,i2), (k2,i2)
        #          Contribution to T[i1,j1,k1]: U=1,V=1,W=1 -> 1 ✓
        #          Contribution to T[i1,j1,k2]: U=1,V=0,W=1 -> 0 ✗ (should be 0, OK since j1≠j2 for this k)
        #          Wait, T[i1,j1,k2] should be 1 (it's a valid matmul entry).
        #          
        # I think I'm overcomplicating this. Let me just directly build the rank-44 decomposition.
        
        return col
    
    # Let me just build it step by step with explicit rank-1 terms.
    # 
    # Part 1: 2x4 * 4x4 using Strassen on 2x2 blocks (28 terms)
    # Rows i=0,1 of A; all columns of B
    #
    # Split into 4 sub-problems:
    # S1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2*2x2, Strassen rank 7
    # S2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2*2x2, Strassen rank 7
    # S3: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2*2x2, Strassen rank 7
    # S4: i in {0,1}, j in {2,3}, k in {2,3} -> 2x2*2x2, Strassen rank 7
    #
    # Part 2: 1x4 * 4x4 schoolbook (16 terms)
    # i=2, all j, all k
    
    def strassen_7_terms(U, V, W, col, i_set, j_set, k_set):
        """
        Add 7 Strassen terms for a 2x2*2x2 sub-problem.
        i_set: (i1, i2), j_set: (j1, j2), k_set: (k1, k2)
        """
        i1, i2 = i_set
        j1, j2 = j_set
        k1, k2 = k_set
        
        # p1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # U: 1 at (i1,j1), 1 at (i2,j2)
        # V: 1 at (j1,k1), 1 at (j2,k2)
        # W: 1 at (k1,i1), 1 at (k2,i1), 1 at (k1,i2), 1 at (k2,i2)
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        W[k2*n+i1, col] = 1
        W[k1*n+i2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # p2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # U: 1 at (i2,j1), 1 at (i2,j2)
        # V: 1 at (j1,k1)
        # W: 1 at (k1,i2)
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1
        col += 1
        
        # p3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        # U: 1 at (i1,j1)
        # V: 1 at (j1,k2), -1 at (j2,k2)
        # W: 1 at (k2,i1)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1
        V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1
        col += 1
        
        # p4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        # U: 1 at (i2,j2)
        # V: 1 at (j2,k1), -1 at (j1,k1)
        # W: 1 at (k1,i2)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i2, col] = 1
        col += 1
        
        # p5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # U: 1 at (i1,j1), 1 at (i1,j2)
        # V: 1 at (j2,k2)
        # W: 1 at (k2,i1)
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k2*n+i1, col] = 1
        col += 1
        
        # p6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # U: 1 at (i2,j1), -1 at (i1,j1)
        # V: 1 at (j1,k1), 1 at (j1,k2)
        # W: 1 at (k1,i2), 1 at (k2,i2)
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k1*n+i2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # p7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # U: 1 at (i1,j2), -1 at (i2,j2)
        # V: 1 at (j2,k1), 1 at (j2,k2)
        # W: 1 at (k1,i1), 1 at (k2,i1)
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        W[k2*n+i1, col] = 1
        col += 1
        
        return col
    
    # Part 1: 2x4 * 4x4 with Strassen (28 terms)
    # S1: i={0,1}, j={0,1}, k={0,1}
    col = strassen_7_terms(U, V, W, col, (0,1), (0,1), (0,1))
    # S2: i={0,1}, j={0,1}, k={2,3}
    col = strassen_7_terms(U, V, W, col, (0,1), (0,1), (2,3))
    # S3: i={0,1}, j={2,3}, k={0,1}
    col = strassen_7_terms(U, V, W, col, (0,1), (2,3), (0,1))
    # S4: i={0,1}, j={2,3}, k={2,3}
    col = strassen_7_terms(U, V, W, col, (0,1), (2,3), (2,3))
    
    # Part 2: 1x4 * 4x4 schoolbook (16 terms)
    # i=2, j=0..3, k=0..3
    for j in range(4):
        for k in range(4):
            U[2*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+2, col] = 1
            col += 1
    
    # Now we have col = 28 + 16 = 44 terms
    # But wait - we need to verify this is correct!
    # The issue: Strassen's 7 terms give the correct 2x2*2x2 product,
    # but we're using them for sub-problems where the (i,j,k) indices are specific.
    # Let me verify: for S1 (i={0,1}, j={0,1}, k={0,1}):
    # T[0,0,0] should get contribution 1 from the sum of all 7 terms.
    # 
    # p1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contrib 1
    # p2: U[0,0]=0, -> 0
    # p3: U[0,0]=1, V[0,1]=0 (k2=1, but V[0*p+1]=V[1]... wait V[j*p+k]
    #     V[0*4+0]=V[0]=1, V[1*4+1]=V[5]=-1 for p3
    #     For T[0,0,0]: j=0,k=0. V[0*4+0]=V[0]. In p3, V[0,3]=1? No.
    #     p3: V[j1*p+k2]=V[0*4+1]=V[1]=1, V[j2*p+k2]=V[1*4+1]=V[5]=-1
    #     For (i=0,j=0,k=0): V[0*4+0]=V[0]=0 in p3. So contrib 0.
    # p4: U[1,1]=1 (i2=1,j2=1). For i=0: U[0,0]=0. Contrib 0.
    # p5: U[0,0]=1, U[0,1]=1. V[1*4+1]=V[5]=1. For (i=0,j=0,k=0): V[0]=0. Contrib 0.
    # p6: U[1,0]=1, U[0,0]=-1. V[0,0]=1, V[0,1]=1. For (i=0,j=0,k=0): U[0,0]=-1, V[0]=1, W[0,0]=1. Contrib -1.
    # p7: U[0,1]=1, U[1,1]=-1. V[1,0]=1, V[1,1]=1. For (i=0,j=0,k=0): U[0,0]=0. Contrib 0.
    #
    # Total for T[0,0,0]: 1 + 0 + 0 + 0 + 0 + (-1) + 0 = 0. But should be 1!
    #
    # Something is wrong. Let me re-derive Strassen properly.
    #
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    #
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 + p4 - p5 + p7 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    #   = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    #   = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg - bg... 
    #   Let me redo: ae+ah+de+dh+dg-de-ah-bh+bg+bh-dg-dh
    #   = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    #   Hmm that gives ae + bg, not ae+cg.
    #
    # I'm making an error. Let me use the standard Strassen formulas:
    # C[0,0] = p1 - p5 + p7 + p4  where:
    # Actually the standard is:
    # C[0,0] = p1 + p4 - p5 + p7
    # Let me expand: (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    # = ae + bg
    # That's wrong! C[0,0] should be ae+cg.
    #
    # I think I have the Strassen formulas wrong. Let me look them up properly.
    #
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 - p5 + p7 + p4
    # Let me re-expand more carefully:
    # p1 = ae+ah+de+dh
    # p4 = dg-de
    # p5 = ah+bh
    # p7 = bg+bh-dg-dh
    #
    # p1 - p5 + p7 + p4 = ae+ah+de+dh - ah-bh + bg+bh-dg-dh + dg-de
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (-bh+bh) + (dg-dg) + bg
    # = ae + bg
    #
    # Still wrong! The issue is I'm confusing which elements go where.
    #
    # Let me use the CORRECT Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 - p5 + p7 + p4
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    #
    # C[0,1] = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh ✓
    # C[1,0] = (c+d)e + d(g-e) = ce+de+dg-de = ce+dg ✓
    # C[1,1] = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    #   = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    #   = (ae-ae) + (ah-ah) + (de-de) + (-ce+ce) + (af-af) + dh + cf
    #   = dh + cf ✓
    #
    # C[0,0] = p1 - p5 + p7 + p4
    #   = ae+ah+de+dh - ah-bh + bg+bh-dg-dh + dg-de
    #   = ae + bg
    #
    # This is STILL wrong. C[0,0] should be ae+cg.
    #
    # I think the issue is that p7 should be (b-d)(g+h) but the formula for C[0,0]
    # might be different. Let me try:
    # C[0,0] = p1 - p5 + p7 + p4 doesn't work.
    #
    # Let me try a different set of Strassen formulas. Maybe:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 - p5 + p7 + p4  -- this gives ae+bg, wrong
    #
    # Alternative: maybe C[0,0] = p1 - p2 + p6 + p7?
    # = ae+ah+de+dh - ce-de + ce+cf-ae-af + bg+bh-dg-dh
    # = (ae-ae) + ah + (de-de) + (-ce+ce) + cf + (-af) + bg + bh + (-dg) + (-dh)
    # = ah + cf - af + bg + bh - dg - dh
    # Still not ae+cg.
    #
    # I'm clearly botching the Strassen formulas. Let me just use a verified set.
    #
    # Verified Strassen (from Wikipedia):
    # p1 = (a + d) × (e + h)
    # p2 = (c + d) × e
    # p3 = a × (f − h)
    # p4 = d × (g − e)
    # p5 = (a + b) × h
    # p6 = (c − a) × (e + f)
    # p7 = (b − d) × (g + h)
    #
    # C[0,0] = p1 − p5 + p7 + p4
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 − p2 + p3 + p6
    #
    # I verified C[0,1], C[1,0], C[1,1] above and they work.
    # C[0,0] = p1 − p5 + p7 + p4 should give ae+cg.
    # Let me re-expand VERY carefully:
    # p1 = (a+d)(e+h) = ae + ah + de + dh
    # p5 = (a+b)h = ah + bh
    # p7 = (b-d)(g+h) = bg + bh - dg - dh
    # p4 = d(g-e) = dg - de
    #
    # p1 - p5 + p7 + p4:
    # ae + ah + de + dh
    # - ah - bh
    # + bg + bh - dg - dh
    # + dg - de
    #
    # = ae + (ah - ah) + (de - de) + (dh - dh) + (-bh + bh) + (dg - dg) + bg
    # = ae + bg
    #
    # This gives ae + bg, but C[0,0] = ae + cg.
    #
    # UNLESS... I have the matrix layout wrong. Maybe B is transposed in my head.
    # If B = [[e, g], [f, h]] (column-major), then:
    # C[0,0] = ae + bf, C[0,1] = ag + bh
    # C[1,0] = ce + df, C[1,1] = cg + dh
    #
    # With B = [[e,g],[f,h]]:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(g-h)  [using f-h becomes g-h]
    # p4 = d(f-e)  [using g-e becomes f-e]
    # p5 = (a+b)h
    # p6 = (c-a)(e+g)
    # p7 = (b-d)(f+h)
    #
    # C[0,0] = p1 - p5 + p7 + p4
    # = ae+ah+de+dh - ah-bh + bf+bh-df-dh + df-de
    # = ae + bf ✓
    #
    # So the issue is the layout of B. In our problem, B[j,k] where j is row, k is col.
    # So B = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]] = [[e, f], [g, h]] in row-major.
    #
    # And C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = ae + cg (where c=A[1,0], g=B[1,0])
    #
    # So with standard row-major B, C[0,0] = ae + cg.
    #
    # The Strassen formula p1-p5+p7+p4 gives ae+bg with B=[[e,f],[g,h]].
    # That means either:
    # 1. My Strassen formula is wrong, or
    # 2. I need to adjust for the layout.
    #
    # Let me just verify with actual numbers.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    #
    # a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
    # p1 = (1+4)(5+8) = 5*13 = 65
    # p2 = (3+4)*5 = 35
    # p3 = 1*(6-8) = -2
    # p4 = 4*(7-5) = 8
    # p5 = (1+2)*8 = 24
    # p6 = (3-1)*(5+6) = 2*11 = 22
    # p7 = (2-4)*(7+8) = -2*15 = -30
    #
    # C[0,0] = p1-p5+p7+p4 = 65-24-30+8 = 19 ✓
    # C[0,1] = p3+p5 = -2+24 = 22 ✓
    # C[1,0] = p2+p4 = 35+8 = 43 ✓
    # C[1,1] = p1-p2+p3+p6 = 65-35-2+22 = 50 ✓
    #
    # So the formulas ARE correct! My algebraic expansion was wrong.
    # Let me re-expand p1-p5+p7+p4 with the actual values:
    # 65 - 24 + (-30) + 8 = 19
    #
    # Symbolically: ae+ah+de+dh - ah-bh + bg+bh-dg-dh + dg-de
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (-bh+bh) + (dg-dg) + bg
    # = ae + bg
    #
    # But with numbers: a=1,e=5 -> ae=5, b=2,g=7 -> bg=14, ae+bg=19.
    # And C[0,0]=19. So ae+bg = 19 = ae+cg? No! ae+cg = 5+21=26≠19.
    #
    # Wait, C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19.
    # In my notation: a=A[0,0]=1, e=B[0,0]=5, b=A[0,1]=2, g=B[1,0]=7.
    # So C[0,0] = ae + bg (NOT ae+cg!).
    #
    # I was confusing myself. C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = ae + bg.
    # The 'c' in my earlier notation was A[1,0], not A[0,1].
    #
    # So the Strassen formulas are correct as stated. My confusion was in the variable naming.
    # Let me redo the tensor decomposition with correct understanding.
    #
    # For 2x2*2x2 with A rows {i1,i2}, A cols {j1,j2}, B rows {j1,j2}, B cols {k1,k2}:
    # a = A[i1,j1], b = A[i1,j2], c = A[i2,j1], d = A[i2,j2]
    # e = B[j1,k1], f = B[j1,k2], g = B[j2,k1], h = B[j2,k2]
    #
    # C[i1,k1] = ae + bg = A[i1,j1]*B[j1,k1] + A[i1,j2]*B[j2,k1]
    # C[i1,k2] = af + bh = A[i1,j1]*B[j1,k2] + A[i1,j2]*B[j2,k2]
    # C[i2,k1] = ce + dg = A[i2,j1]*B[j1,k1] + A[i2,j2]*B[j2,k1]
    # C[i2,k2] = cf + dh = A[i2,j1]*B[j1,k2] + A[i2,j2]*B[j2,k2]
    
    return U, V, W, col

def _build_strassen_44(n, m, p):
    """
    Build rank-44 decomposition using Strassen on 2x2 blocks.
    """
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    
    def strassen_7_terms(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen terms for 2x2*2x2 sub-problem."""
        nonlocal col
        
        # p1 = (a+d)(e+h) where a=A[i1,j1],d=A[i2,j2],e=B[j1,k1],h=B[j2,k2]
        # U: 1 at (i1,j1), 1 at (i2,j2)
        # V: 1 at (j1,k1), 1 at (j2,k2)
        # W: 1 at (k1,i1), 1 at (k2,i1), 1 at (k1,i2), 1 at (k2,i2)
        U[i1*m+j1, col] += 1
        U[i2*m+j2, col] += 1
        V[j1*p+k1, col] += 1
        V[j2*p+k2, col] += 1
        W[k1*n+i1, col] += 1
        W[k2*n+i1, col] += 1
        W[k1*n+i2, col] += 1
        W[k2*n+i2, col] += 1
        col += 1
        
        # p2 = (c+d)e where c=A[i2,j1],d=A[i2,j2],e=B[j1,k1]
        U[i2*m+j1, col] += 1
        U[i2*m+j2, col] += 1
        V[j1*p+k1, col] += 1
        W[k1*n+i2, col] += 1
        col += 1
        
        # p3 = a(f-h) where a=A[i1,j1],f=B[j1,k2],h=B[j2,k2]
        U[i1*m+j1, col] += 1
        V[j1*p+k2, col] += 1
        V[j2*p+k2, col] += -1
        W[k2*n+i1, col] += 1
        col += 1
        
        # p4 = d(g-e) where d=A[i2,j2],g=B[j2,k1],e=B[j1,k1]
        U[i2*m+j2, col] += 1
        V[j2*p+k1, col] += 1
        V[j1*p+k1, col] += -1
        W[k1*n+i2, col] += 1
        col += 1
        
        # p5 = (a+b)h where a=A[i1,j1],b=A[i1,j2],h=B[j2,k2]
        U[i1*m+j1, col] += 1
        U[i1*m+j2, col] += 1
        V[j2*p+k2, col] += 1
        W[k2*n+i1, col] += 1
        col += 1
        
        # p6 = (c-a)(e+f) where c=A[i2,j1],a=A[i1,j1],e=B[j1,k1],f=B[j1,k2]
        U[i2*m+j1, col] += 1
        U[i1*m+j1, col] += -1
        V[j1*p+k1, col] += 1
        V[j1*p+k2, col] += 1
        W[k1*n+i2, col] += 1
        W[k2*n+i2, col] += 1
        col += 1
        
        # p7 = (b-d)(g+h) where b=A[i1,j2],d=A[i2,j2],g=B[j2,k1],h=B[j2,k2]
        U[i1*m+j2, col] += 1
        U[i2*m+j2, col] += -1
        V[j2*p+k1, col] += 1
        V[j2*p+k2, col] += 1
        W[k1*n+i1, col] += 1
        W[k2*n+i1, col] += 1
        col += 1
    
    # Part 1: 2x4 * 4x4 with Strassen (28 terms)
    # S1: i={0,1}, j={0,1}, k={0,1}
    strassen_7_terms(0, 1, 0, 1, 0, 1)
    # S2: i={0,1}, j={0,1}, k={2,3}
    strassen_7_terms(0, 1, 0, 1, 2, 3)
    # S3: i={0,1}, j={2,3}, k={0,1}
    strassen_7_terms(0, 1, 2, 3, 0, 1)
    # S4: i={0,1}, j={2,3}, k={2,3}
    strassen_7_terms(0, 1, 2, 3, 2, 3)
    
    # Part 2: 1x4 * 4x4 schoolbook (16 terms)
    for j in range(4):
        for k in range(4):
            U[2*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+2, col] = 1
            col += 1
    
    return U, V, W

def _check_strassen_44():
    """Verify the Strassen-44 decomposition is exact."""
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen_44(n, m, p)
    max_err = _max_error(U, V, W, n, m, p)
    print(f"Strassen-44 max error: {max_err}")
    if max_err < 1e-9:
        print("Strassen-44 is EXACT!")
    else:
        # Find which entries are wrong
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        print(f"  T[{i},{j},{k}] = {val} (expected 1.0)")
    return max_err

def _try_rank_als(n, m, p, rank, max_restarts=20, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using ALS."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    inits = [
        lambda s: (np.random.randn(n*m, rank)*0.3, np.random.randn(m*p, rank)*0.3, np.random.randn(p*n, rank)*0.3),
        lambda s: _build_strassen_44(n, m, p) if rank >= 44 else (np.zeros((n*m,rank)), np.zeros((m*p,rank)), np.zeros((p*n,rank))),
    ]
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        strat = restart % len(inits)
        U, V, W = inits[strat](seed + restart * 7 if seed is not None else restart * 7)
        
        if U.shape[1] != rank:
            continue
            
        U, V, W = _als_iterate_fast(U, V, W, n, m, p, max_iter=300)
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if max_err < 1e-7:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    # First: try the constructive Strassen-44 decomposition
    U_44, V_44, W_44 = _build_strassen_44(n, m, p)
    if _check_exact(U_44, V_44, W_44, n, m, p, tol=1e-9):
        # Verify it's valid and try to snap
        result = _try_snap_and_check(U_44, V_44, W_44, n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Return as-is if already exact with integer/half-integer entries
            return {"U": U_44.tolist(), "V": V_44.tolist(), "W": W_44.tolist()}
    
    # If Strassen-44 doesn't work, fall back to schoolbook
    # But first try to find a better rank via search
    best_result = None
    best_rank = schoolbook_rank
    
    for rank in range(44, 33, -1):
        if time.time() > deadline - 25:
            break
        
        result = _try_rank_als(n, m, p, rank, max_restarts=8, seed=rank * 1000, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
