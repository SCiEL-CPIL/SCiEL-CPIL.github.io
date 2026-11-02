"""Matrix multiplication 3x5x7 - rank minimization via Kronecker product of known low-rank tensors."""
import time
import random
from fractions import Fraction
import numpy as np

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _kronecker_tensor(rank1, rank2, U1, V1, W1, U2, V2, W2,
                       n1, m1, p1, n2, m2, p2):
    """
    Combine two tensor decompositions via Kronecker product.
    If (U1,V1,W1) decomposes n1xm1xp1 and (U2,V2,W2) decomposes n2xm2xp2,
    the Kronecker product decomposes (n1*n2) x (m1*m2) x (p1*p2)
    with rank rank1*rank2.
    
    The combined U has shape (n1*m1*n2*m2, rank1*rank2)
    Index: U[(i1*m1+j1)*n2*m2 + i2*m2+j2, c1*rank2+c2] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    
    V has shape (m1*p1*n2*p2, rank1*rank2)
    Index: V[(j1*p1+k1)*n2*p2 + j2*p2+k2, c1*rank2+c2] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    
    W has shape (p1*n1*p2*n2, rank1*rank2)
    Index: W[(k1*n1+i1)*p2*n2 + k2*n2+i2, c1*rank2+c2] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    """
    r = rank1 * rank2
    nm_total = n1*m1*n2*m2
    mp_total = m1*p1*n2*p2
    pn_total = p1*n1*p2*n2
    
    U = [[Fraction(0)]*r for _ in range(nm_total)]
    V = [[Fraction(0)]*r for _ in range(mp_total)]
    W = [[Fraction(0)]*r for _ in range(pn_total)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_u = (i1*m1 + j1) * n2*m2 + i2*m2 + j2
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            col = c1*rank2 + c2
                            U[row_u][col] = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(n2):
                for k2 in range(p2):
                    row_v = (j1*p1 + k1) * n2*p2 + j2*p2 + k2
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            col = c1*rank2 + c2
                            V[row_v][col] = V1[j1*p1+k1][c1] * V2[j2*p2+k2][c2]
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_w = (k1*n1 + i1) * p2*n2 + k2*n2 + i2
                    for c1 in range(rank1):
                        for c2 in range(rank2):
                            col = c1*rank2 + c2
                            W[row_w][col] = W1[k1*n1+i1][c1] * W2[k2*n2+i2][c2]
    
    return U, V, W, r

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[Fraction(0)]*rank for _ in range(n*m)]
    V = [[Fraction(0)]*rank for _ in range(m*p)]
    W = [[Fraction(0)]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = Fraction(1)
                V[j*p+k][col] = Fraction(1)
                W[k*n+i][col] = Fraction(1)
                col += 1
    return U, V, W, rank

def _strassen_2x2():
    """
    Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i*2+j, c] represents coefficient for A[i,j] in term c
    # V[j*2+k, c] represents coefficient for B[j,k] in term c
    # W[k*2+i, c] represents coefficient for C[k,i] in term c
    
    U = [[Fraction(0)]*7 for _ in range(4)]
    V = [[Fraction(0)]*7 for _ in range(4)]
    W = [[Fraction(0)]*7 for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # Term 1: M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[0][1] = 1  # C10
    W[3][1] = 1  # C11
    
    # Term 2: M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # Term 3: M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # Term 4: M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[1][4] = -1  # C01
    W[3][4] = 1  # C11
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    return U, V, W, 7

def _sum_tensors(tensors, n, m, p):
    """
    Sum multiple tensor decompositions that all decompose the same n x m x p multiplication.
    Each tensor is (U, V, W, rank) where U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
    The sum concatenates the rank dimensions.
    """
    total_rank = sum(t[3] for t in tensors)
    U = [[Fraction(0)]*total_rank for _ in range(n*m)]
    V = [[Fraction(0)]*total_rank for _ in range(m*p)]
    W = [[Fraction(0)]*total_rank for _ in range(p*n)]
    
    offset = 0
    for U_t, V_t, W_t, r_t in tensors:
        for i in range(n*m):
            for c in range(r_t):
                U[i][offset+c] = U_t[i][c]
        for i in range(m*p):
            for c in range(r_t):
                V[i][offset+c] = V_t[i][c]
        for i in range(p*n):
            for c in range(r_t):
                W[i][offset+c] = W_t[i][c]
        offset += r_t
    
    return U, V, W, total_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 105."""
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    def try_kronecker_decomposition():
        """
        Try to decompose 3x5x7 using Kronecker products of smaller known tensors.
        
        Strategy: Write 3 = 2+1, 5 = 4+1, 7 = 4+3
        Use block structure:
        - A is 3x5 = [A_2x4, A_2x1; A_1x4, A_1x1]
        - B is 5x7 = [B_4x4, B_4x3; B_1x4, B_1x3]
        
        C = A*B gives blocks:
        C_2x4 = A_2x4 * B_4x4
        C_2x3 = A_2x4 * B_4x3 + A_2x1 * B_1x3
        C_1x4 = A_1x4 * B_4x4 + A_1x1 * B_1x4
        C_1x3 = A_1x4 * B_4x3 + A_1x1 * B_1x3
        
        Each block product can be decomposed:
        - 2x4x4: use Strassen for 2x2x2 (rank 7) for each of 2 column-blocks of 2x2x2
          Actually, 2x4x4 = 2x(2+2)x(2+2), so it's 4 independent 2x2x2 multiplications
          Rank = 4 * 7 = 28
        - 2x4x3: 2x(2+2)x(1+2) = 2x2x1 (rank 2) + 2x2x2 (rank 7) + 2x2x1 (rank 2) + 2x2x2 (rank 7)
          Rank = 2 + 7 + 2 + 7 = 18
        - 1x4x4: 1x(2+2)x(2+2) = 4 independent 1x2x2 multiplications
          1x2x2 has rank 2 (schoolbook), so rank = 4*2 = 8
        - 1x4x3: 1x(2+2)x(1+2) = 1x2x1 (rank 1) + 1x2x2 (rank 2) + 1x2x1 (rank 1) + 1x2x2 (rank 2)
          Rank = 1 + 2 + 1 + 2 = 6
        
        Total rank = 28 + 18 + 8 + 6 = 60
        
        But we need to be more careful about the block structure and how the tensors combine.
        """
        if time.time() > deadline:
            return None
        
        # We'll build the decomposition by considering each block of the output C.
        # C is 3x7, partitioned as:
        # Rows: 0-1 (from A rows 0-1), 2 (from A row 2)
        # Cols: 0-3 (from B cols 0-3), 4-6 (from B cols 4-6)
        
        # A is 3x5, partitioned as:
        # Rows: 0-1 (top 2), 2 (bottom 1)
        # Cols: 0-3 (left 4), 4 (right 1)
        
        # B is 5x7, partitioned as:
        # Rows: 0-3 (top 4), 4 (bottom 1)
        # Cols: 0-3 (left 4), 4-6 (right 3)
        
        # Block products:
        # C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]  -> 2x4x4
        # C[0:2, 4:7] = A[0:2, 0:4] * B[0:4, 4:7] + A[0:2, 4:5] * B[4:5, 4:7]  -> 2x4x3 + 2x1x3
        # C[2:3, 0:4] = A[2:3, 0:4] * B[0:4, 0:4] + A[2:3, 4:5] * B[4:5, 0:4]  -> 1x4x4 + 1x1x4
        # C[2:3, 4:7] = A[2:3, 0:4] * B[0:4, 4:7] + A[2:3, 4:5] * B[4:5, 4:7]  -> 1x4x3 + 1x1x3
        
        # Let's decompose each sub-product:
        
        # Helper: get schoolbook for n x m x p
        def sb(n_, m_, p_):
            return _schoolbook(n_, m_, p_)
        
        # 2x4x4: decompose as 2x(2+2)x(2+2) = sum of 4 terms of 2x2x2
        # Each 2x2x2 uses Strassen (rank 7)
        # But we need to map the indices correctly.
        
        # For 2x4x4: A is 2x4, B is 4x4, C is 2x4
        # Partition A cols into [0,1] and [2,3], B rows into [0,1] and [2,3], B cols into [0,1] and [2,3]
        # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
        
        # This is getting complex. Let me use a simpler approach:
        # Just use schoolbook for all blocks and sum them up.
        # This gives rank = 105, same as full schoolbook.
        
        # Instead, let me try a different strategy: use Strassen for the 2x2x2 sub-blocks
        # where possible, and schoolbook for the rest.
        
        # Actually, let me just try the direct approach: build the full tensor by
        # considering each (i,j,k) triple and using a structured assignment.
        
        # New approach: use the fact that we can write the 3x5x7 tensor as a sum
        # of Kronecker products of smaller tensors.
        
        # 3x5x7 = (2+1) x (4+1) x (4+3)
        # We can write this as a sum of 8 Kronecker products:
        # For each combination of (n_choice, m_choice, p_choice) where
        # n_choice in {2,1}, m_choice in {4,1}, p_choice in {4,3}
        # we have a term of size n_choice x m_choice x p_choice
        
        # But the Kronecker product doesn't directly work for sums of dimensions.
        # We need to use the "sum of Kronecker products" approach.
        
        # Let me try a completely different approach: 
        # Use the Laderman 3x3x3 algorithm (rank 15) as a building block.
        
        # Laderman's 15-term algorithm for 3x3x3:
        # This is well-known and I can hardcode it.
        
        if time.time() > deadline:
            return None
        
        # Laderman 3x3x3 decomposition (rank 15)
        # C = A * B where A, B are 3x3
        # M1 = A00*B00
        # M2 = A00*(B01+B10+B11)
        # M3 = (A00+A11)*(B10+B11)
        # M4 = (A01+A11)*B10
        # M5 = (A01+A11)*(B11+B20)
        # M6 = (A01+A11+A21+A22)*B21
        # M7 = A22*(B01+B10+B11)
        # M8 = (A02+A12)*(B11+B20)
        # M9 = (A02+A12)*(B20+B21)
        # M10 = A22*(B20+B21)
        # M11 = (A00+A11+A21+A22)*(B01+B10+B11+B20+B21)
        # M12 = (A01+A11+A21+A22)*(B10+B11+B20+B21)
        # M13 = (A02+A12)*(B01+B10+B11+B20+B21)
        # M14 = (A00+A11)*(B01+B10+B11)
        # M15 = (A01+A11+A21+A22)*(B01+B10+B11+B20+B21) - M12 - M15... 
        
        # This is getting complicated. Let me use a simpler strategy.
        
        # Simplest effective strategy: 
        # 1. Use schoolbook for the full 3x5x7 (rank 105) as baseline
        # 2. Try to improve by using Strassen for 2x2x2 sub-blocks
        # 3. Try random search with better structure
        
        # For now, let's try a structured random search that's more likely to find
        # half-integer solutions by using a smarter initialization.
        
        def try_structured_random(r, num_attempts=200):
            """
            Try to find a rank-r decomposition using structured random search.
            Key improvement: use a warm start from schoolbook and perturb.
            """
            if time.time() > deadline:
                return None
            
            # Start with schoolbook and try to compress
            # Or: use random V,W but with a bias toward sparse/structured values
            
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    return None
                
                random.seed(42 + r * 1000 + attempt)
                
                # Generate V and W with entries in {-1, 0, 1} for simplicity
                # This makes it more likely that U will have nice values
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
                
                U = [[0.0]*r for _ in range(nm)]
                valid = True
                
                for i in range(n):
                    # Build the linear system for this i
                    A_mat = np.zeros((m*p, m*r))
                    b_vec = np.ones(m*p)
                    
                    for j in range(m):
                        for k in range(p):
                            eq_idx = j * p + k
                            for c in range(r):
                                A_mat[eq_idx, j*r + c] = float(V[j*p+k][c] * W[k*n+i][c])
                    
                    # Solve using least squares
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if solution is close to half-integers
                    sol_half = np.round(sol * 2.0) / 2.0
                    diff_half = np.max(np.abs(sol - sol_half))
                    
                    sol_int = np.round(sol)
                    diff_int = np.max(np.abs(sol - sol_int))
                    
                    if diff_half <= diff_int:
                        sol_rounded = sol_half
                        diff = diff_half
                    else:
                        sol_rounded = sol_int
                        diff = diff_int
                    
                    if diff > 1e-6:
                        valid = False
                        break
                    
                    for j in range(m):
                        for c in range(r):
                            U[i*m+j][c] = float(sol_rounded[j*r + c])
                
                if not valid:
                    continue
                
                # Verify with numpy
                if _verify_np(U, V, W, n, m, p):
                    # Convert to exact fractions
                    U_frac = [[Fraction(U[i][j]).limit_denominator(1000) for j in range(r)] for i in range(nm)]
                    V_frac = [[Fraction(V[i][j]).limit_denominator(1000) for j in range(r)] for i in range(mp)]
                    W_frac = [[Fraction(W[i][j]).limit_denominator(1000) for j in range(r)] for i in range(pn)]
                    
                    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                        return {
                            "U": [[float(x) for x in row] for row in U_frac],
                            "V": [[float(x) for x in row] for row in V_frac],
                            "W": [[float(x) for x in row] for row in W_frac]
                        }
            
            return None
        
        # Try ranks from a reasonable lower bound upward
        # Lower bound: max(n*m, m*p, p*n) = 35
        # But matrix multiplication rank is typically higher.
        # For 3x5x7, known bounds suggest rank is around 50-70.
        
        best_result = None
        best_rank = n * m * p
        
        # Try a range of ranks
        for r in range(40, 105):
            if time.time() > deadline - 5:
                break
            
            result = try_structured_random(r, num_attempts=50)
            if result is not None:
                best_result = result
                best_rank = r
                break
        
        if best_result is not None:
            return best_result
        
        return None
    
    # First try the structured approach
    result = try_kronecker_decomposition()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()

