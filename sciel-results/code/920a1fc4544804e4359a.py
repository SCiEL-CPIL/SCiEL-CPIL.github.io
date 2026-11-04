import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    deadline = time.time() + TIME_LIMIT - 5  # safety margin
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=np.float64))
        V_cols.append(np.array(v_vec, dtype=np.float64))
        W_cols.append(np.array(w_vec, dtype=np.float64))
    
    def strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 rank-1 terms for Strassen 2x2*2x2."""
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        def term(a_entries, b_entries, c_entries):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in a_entries:
                u[idx] = val
            for idx, val in b_entries:
                v[idx] = val
            for idx, val in c_entries:
                w[idx] = val
            add_term(u, v, w)
        
        # M1 = (A11+A22)*(B11+B22)
        term([(ai(0,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0),(ci(1,1),1.0)])
        # M2 = (A21+A22)*B11
        term([(ai(1,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0)], [(ci(1,0),1.0),(ci(1,1),-1.0)])
        # M3 = A11*(B12-B22)
        term([(ai(0,0),1.0)], [(bi(0,1),1.0),(bi(1,1),-1.0)], [(ci(0,1),1.0),(ci(1,1),1.0)])
        # M4 = A22*(B21-B11)
        term([(ai(1,1),1.0)], [(bi(1,0),1.0),(bi(0,0),-1.0)], [(ci(0,0),1.0),(ci(1,0),1.0)])
        # M5 = (A11+A12)*B22
        term([(ai(0,0),1.0),(ai(0,1),1.0)], [(bi(1,1),1.0)], [(ci(0,0),-1.0),(ci(0,1),1.0)])
        # M6 = (A21-A11)*(B11+B12)
        term([(ai(1,0),1.0),(ai(0,0),-1.0)], [(bi(0,0),1.0),(bi(0,1),1.0)], [(ci(1,1),1.0)])
        # M7 = (A12-A22)*(B21+B22)
        term([(ai(0,1),1.0),(ai(1,1),-1.0)], [(bi(1,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0)])
    
    def strassen_2x2_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 rank-1 terms for a 2x2 * 2x1 product using Strassen-style.
        Actually for 2x2 * 2x1, the rank is 4 (schoolbook). But we can use
        a 7-term construction if we embed into 2x2*2x2 and ignore one column.
        Let's just use schoolbook rank 4 for 2x2*2x1.
        """
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        # Schoolbook: 4 terms
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # But we can do better with Strassen on the embedded 2x2*2x2.
        # Actually, let's just use 4 schoolbook terms for simplicity.
        for i in range(2):
            for j in range(2):
                u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
                u[ai(i,j)] = 1.0
                v[bi(i,0)] = 1.0
                w[ci(i,0)] = 1.0
                add_term(u, v, w)
    
    def strassen_2x2_2x3(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add terms for 2x2 * 2x3 product.
        Split B into two 2x1 columns.
        Use Strassen for the first 2x2 part (embedded in 2x2*2x2) and schoolbook for the rest.
        Actually, let's split B into B1 (2x2) and B2 (2x1).
        X * B = X * B1 + X * B2
        X * B1: use Strassen 2x2*2x2 (7 terms)
        X * B2: use schoolbook 2x2*2x1 (4 terms)
        Total: 11 terms
        """
        # First 2 columns: Strassen
        strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c)
        # Last 1 column: schoolbook
        strassen_2x2_2x1(a_r, a_c, b_r, b_c + 2, c_r, c_c + 2)
    
    # Build the decomposition by tiling
    # A is 4x4, B is 4x6
    # Split A into 2x2 blocks, B into 2x3 blocks
    # C_ab = A_a0 * B_0b + A_a1 * B_1b
    # Each term is (2x2)*(2x3) which takes 11 terms
    # Total: 4 blocks * 2 * 11 = 88 terms? That's worse than 84.
    
    # Let's stick with the rank-84 construction from the current program
    # which uses 4 Strassen 2x2*2x2 blocks for the first 4 columns of B
    # and then handle the last 2 columns separately.
    
    # Actually, the current program does:
    # for block in range(3):
    #     b_col_offset = block * 2
    #     c_col_offset = block * 2
    #     strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
    #     strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
    #     strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
    #     strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    # This gives 3 * 4 * 7 = 84 terms.
    # But this only covers B columns 0-5 (3 blocks of 2 columns each).
    # Wait, B has 6 columns, so 3 blocks of 2 columns each is correct.
    # But the A matrix is 4x4, so we need to cover all 4 rows and 4 columns of A.
    # The 4 Strassen calls per block cover:
    #   (0,0): A rows 0-1, cols 0-1; B rows 0-1, cols b_col_offset..b_col_offset+1; C rows 0-1, cols c_col_offset..c_col_offset+1
    #   (0,2): A rows 0-1, cols 2-3; B rows 0-1, cols b_col_offset..b_col_offset+1; C rows 0-1, cols c_col_offset..c_col_offset+1
    #   (2,0): A rows 2-3, cols 0-1; B rows 2-3, cols b_col_offset..b_col_offset+1; C rows 2-3, cols c_col_offset..c_col_offset+1
    #   (2,2): A rows 2-3, cols 2-3; B rows 2-3, cols b_col_offset..b_col_offset+1; C rows 2-3, cols c_col_offset..c_col_offset+1
    # This covers all of A and B for each pair of B columns.
    # So 3 blocks * 4 Strassen = 12 Strassen calls * 7 terms = 84 terms.
    # This is correct.
    
    for block in range(3):
        b_col_offset = block * 2
        c_col_offset = block * 2
        
        strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
        strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    rank = len(U_cols)
    
    if time.time() > deadline:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Build target tensor
    T = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)
    
    best_U = np.column_stack(U_cols)
    best_V = np.column_stack(V_cols)
    best_W = np.column_stack(W_cols)
    best_rank = rank
    best_err = err_of(best_U, best_V, best_W)
    
    # More aggressive rank reduction: try to find linear dependencies
    # For each column, check if it can be expressed as a combination of others
    # by solving least squares on the tensor representation
    
    # Flatten the tensor contribution of each column
    # Column r contributes: outer(U[:,r], V[:,r], W[:,r])
    # If col r = sum_s c_s * col_s, then the tensor contributions match
    
    # Approach: for each column r, try to find c such that
    # U[:,r] ≈ U[:,mask] @ c, V[:,r] ≈ V[:,mask] @ c, W[:,r] ≈ W[:,mask] @ c
    # Use all three constraints jointly
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for r in range(best_rank):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(best_rank) if i != r]
            if len(mask) < 3:
                break
            
            # Try to find c such that the tensor contribution of col r
            # is approximated by sum of others
            # Flatten: tensor[r] = outer(u_r, v_r, w_r)
            # We want: outer(u_r, v_r, w_r) ≈ sum_s c_s * outer(u_s, v_s, w_s)
            # This is a bilinear constraint, hard to solve directly.
            
            # Simpler: just try removing and check error
            U_trial = best_U[:, mask]
            V_trial = best_V[:, mask]
            W_trial = best_W[:, mask]
            
            err_trial = err_of(U_trial, V_trial, W_trial)
            
            if err_trial < 1e-12:
                best_U = U_trial
                best_V = V_trial
                best_W = W_trial
                best_rank -= 1
                best_err = err_trial
                improved = True
                break
        
        if not improved:
            break
    
    # Try a different approach: perturb and re-optimize
    # Use randomized restarts with small perturbations to escape local minima
    
    rng = np.random.RandomState(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start from current best, try removing random subsets
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        rank_cur = best_rank
        err_cur = best_err
        
        # Try removing one random column at a time
        for step in range(rank_cur):
            if time.time() > deadline:
                break
            if rank_cur <= 1:
                break
            
            r = rng.randint(rank_cur)
            mask = [i for i in range(rank_cur) if i != r]
            
            U_trial = U_cur[:, mask]
            V_trial = V_cur[:, mask]
            W_trial = W_cur[:, mask]
            
            err_trial = err_of(U_trial, V_trial, W_trial)
            
            if err_trial < 1e-12:
                U_cur = U_trial
                V_cur = V_trial
                W_cur = W_trial
                rank_cur -= 1
                err_trial = 0.0
        
        if rank_cur < best_rank:
            best_U = U_cur
            best_V = V_cur
            best_W = W_cur
            best_rank = rank_cur
            best_err = err_cur
    
    # Verify correctness
    if check(best_U, best_V, best_W):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback to original
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    return {"U": to_list(np.column_stack(U_cols)), "V": to_list(np.column_stack(V_cols)), "W": to_list(np.column_stack(W_cols))}
