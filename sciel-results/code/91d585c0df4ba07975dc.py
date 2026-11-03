import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def build_4x4x4_strassen_blocks():
        """
        Build 4x4x4 decomposition using 2x2 Strassen blocks with cross-block terms.
        View A, B, C as 2x2 blocks of 2x2 matrices.
        C = A*B where:
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
        
        We can use Strassen for the outer 2x2 structure, but each "element" is a 2x2 matrix.
        This requires 7 products of 2x2 matrices, each done with rank 7, giving 49.
        But we can do better by sharing structure.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        # For 4x4x4: i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
        # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        
        # Use Kronecker: rank = 7*7 = 49
        r_total = 49
        U4 = [[0]*r_total for _ in range(16)]
        V4 = [[0]*r_total for _ in range(16)]
        W4 = [[0]*r_total for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4
    
    def build_4x4x7_combined():
        """
        Build 4x4x7 by combining 4x4x4 Strassen-Kronecker with schoolbook for extra columns.
        """
        U4, V4, W4 = build_4x4x4_strassen_blocks()
        r4 = len(U4[0])
        
        # Extra 3 columns: schoolbook
        r_extra = 4 * 4 * 3  # 48
        r_total = r4 + r_extra
        
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # Copy 4x4x4 part
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # Schoolbook for k in [4,7)
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j][col_offset] = 1
                    V[j*7 + k][col_offset] = 1
                    W[k*4 + i][col_offset] = 1
                    col_offset += 1
        
        return U, V, W
    
    def try_eliminate_column(U_np, V_np, W_np, col_idx, r):
        """
        Try to eliminate column col_idx by expressing it as a linear combination
        of other columns. Returns (new_U, new_V, new_W, success) or None.
        
        For column c to be eliminable, we need:
        U[:,c] = sum_{c' != c} alpha_c' * U[:,c']
        V[:,c] = sum_{c' != c} alpha_c' * V[:,c']  
        W[:,c] = sum_{c' != c} alpha_c' * W[:,c']
        
        This is a system of linear equations. If it has a solution, we can eliminate.
        """
        if r <= 1:
            return None
        
        # Build the system: for each row in U, V, W, we have an equation
        # U[row, c] = sum_{c' != c} alpha[c'] * U[row, c']
        # V[row, c] = sum_{c' != c} alpha[c'] * V[row, c']
        # W[row, c] = sum_{c' != c} alpha[c'] * W[row, c']
        
        other_cols = [i for i in range(r) if i != col_idx]
        n_unknowns = len(other_cols)
        
        # Build augmented matrix
        # Each row of U, V, W gives one equation
        num_rows_U = U_np.shape[0]
        num_rows_V = V_np.shape[0]
        num_rows_W = W_np.shape[0]
        total_eqs = num_rows_U + num_rows_V + num_rows_W
        
        A_mat = np.zeros((total_eqs, n_unknowns))
        b_vec = np.zeros(total_eqs)
        
        # U equations
        for i in range(num_rows_U):
            for idx, c_prime in enumerate(other_cols):
                A_mat[i, idx] = U_np[i, c_prime]
            b_vec[i] = U_np[i, col_idx]
        
        # V equations
        for i in range(num_rows_V):
            for idx, c_prime in enumerate(other_cols):
                A_mat[num_rows_U + i, idx] = V_np[i, c_prime]
            b_vec[num_rows_U + i] = V_np[i, col_idx]
        
        # W equations
        for i in range(num_rows_W):
            for idx, c_prime in enumerate(other_cols):
                A_mat[num_rows_U + num_rows_V + i, idx] = W_np[i, c_prime]
            b_vec[num_rows_U + num_rows_V + i] = W_np[i, col_idx]
        
        # Solve least squares
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return None
        
        # Check if the solution actually satisfies the equations
        A_alpha = A_mat @ alpha
        if np.max(np.abs(A_alpha - b_vec)) > 1e-9:
            return None
        
        # Verify that the elimination preserves the tensor product structure
        # The key check: for all (i,j,k), the sum over columns of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must remain 1
        # After elimination, column c is removed and its contribution is absorbed into other columns.
        # But this only works if the linear combination preserves the trilinear form.
        # Actually, simple linear combination of columns doesn't preserve the trilinear form!
        # We need: for all (i,j,k):
        #   sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # If we replace column c with alpha*c', the trilinear form changes.
        # So this approach only works if the column is truly redundant in the tensor sense.
        
        # A column is redundant if there exists alpha such that for ALL (i,j,k):
        #   sum_{c' != c} alpha[c'] * (U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]) = 0
        # This is a much harder condition. Let's just check validity after elimination.
        
        # Create new arrays without column c
        mask = [i for i in range(r) if i != col_idx]
        U_new = U_np[:, mask].copy()
        V_new = V_np[:, mask].copy()
        W_new = W_np[:, mask].copy()
        
        # Add the linear combination to the remaining columns
        for idx, c_prime in enumerate(other_cols):
            U_new[:, idx] += alpha[idx] * U_np[:, col_idx]
            V_new[:, idx] += alpha[idx] * V_np[:, col_idx]
            W_new[:, idx] += alpha[idx] * W_np[:, col_idx]
        
        # Check validity
        r_new = len(U_new[0])
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r_new):
                        s += U_new[i*m + j, c] * V_new[j*p + k, c] * W_new[k*n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_new, V_new, W_new, True
        
        return None
    
    def greedy_rank_reduction(U, V, W, max_time=60.0):
        """
        Greedily try to eliminate columns one by one.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        start = time.time()
        
        progress = True
        while progress and r > 1:
            progress = False
            if time.time() - start > max_time:
                break
            
            # Try each column
            for col_idx in range(r):
                if time.time() - start > max_time:
                    break
                
                result = try_eliminate_column(U_np, V_np, W_np, col_idx, r)
                if result is not None:
                    U_np, V_np, W_np, _ = result
                    r = U_np.shape[1]
                    progress = True
                    break
        
        return U_np.tolist(), V_np.tolist(), W_np.tolist()
    
    def randomized_rank_reduction(U, V, W, max_time=60.0):
        """
        Randomized approach: try random linear combinations of pairs of columns
        to see if we can merge them.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        start = time.time()
        best_rank = r
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
        
        random.seed(42)
        
        while r > 1:
            if time.time() - start > max_time:
                break
            
            # Try random pairs
            found = False
            for _ in range(200):
                if time.time() - start > max_time:
                    break
                if r <= 1:
                    break
                
                c1 = random.randint(0, r-1)
                c2 = random.randint(0, r-1)
                if c1 == c2:
                    continue
                
                # Try to express column c2 as a scalar multiple of column c1 plus perturbation
                # Or try: new_c1 = c1 + lambda*c2, remove c2
                for lam in [0.5, -0.5, 0.25, -0.25, 1.0, -1.0]:
                    U_test = U_np.copy()
                    V_test = V_np.copy()
                    W_test = W_np.copy()
                    
                    U_test[:, c1] = U_test[:, c1] + lam * U_test[:, c2]
                    V_test[:, c1] = V_test[:, c1] + lam * V_test[:, c2]
                    W_test[:, c1] = W_test[:, c1] + lam * W_test[:, c2]
                    
                    mask = [i for i in range(r) if i != c2]
                    U_test = U_test[:, mask]
                    V_test = V_test[:, mask]
                    W_test = W_test[:, mask]
                    
                    # Check validity
                    r_test = U_test.shape[1]
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(r_test):
                                    s += U_test[i*m + j, c] * V_test[j*p + k, c] * W_test[k*n + i, c]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        U_np, V_np, W_np = U_test, V_test, W_test
                        r = r_test
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                break
        
        if r < best_rank:
            best_U, best_V, best_W = U_np, V_np, W_np
            best_rank = r
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Build the combined 4x4x7 decomposition
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_comb, V_comb, W_comb = build_4x4x7_combined()
            r_comb = len(U_comb[0])
            
            if check_valid(U_comb, V_comb, W_comb):
                if r_comb < best_rank:
                    best_result = {"U": U_comb, "V": V_comb, "W": W_comb}
                    best_rank = r_comb
        except Exception as e:
            pass
    
    # Apply greedy rank reduction
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_red, V_red, W_red = greedy_rank_reduction(
                best_result["U"], best_result["V"], best_result["W"],
                max_time=min(40.0, WALL_DEADLINE - SAFETY_MARGIN - (time.time() - start_time))
            )
            r_red = len(U_red[0])
            
            if check_valid(U_red, V_red, W_red):
                if r_red < best_rank:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = r_red
        except Exception as e:
            pass
    
    # Apply randomized rank reduction
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_rnd, V_rnd, W_rnd, r_rnd = randomized_rank_reduction(
                best_result["U"], best_result["V"], best_result["W"],
                max_time=min(40.0, WALL_DEADLINE - SAFETY_MARGIN - (time.time() - start_time))
            )
            
            if check_valid(U_rnd, V_rnd, W_rnd):
                if r_rnd < best_rank:
                    best_result = {"U": U_rnd, "V": V_rnd, "W": W_rnd}
                    best_rank = r_rnd
        except Exception as e:
            pass
    
    return best_result
