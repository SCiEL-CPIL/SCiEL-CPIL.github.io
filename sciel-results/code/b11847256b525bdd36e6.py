import numpy as np
import time
from fractions import Fraction

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback (rank 60)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Strassen-based decomposition for 2x5x6
    # Strategy: Split the 5 middle dimension into 2+3, and 6 output into 2+4
    # Use Strassen on 2x2 sub-blocks where possible
    # 
    # For M(2,5,6): 
    #   A is 2x5, B is 5x6, C is 2x6
    #   Split A into [A1 (2x2), A2 (2x3)]
    #   Split B into [[B1 (2x2), B2 (2x4)], [B3 (3x2), B4 (3x4)]]
    #   Then C = A1*B1 + A2*B3 for the 2x2 part, and A1*B2 + A2*B4 for the 2x4 part
    #
    # The 2x2*2x2 part can use Strassen (rank 7 instead of 8)
    # The 2x2*2x4 part: each 2x2*2x2 block uses Strassen, so 2 blocks * 7 = 14
    # The 2x3*3x2 part: rank 6 (schoolbook)
    # The 2x3*3x4 part: each 2x3*3x2 block... let's think differently
    #
    # Alternative: Use the known result that M(2,5,6) <= 54
    # This can be achieved by:
    # - Treating the 5 as 2+3 and 6 as 2+4
    # - For the 2x2 * 2x2 blocks: use Strassen (rank 7 each)
    # - For the 2x2 * 2x4 blocks: 2 independent 2x2*2x2 Strassen (rank 7 each) = 14
    # - For the 2x3 * 3x2 blocks: rank 6
    # - For the 2x3 * 3x4 blocks: 2 independent 2x3*3x2... 
    #
    # Let me construct it more carefully using the block structure.
    
    # Build Strassen decomposition for 2x5x6
    # U shape: (10, rank), V shape: (30, rank), W shape: (12, rank)
    
    # We'll build up the factor matrices column by column
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one rank-1 term. u_vals: dict {row: val}, v_vals: dict {row: val}, w_vals: dict {row: val}"""
        u_col = np.zeros(nm)
        v_col = np.zeros(mp)
        w_col = np.zeros(pn)
        for row, val in u_vals.items():
            u_col[row] = val
        for row, val in v_vals.items():
            v_col[row] = val
        for row, val in w_vals.items():
            w_col[row] = val
        U_list.append(u_col)
        V_list.append(v_col)
        W_list.append(w_col)
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x2 at positions:
    # A rows: i1,i2 (0,1), cols: j1,j2
    # B rows: j1,j2, cols: k1,k2
    # C rows: i1,i2, cols: k1,k2
    
    def strassen_2x2(i_base, j_base, k_base):
        """Add Strassen terms for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2] -> C[i_base:i_base+2, k_base:k_base+2]"""
        # M1 = (A11 + A22)(B11 + B22)
        add_term(
            {i_base*m + j_base: 1, (i_base+1)*m + (j_base+1): 1},
            {j_base*p + k_base: 1, (j_base+1)*p + (k_base+1): 1},
            {(k_base)*n + i_base: 1, (k_base+1)*n + (i_base+1): 1}
        )
        # M2 = (A21 + A22)B11
        add_term(
            {(i_base+1)*m + j_base: 1, (i_base+1)*m + (j_base+1): 1},
            {j_base*p + k_base: 1},
            {(k_base)*n + (i_base+1): 1}
        )
        # M3 = A11(B21 - B22)
        add_term(
            {i_base*m + j_base: 1},
            {(j_base+1)*p + k_base: 1, (j_base+1)*p + (k_base+1): -1},
            {(k_base)*n + i_base: 1}
        )
        # M4 = A22(B12 - B11)
        add_term(
            {(i_base+1)*m + (j_base+1): 1},
            {j_base*p + (k_base+1): 1, j_base*p + k_base: -1},
            {(k_base+1)*n + (i_base+1): 1}
        )
        # M5 = (A11 + A12)B22
        add_term(
            {i_base*m + j_base: 1, i_base*m + (j_base+1): 1},
            {(j_base+1)*p + (k_base+1): 1},
            {(k_base+1)*n + i_base: 1}
        )
        # M6 = (A21 - A11)(B11 + B12)
        add_term(
            {(i_base+1)*m + j_base: 1, i_base*m + j_base: -1},
            {j_base*p + k_base: 1, j_base*p + (k_base+1): 1},
            {(k_base)*n + (i_base+1): 1}
        )
        # M7 = (A12 - A22)(B21 + B22)
        add_term(
            {i_base*m + (j_base+1): 1, (i_base+1)*m + (j_base+1): -1},
            {(j_base+1)*p + k_base: 1, (j_base+1)*p + (k_base+1): 1},
            {(k_base+1)*n + i_base: 1}
        )
    
    # Now construct the full 2x5x6 decomposition
    # Split: A = [A_left (2x2), A_right (2x3)], B = [[B_topleft (2x2), B_topright (2x4)], [B_botleft (3x2), B_botright (3x4)]]
    # C_left (2x2) = A_left * B_topleft
    # C_right (2x4) = A_left * B_topright + A_right * B_botright
    # 
    # But we also need the cross terms for the middle dimension split...
    # Actually, for matrix multiplication, we need:
    # C = A * B where A is 2x5, B is 5x6
    # Split the 5 columns of A (and 5 rows of B) into first 2 and last 3:
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [[B1 (2x6)], [B2 (3x6)]]
    # C = A1*B1 + A2*B2
    # 
    # Now A1*B1 is 2x2 * 2x6. Split B1 into 2x2 + 2x2 + 2x2 (three 2x2 blocks)
    # Each 2x2*2x2 uses Strassen rank 7, so 3*7 = 21
    # A2*B2 is 2x3 * 3x6. Split B2 into 3x2 + 3x2 + 3x2
    # Each 2x3*3x2: rank 6 (schoolbook) since no known better, so 3*6 = 18
    # Total: 21 + 18 = 39... but that's not right because we need to handle the full 6 columns
    
    # Let me redo: B is 5x6. Split rows of B (and cols of A) into 2+3.
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [[B1 (2x6)], [B2 (3x6)]]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 2x2 * 2x6. Split the 6 columns into 2+2+2:
    #   A1*B1_col0 (2x2): Strassen rank 7
    #   A1*B1_col1 (2x2): Strassen rank 7
    #   A1*B1_col2 (2x2): Strassen rank 7
    #   Total: 21
    # 
    # A2*B2: 2x3 * 3x6. Split the 6 columns into 2+2+2:
    #   A2*B2_col0 (2x3 * 3x2): rank 6
    #   A2*B2_col1 (2x3 * 3x2): rank 6
    #   A2*B2_col2 (2x3 * 3x2): rank 6
    #   Total: 18
    # 
    # Grand total: 21 + 18 = 39
    
    # Apply Strassen for A1*B1 (2x2 * 2x6, split into three 2x2*2x2)
    strassen_2x2(0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_2x2(0, 0, 2)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    strassen_2x2(0, 0, 4)  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    
    # Schoolbook for A2*B2 (2x3 * 3x6, split into three 2x3*3x2)
    for k_base in range(0, 6, 2):
        for i in range(2):
            for j in range(2, 5):  # j = 2,3,4 (the last 3 columns of A)
                for k in range(k_base, k_base + 2):
                    add_term(
                        {i*m + j: 1},
                        {j*p + k: 1},
                        {k*n + i: 1}
                    )
    
    # Convert to arrays
    if U_list:
        U_strassen = np.column_stack(U_list)
        V_strassen = np.column_stack(V_list)
        W_strassen = np.column_stack(W_list)
        strassen_rank = U_strassen.shape[1]
    else:
        U_strassen = U_school
        V_strassen = V_school
        W_strassen = W_school
        strassen_rank = rank_school
    
    if strassen_rank < best_rank:
        best_U, best_V, best_W = U_strassen, V_strassen, W_strassen
        best_rank = strassen_rank
    
    # Verify the Strassen decomposition is correct
    if best_rank == strassen_rank and strassen_rank < rank_school:
        verify_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(best_U[i*m+j, :], best_V[j*p+k, :] * best_W[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        verify_ok = False
                        break
                if not verify_ok:
                    break
            if not verify_ok:
                break
        
        if not verify_ok:
            # Fallback to schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = rank_school
    
    # Try to find even lower rank with ALS if time permits
    if time.time() < deadline - 30 and best_rank > 30:
        np.random.seed(42)
        for target_rank in range(best_rank - 1, 29, -1):
            if time.time() > deadline - 15:
                break
            
            for attempt in range(2):
                if time.time() > deadline - 10:
                    break
                
                U_curr = np.random.randn(nm, target_rank)
                V_curr = np.random.randn(mp, target_rank)
                W_curr = np.random.randn(pn, target_rank)
                
                U_curr /= np.linalg.norm(U_curr) * np.sqrt(target_rank)
                V_curr /= np.linalg.norm(V_curr) * np.sqrt(target_rank)
                W_curr /= np.linalg.norm(W_curr) * np.sqrt(target_rank)
                
                for it in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    # Update U
                    U_new = np.zeros_like(U_curr)
                    for i in range(n):
                        for j in range(m):
                            A_ii = np.zeros((p, target_rank))
                            for k in range(p):
                                A_ii[k, :] = V_curr[j*p+k, :] * W_curr[k*n+i, :]
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_ii.T, np.ones(p), rcond=None)
                                U_new[i*m+j, :] = sol
                            except:
                                U_new[i*m+j, :] = U_curr[i*m+j, :]
                    U_curr = U_new
                    
                    # Update V
                    V_new = np.zeros_like(V_curr)
                    for j in range(m):
                        for k in range(p):
                            B_jk = np.zeros((n, target_rank))
                            for i in range(n):
                                B_jk[i, :] = U_curr[i*m+j, :] * W_curr[k*n+i, :]
                            try:
                                sol, _, _, _ = np.linalg.lstsq(B_jk.T, np.ones(n), rcond=None)
                                V_new[j*p+k, :] = sol
                            except:
                                V_new[j*p+k, :] = V_curr[j*p+k, :]
                    V_curr = V_new
                    
                    # Update W
                    W_new = np.zeros_like(W_curr)
                    for k in range(p):
                        for i in range(n):
                            C_ki = np.zeros((m, target_rank))
                            for j in range(m):
                                C_ki[j, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :]
                            try:
                                sol, _, _, _ = np.linalg.lstsq(C_ki.T, np.ones(m), rcond=None)
                                W_new[k*n+i, :] = sol
                            except:
                                W_new[k*n+i, :] = W_curr[k*n+i, :]
                    W_curr = W_new
                    
                    # Check error
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                                err += (val - 1.0)**2
                    err = np.sqrt(err)
                    
                    if err < 1e-10:
                        # Try rationalization
                        U_rat = np.zeros((nm, target_rank))
                        V_rat = np.zeros((mp, target_rank))
                        W_rat = np.zeros((pn, target_rank))
                        
                        for idx in range(nm):
                            for r in range(target_rank):
                                U_rat[idx, r] = Fraction(U_curr[idx, r]).limit_denominator(10**6)
                        for idx in range(mp):
                            for r in range(target_rank):
                                V_rat[idx, r] = Fraction(V_curr[idx, r]).limit_denominator(10**6)
                        for idx in range(pn):
                            for r in range(target_rank):
                                W_rat[idx, r] = Fraction(W_curr[idx, r]).limit_denominator(10**6)
                        
                        exact_ok = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = Fraction(0)
                                    for r in range(target_rank):
                                        val += U_rat[i*m+j, r] * V_rat[j*p+k, r] * W_rat[k*n+i, r]
                                    if val != Fraction(1):
                                        exact_ok = False
                                        break
                                if not exact_ok:
                                    break
                            if not exact_ok:
                                break
                        
                        if exact_ok:
                            best_U, best_V, best_W = U_rat, V_rat, W_rat
                            best_rank = target_rank
                            break
                    if err < 1e-10 or time.time() > deadline - 10:
                        break
            
            if best_rank < target_rank + 1:
                break
    
    # Convert to lists
    U_out = np.zeros((nm, best_rank))
    V_out = np.zeros((mp, best_rank))
    W_out = np.zeros((pn, best_rank))
    
    for i in range(nm):
        for r in range(best_rank):
            U_out[i, r] = float(best_U[i, r])
    for i in range(mp):
        for r in range(best_rank):
            V_out[i, r] = float(best_V[i, r])
    for i in range(pn):
        for r in range(best_rank):
            W_out[i, r] = float(best_W[i, r])
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
