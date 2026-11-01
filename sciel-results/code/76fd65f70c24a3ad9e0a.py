import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb

    def verify(U, V, W, R, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(mat, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    def try_discrete_structured(R):
        """Try structured discrete decompositions with integer/half-integer entries.
        
        Uses hybrid Strassen/Schoolbook:
        - Split A (2x3) = [A1 (2x2) | A2 (2x1)]
        - Split B (3x7) = [B1 (2x7); B2 (1x7)]
        - C = A1*B1 + A2*B2
        
        For A1*B1 (2x2 * 2x7): use Strassen on 2x2x2 blocks
        For A2*B2 (2x1 * 1x7): schoolbook (14 terms)
        
        Total: 7*3 + 7*1 + 14 = 21 + 7 + 14 = 42... wait that's still 42.
        
        Let me reconsider. For 2x2 * 2x7:
        B1 has columns 0-6. Group them as:
        - Columns 0,1: 2x2 block -> Strassen (7 mults)
        - Columns 2,3: 2x2 block -> Strassen (7 mults)  
        - Columns 4,5: 2x2 block -> Strassen (7 mults)
        - Column 6: 2x1 block -> schoolbook (2 mults)
        Total for A1*B1: 7+7+7+2 = 23
        
        For A2*B2 (2x1 * 1x7): 2*1*7 = 14 schoolbook terms
        
        Grand total: 23 + 14 = 37
        """
        # Build the hybrid decomposition
        U_list = []
        V_list = []
        W_list = []
        
        # Part 1: A1 (2x2) * B1 (2x7) using Strassen on 2x2 blocks
        # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B1 columns 0-5 in pairs, column 6 separate
        
        def add_strassen_block(col_start, col_end):
            """Add Strassen decomposition for A1 * B1[:, col_start:col_end+1]"""
            # B1_block is 2x2 with columns col_start, col_start+1
            # Strassen: M1 = (A00+A11)*(B00+B11)
            #           M2 = (A10+A11)*B00
            #           M3 = A00*(B01-B11)
            #           M4 = A11*(B00+B10)
            #           M5 = (A00+A10)*B11
            #           M6 = (A11-A10)*(B00+B01)
            #           M7 = (A01-A11)*(B10+B11)
            # C00 = M1 + M3 - M4 - M7
            # C01 = M5 + M7
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M5
            
            c0, c1 = col_start, col_start + 1
            
            # M1 = (A00+A11)*(B00+B11)
            U_list.append([1, 0, 1, 0, 0, 0, 0])  # A00+A11
            V_list.append([1, 0, 1, 0, 0, 0, 0])  # B00+B11 (rows 0,1 of B1)
            W_list.append([0, 0, 1, 0, 1, 0, 0])  # contributes to C[0,c0] and C[1,c1]
            
            # M2 = (A10+A11)*B00
            U_list.append([0, 0, 1, 1, 0, 0, 0])  # A10+A11
            V_list.append([1, 0, 0, 0, 0, 0, 0])  # B00
            W_list.append([0, 0, 0, 0, 1, 0, 0])  # contributes to C[1,c0]
            
            # M3 = A00*(B01-B11)
            U_list.append([1, 0, 0, 0, 0, 0, 0])  # A00
            V_list.append([0, 1, -1, 0, 0, 0, 0])  # B01-B11
            W_list.append([1, 0, 0, 0, 0, 0, 0])  # contributes to C[0,c0]
            
            # M4 = A11*(B00+B10)
            U_list.append([0, 0, 0, 0, 1, 0, 0])  # A11
            V_list.append([1, 0, 1, 0, 0, 0, 0])  # B00+B10
            W_list.append([0, 0, -1, 0, 1, 0, 0])  # -C[0,c0] + C[1,c0]
            
            # M5 = (A00+A10)*B11
            U_list.append([1, 0, 1, 0, 0, 0, 0])  # A00+A10
            V_list.append([0, 0, 0, 1, 0, 0, 0])  # B11
            W_list.append([0, 1, 0, 0, 0, 0, 0])  # contributes to C[0,c1]
            
            # M6 = (A11-A10)*(B00+B01)
            U_list.append([0, 0, -1, 1, 0, 0, 0])  # A11-A10
            V_list.append([1, 1, 0, 0, 0, 0, 0])  # B00+B01
            W_list.append([0, 0, 0, 0, 0, 0, 0])  # not used directly in simple Strassen
            
            # M7 = (A01-A11)*(B10+B11)
            U_list.append([0, 1, 0, -1, 0, 0, 0])  # A01-A11
            V_list.append([0, 0, 1, 1, 0, 0, 0])  # B10+B11
            W_list.append([-1, 1, 0, 0, 0, 0, 0])  # -C[0,c0] + C[0,c1]
        
        # Apply Strassen to column pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 1)
        add_strassen_block(2, 3)
        add_strassen_block(4, 5)
        
        # Column 6: schoolbook for A1 * B1[:,6]
        # C[0,6] = A00*B06 + A01*B16
        # C[1,6] = A10*B06 + A11*B16
        U_list.append([1, 0, 0, 0, 0, 0, 0])  # A00
        V_list.append([1, 0, 0, 0, 0, 0, 0])  # B06
        W_list.append([0, 0, 0, 0, 0, 0, 1])  # C[0,6]
        
        U_list.append([0, 1, 0, 0, 0, 0, 0])  # A01
        V_list.append([0, 1, 0, 0, 0, 0, 0])  # B16
        W_list.append([0, 0, 0, 0, 0, 0, 1])  # C[0,6]
        
        U_list.append([0, 0, 1, 0, 0, 0, 0])  # A10
        V_list.append([1, 0, 0, 0, 0, 0, 0])  # B06
        W_list.append([0, 0, 0, 0, 0, 0, 1])  # C[1,6]
        
        U_list.append([0, 0, 0, 0, 1, 0, 0])  # A11
        V_list.append([0, 1, 0, 0, 0, 0, 0])  # B16
        W_list.append([0, 0, 0, 0, 0, 0, 1])  # C[1,6]
        
        # Part 2: A2 (2x1) * B2 (1x7) using schoolbook
        # A2 = [A[0,2], A[1,2]]
        # B2 = [B[2,0], B[2,1], ..., B[2,6]]
        # C[0,k] += A[0,2]*B[2,k]
        # C[1,k] += A[1,2]*B[2,k]
        
        for k in range(7):
            U_list.append([0, 0, 0, 1, 0, 0, 0])  # A[0,2]
            V_list.append([0, 0, 0, 0, 0, 0, 1])  # B[2,k] (row 2)
            W_list.append([0, 0, 0, 0, 0, 0, 1])  # C[0,k]
            
            U_list.append([0, 0, 0, 0, 0, 1, 0])  # A[1,2]
            V_list.append([0, 0, 0, 0, 0, 0, 1])  # B[2,k] (row 2)
            W_list.append([0, 0, 0, 0, 0, 0, 1])  # C[1,k]
        
        # Convert to matrices
        R_actual = len(U_list)
        U_mat = np.array(U_list).T  # shape (6, R_actual)
        V_mat = np.array(V_list).T  # shape (21, R_actual)
        W_mat = np.array(W_list).T  # shape (14, R_actual)
        
        # Verify this works
        if verify(U_mat, V_mat, W_mat, R_actual):
            return U_mat, V_mat, W_mat
        
        return None

    def continuous_optimize(R, attempt, deadline):
        """Optimize a rank-R decomposition using least_squares."""
        rng = np.random.default_rng(attempt * 7919 + R * 104729)
        
        # Initialize with structured random values
        vals = [-1, -0.5, 0, 0.5, 1]
        U = rng.choice(vals, size=(n * m, R)).astype(float)
        V = rng.choice(vals, size=(m * p, R)).astype(float)
        W = rng.choice(vals, size=(p * n, R)).astype(float)
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            u_size = n * m * R
            v_size = m * p * R
            U = x[:u_size].reshape(n * m, R)
            V = x[u_size:u_size+v_size].reshape(m * p, R)
            W = x[u_size+v_size:].reshape(p * n, R)
            return U, V, W
        
        def residuals(x):
            U, V, W = unflatten(x)
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        res[idx] = s - 1.0
                        idx += 1
            return res
        
        x0 = flatten(U, V, W)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=3000, ftol=1e-14, xtol=1e-14)
        except Exception:
            return None
        
        if sol.cost < 1e-12:
            U_sol, V_sol, W_sol = unflatten(sol.x)
            
            # Try to snap to half-integers
            U_snapped = snap_to_half_integer(U_sol, tol=1e-5)
            V_snapped = snap_to_half_integer(V_sol, tol=1e-5)
            W_snapped = snap_to_half_integer(W_sol, tol=1e-5)
            
            if verify(U_snapped, V_snapped, W_snapped, R):
                return U_snapped, V_snapped, W_snapped
            
            # If snapping doesn't work, check if raw solution is close to half-integers
            all_close = True
            for mat in [U_sol, V_sol, W_sol]:
                snapped = snap_to_half_integer(mat, tol=1e-4)
                if np.max(np.abs(mat - snapped)) > 1e-4:
                    all_close = False
                    break
            
            if all_close:
                return U_snapped, V_snapped, W_snapped
        
        return None

    # First, try the hybrid Strassen/Schoolbook decomposition
    result = try_discrete_structured(37)
    if result is not None:
        U_sol, V_sol, W_sol = result
        if verify(U_sol, V_sol, W_sol, 37):
            return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # Try ranks from a reasonable lower bound up to schoolbook
    min_rank_to_try = 21
    max_attempts_per_rank = 3
    
    for R in range(min_rank_to_try, min(R_sb + 1, 42)):
        if time.time() > deadline - 20:
            break
        
        # Try continuous optimization with fewer attempts to save time
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 20:
                break
            
            result = continuous_optimize(R, attempt, deadline)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if verify(U_sol, V_sol, W_sol, R):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
