import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

def get_schoolbook(n, m, p):
    """Returns U, V, W for schoolbook decomposition of M_{n,m,p}."""
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

def tensor_product(U, V, W):
    """Computes the tensor T = sum_r U[:,r] outer V[:,r] outer W[:,r]."""
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # T: (n, m, p)
    n, m, p = U.shape[0] // m, m, V.shape[0] // m # This is tricky, need dimensions
    # Better to pass n, m, p
    pass

def compute_tensor(U, V, W, n, m, p):
    """
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Returns T: (n, m, p)
    """
    R = U.shape[1]
    T = np.zeros((n, m, p))
    # Reshape U, V, W to (n, m, R), (m, p, R), (p, n, R)
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W index is k*n + i, so W_reshaped[k, i, r]
    for r in range(R):
        T += U_reshaped[:, :, r][:, :, np.newaxis] * V_reshaped[:, :, r][np.newaxis, :, :] * W_reshaped[:, :, r][np.newaxis, np.newaxis, :]
        # Wait, dimensions:
        # U_reshaped[:, :, r] is (n, m)
        # V_reshaped[:, :, r] is (m, p)
        # W_reshaped[:, :, r] is (p, n)
        # We want T[i, j, k] = U[i, j] * V[j, k] * W[k, i]
        # This is not a standard outer product.
        # Let's use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T

def solve_block(n, m, p, time_limit=20):
    """
    Attempts to find a low-rank decomposition for M_{n,m,p}.
    Returns (U, V, W) or None if failed.
    """
    # Target tensor T
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0
                
    schoolbook_rank = n * m * p
    U_school, V_school, W_school = get_schoolbook(n, m, p)
    
    # Try ranks from lower bound to schoolbook
    # Lower bound: max(n*m, m*p, p*n) is not a lower bound for rank.
    # Rank >= max(n, m, p) * min(n, m, p) / something?
    # Let's just try a few ranks below schoolbook.
    # For 3x3x3, try 23, 24, 25, 26, 27.
    # For 3x3x2, try 17, 18.
    
    best_result = None
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Define a list of ranks to try
    ranks_to_try = list(range(max(1, schoolbook_rank - 5), schoolbook_rank + 1))
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random restarts for this rank
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W
            U_init = np.random.rand(n * m, R) * 0.1
            V_init = np.random.rand(m * p, R) * 0.1
            W_init = np.random.rand(p * n, R) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n, m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
                W = x[n*m*R + m*p*R :].reshape(p, n, R)
                
                T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                diff = T - T_target
                return np.sum(diff**2)
                
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
                if res.success and res.fun < 1e-6:
                    # Check if we can snap to half-integers
                    U_opt = res.x[:n*m*R].reshape(n, m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m, p, R)
                    W_opt = res.x[n*m*R + m*p*R :].reshape(p, n, R)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, T_target, atol=1e-9):
                        # Valid decomposition found
                        U_final = U_snap.flatten().reshape(n*m, R)
                        V_final = V_snap.flatten().reshape(m*p, R)
                        W_final = W_snap.flatten().reshape(p*n, R)
                        return U_final, V_final, W_final
            except Exception:
                pass
                
    # If no better decomposition found, return schoolbook
    return U_school, V_school, W_school

def construct():
    n, m, p = DIMENSIONS
    # Split A (n x m) into A1 (n x m1) and A2 (n x m2)
    # Split B (m x p) into B1 (m1 x p) and B2 (m2 x p)
    # We chose m1=3, m2=2 for m=5
    m1 = 3
    m2 = m - m1
    
    # Sub-blocks:
    # 1. A1 B1: n x m1 x p -> 3 x 3 x 5
    # 2. A2 B2: n x m2 x p -> 3 x 2 x 5
    # Wait, B is m x p. If we split A into columns, we must split B into rows.
    # A = [A1, A2] (n x m1, n x m2)
    # B = [B1; B2] (m1 x p, m2 x p)
    # C = A1 B1 + A2 B2
    
    # So we need decompositions for:
    # M_{3,3,5} and M_{3,2,5}
    
    # We can further split M_{3,3,5} into M_{3,3,3} + M_{3,3,2}
    # And M_{3,2,5} into M_{3,2,3} + M_{3,2,2}
    
    # Let's solve for these 4 blocks
    blocks = [
        (3, 3, 3),
        (3, 3, 2),
        (3, 2, 3),
        (3, 2, 2)
    ]
    
    decomps = []
    for bn, bm, bp in blocks:
        U, V, W = solve_block(bn, bm, bp, time_limit=25)
        decomps.append((U, V, W, bn, bm, bp))
        
    # Now combine them
    # Total Rank = sum of ranks of blocks
    total_rank = sum(U.shape[1] for U, V, W, _, _, _ in decomps)
    
    U_total = np.zeros((n * m, total_rank))
    V_total = np.zeros((m * p, total_rank))
    W_total = np.zeros((p * n, total_rank))
    
    col_offset = 0
    # Block 1: A1 B1 (3x3x3) -> A1 is cols 0-2 of A, B1 is rows 0-2 of B
    # U rows: 0..2 (i) x 0..2 (j) -> indices i*5+j? No, U is for A (3x5).
    # A1 is 3x3. So U rows are i*3+j? No, U shape is (n*m, R).
    # In the sub-block problem, U is (n*bm, R).
    # We need to map sub-block U to total U.
    # Sub-block A1 is A[:, 0:3]. So rows of U_sub correspond to i*3+j.
    # In total U, rows correspond to i*5+j.
    # So for row r_sub = i*3+j in U_sub, row r_tot = i*5+j in U_total.
    
    for idx, (U_sub, V_sub, W_sub, bn, bm, bp) in enumerate(decomps):
        R_sub = U_sub.shape[1]
        
        # Map U
        # U_sub is (bn*bm, R_sub)
        # U_total is (n*m, total_rank)
        # bn = n, bm is the width of A sub-block
        # A sub-blocks:
        # Block 0: A1 (3x3) -> cols 0-2
        # Block 1: A1 (3x3) part of M_{3,3,2}? No.
        
        # Let's re-verify the block structure.
        # C = A1 B1 + A2 B2
        # A1 is 3x3, B1 is 3x5. Product is 3x5.
        # A2 is 3x2, B2 is 2x5. Product is 3x5.
        
        # So we need:
        # M_{3,3,5} for A1 B1
        # M_{3,2,5} for A2 B2
        
        # My previous split was:
        # M_{3,3,5} = M_{3,3,3} + M_{3,3,2}
        # M_{3,2,5} = M_{3,2,3} + M_{3,2,2}
        
        # So the 4 blocks are:
        # 0: M_{3,3,3} (part of A1 B1)
        # 1: M_{3,3,2} (part of A1 B1)
        # 2: M_{3,2,3} (part of A2 B2)
        # 3: M_{3,2,2} (part of A2 B2)
        
        # For Block 0 (3x3x3):
        # A part: 3x3 (cols 0-2 of A1)
        # B part: 3x3 (rows 0-2 of B1, cols 0-2 of B1? No, B1 is 3x5. Split B1 into 3x3 and 3x2)
        # So Block 0 uses A[:, 0:3] and B[0:3, 0:3].
        # Block 1 uses A[:, 0:3] and B[0:3, 3:5].
        # Block 2 uses A[:, 3:5] and B[3:5, 0:3].
        # Block 3 uses A[:, 3:5] and B[3:5, 3:5].
        
        # Mapping U:
        # U_sub is (bn*bm, R_sub).
        # For Block 0: bn=3, bm=3.
        # U_sub row i*3+j corresponds to A[i, j] where j is 0..2.
        # In Total U (3x5), row i*5+j.
        # So for j in 0..2, Total Row = i*5 + j.
        
        # Mapping V:
        # V_sub is (bm*bp, R_sub).
        # For Block 0: bm=3, bp=3.
        # V_sub row j*3+k corresponds to B[j, k] where j is 0..2 (row of B), k is 0..2 (col of B).
        # In Total V (5x5), row j*5+k.
        # So for j in 0..2, k in 0..2, Total Row = j*5 + k.
        
        # Mapping W:
        # W_sub is (bp*bn, R_sub).
        # For Block 0: bp=3, bn=3.
        # W_sub row k*3+i corresponds to C[i, k] where i is 0..2, k is 0..2.
        # In Total W (5x3? No, p*n = 5*3=15), row k*3+i.
        # Wait, Total W is (p*n, R) = (5*3, R).
        # Index is k*n + i = k*3 + i.
        # So W_sub row k*3+i maps directly to Total W row k*3+i.
        
        # We need to apply this mapping for each block.
        
        # Define mappings
        # Block 0: A cols 0-2, B rows 0-2, B cols 0-2
        # Block 1: A cols 0-2, B rows 0-2, B cols 3-4
        # Block 2: A cols 3-4, B rows 3-4, B cols 0-2
        # Block 3: A cols 3-4, B rows 3-4, B cols 3-4
        
        mappings = [
            {'a_start': 0, 'a_end': 3, 'b_row_start': 0, 'b_row_end': 3, 'b_col_start': 0, 'b_col_end': 3},
            {'a_start': 0, 'a_end': 3, 'b_row_start': 0, 'b_row_end': 3, 'b_col_start': 3, 'b_col_end': 5},
            {'a_start': 3, 'a_end': 5, 'b_row_start': 3, 'b_row_end': 5, 'b_col_start': 0, 'b_col_end': 3},
            {'a_start': 3, 'a_end': 5, 'b_row_start': 3, 'b_row_end': 5, 'b_col_start': 3, 'b_col_end': 5}
        ]
        
        if idx < 4:
            mp = mappings[idx]
            
            # U Mapping
            # U_sub is (bn*bm, R_sub). bn=3.
            # Row r_sub = i*bm + j_local.
            # j_global = mp['a_start'] + j_local.
            # Row r_tot = i*m + j_global.
            for r_sub in range(U_sub.shape[0]):
                i = r_sub // bm
                j_local = r_sub % bm
                j_global = mp['a_start'] + j_local
                r_tot = i * m + j_global
                U_total[r_tot, col_offset:col_offset+R_sub] = U_sub[r_sub, :]
                
            # V Mapping
            # V_sub is (bm*bp, R_sub).
            # Row r_sub = j_local*bp + k_local.
            # j_global = mp['b_row_start'] + j_local.
            # k_global = mp['b_col_start'] + k_local.
            # Row r_tot = j_global*p + k_global.
            for r_sub in range(V_sub.shape[0]):
                j_local = r_sub // bp
                k_local = r_sub % bp
                j_global = mp['b_row_start'] + j_local
                k_global = mp['b_col_start'] + k_local
                r_tot = j_global * p + k_global
                V_total[r_tot, col_offset:col_offset+R_sub] = V_sub[r_sub, :]
                
            # W Mapping
            # W_sub is (bp*bn, R_sub).
            # Row r_sub = k_local*bn + i.
            # k_global = mp['b_col_start'] + k_local.
            # i is same (0..2).
            # Row r_tot = k_global*n + i.
            for r_sub in range(W_sub.shape[0]):
                k_local = r_sub // bn
                i = r_sub % bn
                k_global = mp['b_col_start'] + k_local
                r_tot = k_global * n + i
                W_total[r_tot, col_offset:col_offset+R_sub] = W_sub[r_sub, :]
                
        col_offset += R_sub
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
