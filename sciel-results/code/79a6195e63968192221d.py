import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Precompute target tensor entries
    ones_indices = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    def verify(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid decomposition.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Vectorized computation: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check ones
        for a, b, c in ones_indices:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            if abs(T[a, b, c]) > tol:
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    # --- Step 1: Construct Rank-23 Solution (Landsberg) ---
    R23 = 23
    U23 = np.zeros((nm, R23))
    V23 = np.zeros((mp, R23))
    W23 = np.zeros((pn, R23))
    
    # The 23-term Landsberg decomposition for 3x3x3 matmul.
    # Each term r has support on specific (i,j) for U, (j,k) for V, (k,i) for W.
    # Format: list of (i,j), (j,k), (k,i) entries with coefficient 1 or -1
    
    # Term definitions: (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, coeff)
    # v_entries: list of (j, k, coeff)  
    # w_entries: list of (k, i, coeff)
    
    terms = [
        # r=0: (A00+A11)(B00+B11) -> C00 + C11
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # r=1: (A10+A11)B00 -> C10 - C01... let me use standard Strassen
        # Standard Strassen for 2x2:
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00
        # M3=A00(B01-B11)
        # M4=A11(B10-B00)
        # M5=(A00+A01)B11
        # M6=(A10-A00)(B00+B01)
        # M7=(A01-A11)(B10+B11)
        # C00=M1+M4-M5+M7
        # C01=M3+M5
        # C10=M2+M4
        # C11=M1-M2+M3+M6
        
        # r=0: M1=(A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # r=1: M2=(A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,-1),(1,1,1)]),
        # r=2: M3=A00(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # r=3: M4=A11(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # r=4: M5=(A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        # r=5: M6=(A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # r=6: M7=(A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        
        # r=7-22: Border terms for the 3x3 case (16 terms)
        # These handle all entries involving index 2
        # C02 = A00*B02 + A01*B12 + A02*B22
        # C12 = A10*B02 + A11*B12 + A12*B22
        # C20 = A20*B00 + A21*B10 + A22*B20
        # C21 = A20*B01 + A21*B11 + A22*B21
        # C22 = A20*B02 + A21*B12 + A22*B22
        # C02, C12, C20, C21, C22 need to be covered
        
        # We need 16 more terms. Let's use a systematic approach:
        # For each (i,j,k) with at least one index = 2, we need to cover it.
        # Total entries with index 2: 27 - 8 = 19, but some are shared.
        
        # Actually, let me use the known explicit Landsberg rank-23.
        # The additional 16 terms are:
        
        # r=7: A02*B22 -> C22 (partial)
        ([(0,2,1)], [(2,2,1)], [(2,2,1)]),
        # r=8: A20*B00 -> C00... no, C20
        ([(2,0,1)], [(0,0,1)], [(0,2,1)]),
        # r=9: A22*B20 -> C02... 
        ([(2,2,1)], [(2,0,1)], [(0,2,1)]),
        # r=10: A22*B21 -> C12
        ([(2,2,1)], [(2,1,1)], [(1,2,1)]),
        # r=11: A22*B22 -> C22
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # r=12: A02*B02 -> C02
        ([(0,2,1)], [(0,2,1)], [(2,0,1)]),
        # r=13: A02*B12 -> C12
        ([(0,2,1)], [(1,2,1)], [(2,0,1)]),
        # r=14: A12*B02 -> C02
        ([(1,2,1)], [(0,2,1)], [(2,1,1)]),
        # r=15: A12*B12 -> C12
        ([(1,2,1)], [(1,2,1)], [(2,1,1)]),
        # r=16: A20*B10 -> C10... 
        ([(2,0,1)], [(1,0,1)], [(0,2,1)]),
        # r=17: A20*B20 -> C02... 
        ([(2,0,1)], [(2,0,1)], [(0,2,1)]),
        # r=18: A21*B01 -> C01...
        ([(2,1,1)], [(0,1,1)], [(1,2,1)]),
        # r=19: A21*B11 -> C11...
        ([(2,1,1)], [(1,1,1)], [(1,2,1)]),
        # r=20: A21*B21 -> C12...
        ([(2,1,1)], [(2,1,1)], [(1,2,1)]),
        # r=21: A22*B02 -> C22...
        ([(2,2,1)], [(0,2,1)], [(2,2,1)]),
        # r=22: A22*B12 -> C22...
        ([(2,2,1)], [(1,2,1)], [(2,2,1)]),
    ]
    
    # This ad-hoc construction is error-prone. Let me use a verified approach.
    # Instead, let me try to directly construct and verify.
    
    # Better approach: use the rank-26 as base and try to merge/eliminate terms
    # via optimization, but with better time management.
    
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # Strassen terms for 2x2 core (indices 0,1)
    # r=0: (A00+A11)(B00+B11)
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    # r=1: (A10+A11)B00
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    # r=2: A00(B01-B11)
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    # r=3: A11(B10-B00)
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    # r=4: (A00+A01)B11
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Border terms (indices involving 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    assert col == 26, f"Expected 26 columns, got {col}"
    
    # Verify Rank 26
    if not verify(U26, V26, W26):
        # Fallback to Schoolbook Rank 27
        R27 = 27
        U27 = np.zeros((nm, R27))
        V27 = np.zeros((mp, R27))
        W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26
    
    # --- Step 2: Try to find rank 23 (Landsberg) using targeted optimization ---
    
    def try_optimize_rank(R, time_budget, init_U=None, init_V=None, init_W=None):
        if time.time() > start_time + time_budget:
            return None
        
        # Precompute target tensor as flat array
        targets = np.zeros(nm * mp * pn)
        for a, b, c in ones_indices:
            targets[a * mp * pn + b * pn + c] = 1.0
        
        from scipy.optimize import least_squares
        
        best_cost = float('inf')
        best_x = None
        
        np.random.seed(42)
        
        # Try at most a few restarts
        max_restarts = 5
        for i in range(max_restarts):
            if time.time() > start_time + time_budget:
                break
            
            if i == 0 and init_U is not None:
                x0 = np.concatenate([
                    init_U.flatten(),
                    init_V.flatten(),
                    init_W.flatten()
                ])
            else:
                x0 = np.concatenate([
                    np.random.randn(nm, R) * 0.5,
                    np.random.randn(mp, R) * 0.5,
                    np.random.randn(pn, R) * 0.5
                ]).flatten()
            
            def residual(x):
                U_f = x[:nm*R].reshape(nm, R)
                V_f = x[nm*R:2*nm*R].reshape(mp, R)
                W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
                T = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
                return T.flatten() - targets

            try:
                sol = least_squares(residual, x0, method='trf', max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-14:
                        break
            except Exception:
                pass

        if best_x is not None and best_cost < 1e-12:
            U_sol = best_x[:nm*R].reshape(nm, R)
            V_sol = best_x[nm*R:2*nm*R].reshape(mp, R)
            W_sol = best_x[2*nm*R:3*nm*R].reshape(pn, R)
            snapped = snap_to_half_integers(U_sol, V_sol, W_sol)
            if snapped:
                return snapped
        return None

    # Try Rank 23 with warm starts from rank-26 (drop 3 columns)
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        time_budget_23 = min(100.0, remaining_time - 10.0)
        
        found = False
        # Try dropping pairs of border columns (most likely to be mergeable)
        # Border columns are 7-25 (indices 7 through 25)
        border_cols = list(range(7, 26))
        
        # Try dropping 3 border columns at a time
        import itertools
        combos_tried = 0
        for drop_set in itertools.combinations(border_cols, 3):
            if time.time() > start_time + time_budget_23:
                break
            if combos_tried >= 30:  # Limit number of attempts
                break
            combos_tried += 1
            
            keep_cols = [c for c in range(26) if c not in drop_set]
            U_init = U26[:, keep_cols]
            V_init = V26[:, keep_cols]
            W_init = W26[:, keep_cols]
            
            time_per_attempt = max(1.0, min(5.0, (start_time + time_budget_23 - time.time()) / max(1, 30 - combos_tried)))
            res = try_optimize_rank(23, time_budget=time_per_attempt, init_U=U_init, init_V=V_init, init_W=W_init)
            if res is not None:
                best_U, best_V, best_W = res
                best_rank = 23
                found = True
                break
        
        if not found and time.time() < start_time + time_budget_23 - 5.0:
            # Try rank 25 as intermediate step
            time_budget_25 = min(30.0, start_time + time_budget_23 - time.time() - 5.0)
            for drop_col in border_cols:
                if time.time() > start_time + time_budget_25:
                    break
                U_init = np.delete(U26, drop_col, axis=1)
                V_init = np.delete(V26, drop_col, axis=1)
                W_init = np.delete(W26, drop_col, axis=1)
                time_per_attempt = max(1.0, min(3.0, (start_time + time_budget_25 - time.time()) / max(1, 19)))
                res = try_optimize_rank(25, time_budget=time_per_attempt, init_U=U_init, init_V=V_init, init_W=W_init)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = 25
                    found = True
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
