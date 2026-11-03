def construct():
    import time
    import numpy as np
    
    _t0 = time.time()
    _DEADLINE = 135.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6

    # Start with the structured decomposition: schoolbook for row 0, Strassen for rows 1-2
    # This gives rank 66
    
    def build_structured():
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1)
            b = (i1, j2)
            c = (i2, j1)
            d = (i2, j2)
            e = (j1, k1)
            f = (j1, k2)
            g = (j2, k1)
            h = (j2, k2)
            
            C11 = (i1, k1)
            C12 = (i1, k2)
            C21 = (i2, k1)
            C22 = (i2, k2)
            
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(i, k): 1}
            ))
        
        # Row 0: schoolbook
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        # Rows 1-2: Strassen
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        return U, V, W
    
    U, V, W = build_structured()
    rank = len(U[0])
    
    # Convert to numpy for fast operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the structured decomposition
    def verify(U_arr, V_arr, W_arr, n, m, p, tol=1e-9):
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U_np, V_np, W_np, n, m, p):
        # Fallback to pure schoolbook
        rank = n * m * p
        U_np = np.zeros((n*m, rank))
        V_np = np.zeros((m*p, rank))
        W_np = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_np[i*m+j, col] = 1
                    V_np[j*p+k, col] = 1
                    W_np[k*n+i, col] = 1
                    col += 1
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    # Now attempt rank reduction via algebraic peeling / dependency search
    # Strategy: Try to find linear combinations of columns that can be eliminated
    # by absorbing into other columns (i.e., find a column that is in the span of others
    # in a way that preserves the tensor identity)
    
    # A more practical approach: try to find a rank-1 update that allows removing a term
    # Specifically, look for pairs of terms where one can be expressed as a linear combination
    # of the other plus a rank-1 correction that cancels elsewhere
    
    # Alternative: Use the fact that the tensor has structure. Try to find a valid
    # decomposition with rank < 66 by searching for a different structured decomposition.
    
    # Let's try a hybrid: use Strassen on a 2x4x4 block (rows 1-2, all 4 columns, 4 output cols)
    # This would be: split m=4 into 2+2, use Strassen on each 2x2x2, giving 7+7=14 terms
    # for rows 1-2, cols 0-3, outputs 0-3. Plus 2x2x2 for outputs 4-5: 7 terms.
    # Total for rows 1-2: 14 + 7 = 21 terms. Row 0: 24 terms. Total: 45 terms.
    
    def build_hybrid():
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1)
            b = (i1, j2)
            c = (i2, j1)
            d = (i2, j2)
            e = (j1, k1)
            f = (j1, k2)
            g = (j2, k1)
            h = (j2, k2)
            
            C11 = (i1, k1)
            C12 = (i1, k2)
            C21 = (i2, k1)
            C22 = (i2, k2)
            
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(i, k): 1}
            ))
        
        # Row 0: schoolbook
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        # Rows 1-2: 
        # For columns 0-1, outputs 0-5: split into three 2x2x2 blocks
        # For columns 2-3, outputs 0-5: split into three 2x2x2 blocks
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        
        return U, V, W
    
    # The hybrid is the same as structured here. Let me think differently.
    
    # Try: Can we do better by using a 3x4x6 decomposition that isn't block-structured?
    # The theoretical lower bound is max(n*m, m*p, n*p) = max(12, 24, 18) = 24.
    # Upper bound is n*m*p = 72.
    
    # Let's try to use the inspiration's approach of splitting p=6 into 2+2+2
    # and using the 3x4x2 decomposition for each, then combining.
    
    def build_split_p():
        # For 3x4x2: split n=3 into 1+2
        # 1x4x2: schoolbook = 8 terms
        # 2x4x2: split m=4 into 2+2, each 2x2x2 uses Strassen = 7 terms each, total 14
        # So 3x4x2 has rank 8 + 14 = 22
        # 3x4x6 = 3 * 22 = 66 (same as before)
        # This doesn't help.
        
        # What if we use a different split for 2x4x2?
        # 2x4x2: could we do better than 14?
        # Lower bound: max(2*4, 4*2, 2*2) = max(8, 8, 4) = 8
        # Strassen gives 14. Is there a 10 or 12 term decomposition?
        
        # For now, stick with the structured approach and try local optimization.
        pass
    
    # Let's try a direct search: start with the rank-66 decomposition and try to
    # find a linear dependency among the columns that allows rank reduction.
    
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k).
    # A decomposition is valid if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
    
    # To reduce rank, we need to find a column r such that U[:,r], V[:,r], W[:,r]
    # can be expressed as linear combinations of other columns in a way that preserves
    # the tensor identity.
    
    # This is complex. Instead, let's try a randomized local search:
    # 1. Start with the rank-66 decomposition
    # 2. Randomly perturb some entries
    # 3. Try to solve for a lower-rank decomposition that fits the tensor
    
    # Actually, let's try a simpler approach: use the inspiration's _decompose_specific
    # which gives rank 66 for 3x4x6, and see if we can do better.
    
    # Let me try: 3x4x6 = 3x(4x6). 
    # 4x6 = 4x(2+2+2). Each 4x2 block:
    # 4x2: schoolbook = 8 terms. Or split 4 into 2+2: each 2x2 is rank 1 (just A[i,j]*B[j,k]),
    # so 2x2x2... wait, 4x2 means 4 rows, 2 cols of A times 2 rows, 2 cols of B? No.
    
    # I'm getting confused. Let me just return the rank-66 solution and see if there's
    # a quick win.
    
    # Actually, let me try one more thing: use the fact that for the 2x2x2 Strassen,
    # there might be a 6-term decomposition over certain fields, but over reals it's 7.
    
    # Let's just return the structured rank-66 solution.
    
    U_list = U_np.tolist()
    V_list = V_np.tolist()
    W_list = W_np.tolist()
    
    if time.time() - _t0 > _DEADLINE:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
