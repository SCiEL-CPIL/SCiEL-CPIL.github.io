    def strassen_core(terms, ioff, joff, koff, n, m, p):
        # ... indices ...
        # Define the 7 Strassen products' u and v vectors
        # u_k, v_k are dicts
        # w_k will be computed
        
        # M1 = (A11+A22)(B11+B22)
        u1 = {(0,0):1, (1,1):1}
        v1 = {(0,0):1, (1,1):1}
        
        # M2 = (A21+A22)B11
        u2 = {(1,0):1, (1,1):1}
        v2 = {(0,0):1}
        
        # M3 = A11(B12-B22)
        u3 = {(0,0):1}
        v3 = {(0,1):1, (1,1):-1}
        
        # M4 = A22(B21-B11)
        u4 = {(1,1):1}
        v4 = {(1,0):1, (0,0):-1}
        
        # M5 = (A11+A12)B22
        u5 = {(0,0):1, (0,1):1}
        v5 = {(1,1):1}
        
        # M6 = (A21-A11)(B11+B12)
        u6 = {(1,0):1, (0,0):-1}
        v6 = {(0,0):1, (0,1):1}
        
        # M7 = (A12-A22)(B21+B22)
        u7 = {(0,1):1, (1,1):-1}
        v7 = {(1,0):1, (1,1):1}
        
        u_vecs = [u1, u2, u3, u4, u5, u6, u7]
        v_vecs = [v1, v2, v3, v4, v5, v6, v7]
        
        # C combinations
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Build w vectors for each term k (0..6)
        # w_k is a dict over (k_idx, i_idx) in the block
        # Actually w is over (k, i) where k is col, i is row of C
        # In our index space: idxc(k, i)
        
        w_vecs = [{} for _ in range(7)]
        
        # Helper to add to w_vec
        def add_w(k, ci, ck, val):
            if k not in w_vecs: w_vecs[k] = {}
            key = idxc(ck, ci)
            w_vecs[k][key] = w_vecs[k].get(key, 0) + val
            
        # C(0,0) = M1 + M4 - M5 + M7
        add_w(0, 0, 0, 1) # M1
        add_w(3, 0, 0, 1) # M4
        add_w(4, 0, 0, -1) # -M5
        add_w(6, 0, 0, 1) # M7
        
        # C(0,1) = M3 + M5
        add_w(2, 0, 1, 1) # M3
        add_w(4, 0, 1, 1) # M5
        
        # C(1,0) = M2 + M4
        add_w(1, 1, 0, 1) # M2
        add_w(3, 1, 0, 1) # M4
        
        # C(1,1) = M1 - M2 + M3 + M6
        add_w(0, 1, 1, 1) # M1
        add_w(1, 1, 1, -1) # -M2
        add_w(2, 1, 1, 1) # M3
        add_w(5, 1, 1, 1) # M6
        
        # Add terms
        for k in range(7):
            u = {idx(i, j): val for (i, j), val in u_vecs[k].items()}
            v = {idxb(j, k): val for (j, k), val in v_vecs[k].items()}
            w = w_vecs[k]
            if w: # Only add if non-zero
                add_terms(terms, u, v, w)
