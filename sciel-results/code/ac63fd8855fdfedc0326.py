import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)

    # Part 1: 4x4x4 using nested Strassen (49 terms)
    # Define the 7 Strassen products for 4x4x4 (outer level)
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},  # M1
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},          # M2
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},         # M3
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},         # M4
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},          # M5
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},   # M6
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},   # M7
    ]
    
    c_contributions = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: -1},  # M2
        {1: 1, 3: 1},   # M3
        {0: 1, 2: 1},   # M4
        {0: -1, 1: 1},  # M5
        {3: 1},         # M6
        {0: 1},         # M7
    ]
    
    def a_global(block, li, lk):
        if block == 0:
            return li, lk
        elif block == 1:
            return li, lk + 2
        elif block == 2:
            return li + 2, lk
        elif block == 3:
            return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0:
            return lk, lj
        elif block == 1:
            return lk, lj + 2
        elif block == 2:
            return lk + 2, lj
        elif block == 3:
            return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0:
            return li, lj
        elif block == 1:
            return li, lj + 2
        elif block == 2:
            return li + 2, lj
        elif block == 3:
            return li + 2, lj + 2
    
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook (16 terms)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: 4x1x4 (k=4, j in 0..3) - schoolbook (16 terms)
    for i in range(4):
        for j in range(4):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    # Part 4: 4x1x1 (k=4, j=4) - 4 terms
    for i in range(4):
        add_term(i, 4, 4, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1

    # Local search to reduce rank: try to find linear dependencies
    # A term can be removed if its tensor contribution is in the span of others.
    # We check: for term t, does there exist a linear combination of other terms
    # that reproduces its contribution to the multiplication tensor?
    
    # The multiplication tensor T has T[i,k,j] = 1 if i,j,k in range, else 0.
    # Term t contributes: u_t[i,k] * v_t[k,j] * w_t[j,i] for all (i,k,j).
    
    # We can set up: for each term t, solve for coefficients c_s (s != t) such that
    # sum_s c_s * u_s[i,k]*v_s[k,j]*w_s[j,i] = u_t[i,k]*v_t[k,j]*w_t[j,i]
    # for all (i,k,j). This is a linear system in c_s if we fix t.
    # But the left side is bilinear in (c_s, u_s*v_s*w_s), so it's not linear in c_s
    # unless we treat each (i,k,j) as a separate equation.
    
    # Actually, for fixed t, the constraint is:
    # sum_{s!=t} c_s * (u_s[i,k]*v_s[k,j]*w_s[j,i]) = u_t[i,k]*v_t[k,j]*w_t[j,i]
    # for all (i,k,j). This IS linear in c_s! Each (i,k,j) gives one equation.
    
    # Number of unknowns: rank - 1 (coefficients for all terms except t)
    # Number of equations: n*m*p = 4*5*5 = 100
    
    # If rank - 1 >= 100, we might have a solution. But rank=85, so 84 < 100.
    # The system is overdetermined. We need the system to be consistent.
    
    # Let's try: for each term t, check if the system is consistent.
    # If yes, remove t and continue.
    
    elapsed = time.time() - _START
    if elapsed < _DEADLINE - 30:  # safety margin
        n_terms = len(U_cols)
        if n_terms > 0:
            U_arr = np.array(U_cols).T  # shape (n*m, n_terms)
            V_arr = np.array(V_cols).T  # shape (m*p, n_terms)
            W_arr = np.array(W_cols).T  # shape (p*n, n_terms)
            
            # Build the tensor: T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[j,i,t]
            # We want to check if term t can be expressed as combo of others.
            
            # For efficiency, only try to remove terms from the boundary (last 36)
            # since they're more likely to be redundant with core terms.
            
            # Actually, let's try a simpler approach: check if any term has
            # U, V, or W vector that is zero or nearly zero.
            
            # More practical: use the fact that in our construction, some terms
            # might have identical U,V,W patterns. Check for duplicates.
            
            seen = set()
            to_remove = []
            for t in range(n_terms):
                # Create a hash of the term
                u_key = tuple(U_arr[:, t].round(6))
                v_key = tuple(V_arr[:, t].round(6))
                w_key = tuple(W_arr[:, t].round(6))
                key = (u_key, v_key, w_key)
                if key in seen:
                    to_remove.append(t)
                else:
                    seen.add(key)
            
            if to_remove:
                # Remove duplicate terms
                keep = [t for t in range(n_terms) if t not in set(to_remove)]
                U_cols = [U_cols[t] for t in keep]
                V_cols = [V_cols[t] for t in keep]
                W_cols = [W_cols[t] for t in keep]
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
