import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    n, m, p = 3, 4, 6
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    def build_strassen_block(i1, i2, j1, j2, k1, k2):
        """
        Generate 7 Strassen terms for a 2x2x2 block.
        A rows i1,i2; A cols j1,j2; B rows j1,j2; B cols k1,k2; C rows i1,i2; C cols k1,k2
        Returns list of (u_dict, v_dict, w_dict) where each dict maps (row_idx) -> coeff
        U index: i*m+j, V index: j*p+k, W index: k*n+i
        """
        terms = []
        # Standard Strassen for 2x2x2:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1 - P4 + P7
        # C12 = P3 + P5
        # C21 = P2 + P6
        # C22 = P1 - P3 + P4
        
        # U indices
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        # V indices
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        # W indices (k*n+i)
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # P1 = (A11+A22)(B11+B22)
        terms.append(({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        # P2 = (A21+A22)B11
        terms.append(({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        # P3 = A11(B12-B22)
        terms.append(({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        # P4 = A22(B21-B11)
        terms.append(({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        # P5 = (A11+A12)B22
        terms.append(({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        # P6 = (A21-A11)(B11+B12)
        terms.append(({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        # P7 = (A12-A22)(B21+B22)
        terms.append(({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        
        return terms
    
    def build_schoolbook_row(i, j_start, j_end, k_start, k_end):
        """Schoolbook for a single row i, columns j_start..j_end-1, all k in k_start..k_end-1"""
        terms = []
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms
    
    def build_full_schoolbook():
        """Pure schoolbook: n*m*p terms"""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        return terms
    
    def terms_to_matrices(terms):
        """Convert list of term dicts to U, V, W matrices"""
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, coeff in u_dict.items():
                U[idx, r] = coeff
            for idx, coeff in v_dict.items():
                V[idx, r] = coeff
            for idx, coeff in w_dict.items():
                W[idx, r] = coeff
        
        return U, V, W
    
    def verify(U, V, W):
        """
        Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        Use vectorized numpy for speed.
        """
        rank = U.shape[1]
        # For each (i,j,k), compute sum over r
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # For each i, j, k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is: einsum('ijr,jkr,kir->ijk', U, V, W)
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check all entries equal 1
        return np.allclose(result, 1.0, atol=1e-9)
    
    # Strategy 0: Original - row 0 schoolbook, rows 1-2 Strassen
    def strategy0():
        terms = []
        # Row 0: schoolbook
        terms.extend(build_schoolbook_row(0, 0, m, 0, p))
        # Rows 1-2: Strassen 2x2x2 blocks
        for j1 in range(0, m, 2):
            for k1 in range(0, p, 2):
                terms.extend(build_strassen_block(1, 2, j1, j1+1, k1, k1+1))
        return terms
    
    # Strategy 1: Row 2 schoolbook, rows 0-1 Strassen
    def strategy1():
        terms = []
        # Row 2: schoolbook
        terms.extend(build_schoolbook_row(2, 0, m, 0, p))
        # Rows 0-1: Strassen 2x2x2 blocks
        for j1 in range(0, m, 2):
            for k1 in range(0, p, 2):
                terms.extend(build_strassen_block(0, 1, j1, j1+1, k1, k1+1))
        return terms
    
    # Strategy 2: Pure schoolbook
    def strategy2():
        return build_full_schoolbook()
    
    # Strategy 3: Try different partition - rows 0-1 Strassen for j=0,1 and j=2,3 separately
    # This is same as strategy1 but let's verify
    def strategy3():
        terms = []
        terms.extend(build_schoolbook_row(2, 0, m, 0, p))
        for j1 in [0, 2]:
            for k1 in [0, 2, 4]:
                terms.extend(build_strassen_block(0, 1, j1, j1+1, k1, k1+1))
        return terms
    
    # Try all strategies
    strategies = [strategy0, strategy1, strategy2, strategy3]
    best_result = None
    best_rank = float('inf')
    
    for idx, strat_fn in enumerate(strategies):
        if not time_ok():
            break
        
        terms = strat_fn()
        rank = len(terms)
        
        if rank >= best_rank:
            continue
        
        U, V, W = terms_to_matrices(terms)
        
        if verify(U, V, W):
            best_result = (U, V, W, rank)
            best_rank = rank
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to list format as required
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: pure schoolbook
    terms = build_full_schoolbook()
    U, V, W = terms_to_matrices(terms)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
