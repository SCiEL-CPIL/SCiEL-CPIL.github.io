import time
import numpy as np
from fractions import Fraction

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    nm, mp, pn = n * m, m * p, p * n
    for u in range(nm):
        i, j = u // m, u % m
        for v in range(mp):
            j_v, k = v // p, v % p
            for w in range(pn):
                i_w, k_w = w % n, w // n
                expected = 1 if (j == j_v and i == i_w and k == k_w) else 0
                val = Fraction(0)
                for c in range(rank):
                    val += U[u][c] * V[v][c] * W[w][c]
                if val != expected:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n, m, p = 3, 5, 7
    
    # Recursive decomposition builder
    # Returns a list of terms (u_idx, v_idx, w_idx, coeff)
    # u_idx is local to A, v_idx local to B, w_idx local to C
    # Actually, easier to return U, V, W matrices for the subproblem
    # But combining matrices is tricky.
    # Let's return a list of (u_vec, v_vec, w_vec) where each is a tuple of (index, value)
    # Or simply build the global U, V, W directly.
    
    # We will use a helper to add a term to the global matrices
    # term: (i, j, k, u_val, v_val, w_val)
    # U[i*m+j][rank] += u_val
    # V[j*p+k][rank] += v_val
    # W[k*n+i][rank] += w_val
    
    def add_term(terms, i, j, k, u_val, v_val, w_val):
        # i, j, k are global indices
        terms.append((i, j, k, u_val, v_val, w_val))

    def get_decomp(n, m, p):
        # Returns list of terms: (i, j, k, u, v, w)
        # i in [0, n), j in [0, m), k in [0, p)
        
        if time.time() > deadline:
            return []
            
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B12-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to express this as sum of rank-1 tensors.
            # Each M is a product of sums.
            # We expand them.
            
            terms = []
            
            # M1: (A00+A11)(B00+B11)
            # A00*B00, A00*B11, A11*B00, A11*B11
            # Indices: A(i,j), B(j,k), C(i,k)
            # A00: i=0,j=0. B00: j=0,k=0. C00: i=0,k=0.
            # Term: u=1, v=1, w=1 for (0,0,0)
            # But we need to combine them into the 7 M's.
            
            # Let's just generate the 7 M's as separate rank-1 terms?
            # No, M1 is a single multiplication.
            # U_vec for M1: A00=1, A11=1.
            # V_vec for M1: B00=1, B11=1.
            # W_vec for M1: C00=1, C11=1 (from C22=M1...), C01=-1 (from C11=-M5? No)
            
            # Let's trace C contributions:
            # C00: M1 + M4 - M5 + M7
            # C01: M3 + M5
            # C10: M2 + M4
            # C11: M1 - M2 + M3 + M6
            
            # M1:
            # U: A00=1, A11=1
            # V: B00=1, B11=1
            # W: C00=1, C11=1
            add_term(terms, 0, 0, 0, 1, 1, 1) # This is not right.
            
            # The term (i,j,k) corresponds to the element.
            # A term is defined by:
            # u_vec: non-zero at (i1, j1), (i2, j2)...
            # v_vec: non-zero at (j1, k1), (j2, k2)...
            # w_vec: non-zero at (i1, k1), (i2, k2)...
            
            # For M1:
            # u_vec has 1 at (0,0) and (1,1).
            # v_vec has 1 at (0,0) and (1,1).
            # w_vec has 1 at (0,0) and (1,1).
            # So we add a term with u_vec, v_vec, w_vec.
            
            def add_rank1_term(terms, u_dict, v_dict, w_dict):
                # u_dict: {(i,j): val}
                # v_dict: {(j,k): val}
                # w_dict: {(i,k): val}
                # We need to store this in the global U, V, W later.
                # For now, store the dicts.
                terms.append((u_dict, v_dict, w_dict))
                
            # M1
            add_rank1_term(terms, 
                           {(0,0): 1, (1,1): 1}, 
                           {(0,0): 1, (1,1): 1}, 
                           {(0,0): 1, (1,1): 1})
            
            # M2: (A10+A11)B00
            add_rank1_term(terms, 
                           {(1,0): 1, (1,1): 1}, 
                           {(0,0): 1}, 
                           {(1,0): 1})
            
            # M3: A00(B10-B11)
            add_rank1_term(terms, 
                           {(0,0): 1}, 
                           {(1,0): 1, (1,1): -1}, 
                           {(0,1): 1})
            
            # M4: A11(B01-B00)
            add_rank1_term(terms, 
                           {(1,1): 1}, 
                           {(0,1): 1, (0,0): -1}, 
                           {(1,1): 1})
            
            # M5: (A00+A01)B11
            add_rank1_term(terms, 
                           {(0,0): 1, (0,1): 1}, 
                           {(1,1): 1}, 
                           {(0,0): -1, (0,1): 1})
            
            # M6: (A10-A00)(B00+B01)
            add_rank1_term(terms, 
                           {(1,0): 1, (0,0): -1}, 
                           {(0,0): 1, (0,1): 1}, 
                           {(1,1): 1})
            
            # M7: (A01-A11)(B10+B11)
            add_rank1_term(terms, 
                           {(0,1): 1, (1,1): -1}, 
                           {(1,0): 1, (1,1): 1}, 
                           {(0,0): 1, (1,1): -1})
            
            return terms

        # Base case: any dim is 1
        if n == 1 or m == 1 or p == 1:
            terms = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Only valid if indices match for dot product
                        # For 1xnmxp, it's just dot products
                        # If n=1, C_0k = sum_j A_0j B_jk
                        # Term: u={(0,j):1}, v={(j,k):1}, w={(0,k):1}
                        # This is schoolbook.
                        u_dict = {(i, j): 1}
                        v_dict = {(j, k): 1}
                        w_dict = {(i, k): 1}
                        terms.append((u_dict, v_dict, w_dict))
            return terms

        # Recursive case
        # Split dimensions
        # Strategy: Split largest dim, or split all if possible to hit 2x2x2
        
        # Simple split:
        # n_split = n // 2
        # m_split = m // 2
        # p_split = p // 2
        
        # But we want to hit 2x2x2.
        # If n >= 2, m >= 2, p >= 2:
        # Split n into n1, n2 such that n1, n2 are 1 or 2 if possible.
        
        def split_dim(dim):
            if dim == 1:
                return [1]
            if dim == 2:
                return [2] # Don't split 2, use Strassen
            if dim == 3:
                return [1, 2]
            if dim == 4:
                return [2, 2]
            if dim == 5:
                return [2, 3]
            if dim == 6:
                return [2, 4]
            if dim == 7:
                return [3, 4]
            # General
            half = dim // 2
            return [half, dim - half]
        
        n_splits = split_dim(n)
        m_splits = split_dim(m)
        p_splits = split_dim(p)
        
        terms = []
        for i1, i2 in zip(n_splits, [0]*len(n_splits)): # Dummy
            pass
            
        # Iterate over all combinations of splits
        i_start = 0
        for n1 in n_splits:
            j_start = 0
            for m1 in m_splits:
                k_start = 0
                for p1 in p_splits:
                    # Get sub-decomposition
                    sub_terms = get_decomp(n1, m1, p1)
                    
                    # Map local indices to global
                    for u_dict, v_dict, w_dict in sub_terms:
                        new_u = {}
                        new_v = {}
                        new_w = {}
                        
                        for (li, lj), val in u_dict.items():
                            gi = li + i_start
                            gj = lj + j_start
                            new_u[(gi, gj)] = val
                            
                        for (lj, lk), val in v_dict.items():
                            gj = lj + j_start
                            gk = lk + k_start
                            new_v[(gj, gk)] = val
                            
                        for (li, lk), val in w_dict.items():
                            gi = li + i_start
                            gk = lk + k_start
                            new_w[(gi, gk)] = val
                            
                        terms.append((new_u, new_v, new_w))
                        
                    k_start += p1
                j_start += m1
            i_start += n1
            
        return terms

    # Get the decomposition
    terms = get_decomp(n, m, p)
    
    # Build U, V, W
    rank = len(terms)
    nm, mp, pn = n * m, m * p, p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (i, k), val in w_dict.items():
            W[k * n + i][c] = val
            
    # Convert to Fraction for exact verification
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        return {
            "U": [[float(x) for x in row] for row in U_frac],
            "V": [[float(x) for x in row] for row in V_frac],
            "W": [[float(x) for x in row] for row in W_frac]
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
