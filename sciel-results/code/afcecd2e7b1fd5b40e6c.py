import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Post-process: remove linearly dependent terms
    def remove_dependencies(terms):
        if not terms:
            return terms
        # Build vectors for each term: concatenate U, V, W parts
        # A term is a tensor T_r = U_r ⊗ V_r ⊗ W_r
        # Two terms are dependent if their tensors are linearly dependent
        # Simpler: check if any term's W vector is a scalar multiple of another's
        # AND their U,V structures allow merging
        # Most robust: build the full tensor for each term and do QR
        # But that's expensive for 125^3. Instead, check W-index overlap.
        
        # Group terms by their W index (if W has single nonzero entry)
        # If two terms have same W index, same U index, same V index → duplicate
        # If two terms have same W index but different U/V → can't simply merge
        
        # Check for exact duplicates first
        seen = set()
        unique_terms = []
        for t in terms:
            u_dict, v_dict, w_dict = t
            key = (tuple(sorted(u_dict.items())), tuple(sorted(v_dict.items())), tuple(sorted(w_dict.items())))
            if key not in seen:
                seen.add(key)
                unique_terms.append(t)
        return unique_terms
    
    # Block decomposition: 5 = 2 + 2 + 1
    # For 2x2x2 blocks use Strassen (7 terms)
    # For all other blocks use schoolbook
    
    blocks = [(0, 2), (2, 4), (4, 5)]
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        if time.time() > deadline:
            return
        # Filter zeros
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))
    
    def add_strassen_2x2(r1, c1, k1):
        """Add 7 Strassen terms for a 2x2x2 block at (r1,c1,k1)."""
        # Local indices: A[i,j] at global (r1+i, c1+j), i,j in {0,1}
        #                B[j,k] at global (c1+j, k1+k), j,k in {0,1}
        #                C[i,k] at global (k1+k, r1+i), k,i in {0,1}
        # W convention: W[k*n + i, r]
        
        # U indices: (r1+i)*m + (c1+j)
        # V indices: (c1+j)*p + (k1+k)
        # W indices: (k1+k)*n + (r1+i)
        
        A00 = (r1) * m + c1
        A01 = (r1) * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        # M2 = (A10+A11)B00 -> contributes to C10 and C11 (with -1)
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        # M3 = A00(B01-B11) -> contributes to C01 and C11
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        # M4 = A11(B10-B00) -> contributes to C00 and C10
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        # M5 = (A00+A01)B11 -> contributes to C00 (with -1) and C01
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
    
    def add_schoolbook(r1, c1, k1, ar, ac, bc):
        """Add schoolbook terms for a block of size ar x ac x bc."""
        for ii in range(ar):
            for jj in range(ac):
                for kk in range(bc):
                    u_idx = (r1 + ii) * m + (c1 + jj)
                    v_idx = (c1 + jj) * p + (k1 + kk)
                    w_idx = (k1 + kk) * n + (r1 + ii)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Iterate over all block combinations
    for r_idx, (r1, r2) in enumerate(blocks):
        for c_idx, (c1, c2) in enumerate(blocks):
            for k_idx, (k1, k2) in enumerate(blocks):
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                else:
                    add_schoolbook(r1, c1, k1, ar, ac, bc)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    terms = remove_dependencies(terms)
    rank = len(terms)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
