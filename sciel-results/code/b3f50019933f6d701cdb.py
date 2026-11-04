import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # We will build the decomposition by summing up contributions from blocks.
    # We accumulate U, V, W in lists of columns (rank terms).
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    # u_vec: size n*m
    # v_vec: size m*p
    # w_vec: size p*n
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to create zero vectors
    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)
    
    # Helper to set values in vectors
    def set_u(vec, i, j, val):
        vec[i * m + j] += val
    def set_v(vec, j, k, val):
        vec[j * p + k] += val
    def set_w(vec, k, i, val):
        vec[k * n + i] += val

    # Strassen 2x2x2 decomposition
    # Returns list of (u, v, w) tuples
    def strassen_2x2():
        terms = []
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 0, 0, 1); set_u(u, 1, 1, 1)
        set_v(v, 0, 0, 1); set_v(v, 1, 1, 1)
        set_w(w, 0, 0, 1); set_w(w, 1, 1, 1)
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00 -> C10, C11
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 1, 0, 1); set_u(u, 1, 1, 1)
        set_v(v, 0, 0, 1)
        set_w(w, 0, 1, 1); set_w(w, 1, 1, -1) # C10 (k=0,i=1), C11 (k=1,i=1)
        # Check C10: M2. C11: -M2.
        terms.append((u, v, w))
        
        # M3 = A00(B10-B11) -> C01, C11
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 0, 0, 1)
        set_v(v, 1, 0, 1); set_v(v, 1, 1, -1)
        set_w(w, 0, 0, 1); set_w(w, 1, 1, 1) # C01 (k=0,i=0)? No.
        # C01 is k=0, i=0? No, C is 2x2. C[i][k].
        # W index is k*n + i.
        # C01: i=0, k=1. Index 1*2+0 = 2.
        # C11: i=1, k=1. Index 1*2+1 = 3.
        # Let's re-verify W indices.
        # W[k*n + i].
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        
        # Re-evaluating M3:
        # M3 = A00(B10-B11)
        # C01 = M3. (i=0, k=1). W[2] = 1.
        # C11 = M3. (i=1, k=1). W[3] = 1.
        set_w(w, 1, 0, 1); set_w(w, 1, 1, 1)
        terms.append((u, v, w))
        
        # M4 = A11(B01-B00) -> C00, C10
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 1, 1, 1)
        set_v(v, 0, 1, 1); set_v(v, 0, 0, -1)
        # C00 = M4. (i=0, k=0). W[0] = 1.
        # C10 = M4. (i=1, k=0). W[1] = 1.
        set_w(w, 0, 0, 1); set_w(w, 0, 1, 1)
        terms.append((u, v, w))
        
        # M5 = (A00+A10)B11 -> C00, C01
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 0, 0, 1); set_u(u, 1, 0, 1)
        set_v(v, 1, 1, 1)
        # C00 = -M5. (i=0, k=0). W[0] = -1.
        # C01 = M5. (i=0, k=1). W[2] = 1.
        set_w(w, 0, 0, -1); set_w(w, 1, 0, 1)
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01) -> C11
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 1, 0, 1); set_u(u, 0, 0, -1)
        set_v(v, 0, 0, 1); set_v(v, 0, 1, 1)
        # C11 = M6. (i=1, k=1). W[3] = 1.
        set_w(w, 1, 1, 1)
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11) -> C00
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, 0, 1, 1); set_u(u, 1, 1, -1)
        set_v(v, 1, 0, 1); set_v(v, 1, 1, 1)
        # C00 = M7. (i=0, k=0). W[0] = 1.
        set_w(w, 0, 0, 1)
        terms.append((u, v, w))
        
        return terms

    # Schoolbook 2x2x2 decomposition (Fallback or for smaller blocks)
    def schoolbook_2x2():
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i, j, 1)
                    set_v(v, j, k, 1)
                    set_w(w, k, i, 1)
                    terms.append((u, v, w))
        return terms

    # Schoolbook 2x1x2 (A 2x1, B 1x2)
    def schoolbook_2x1x2():
        terms = []
        for i in range(2):
            for k in range(2):
                u = zero_u(); v = zero_v(); w = zero_w()
                set_u(u, i, 0, 1) # A is 2x1, j=0
                set_v(v, 0, k, 1) # B is 1x2, j=0
                set_w(w, k, i, 1)
                terms.append((u, v, w))
        return terms

    # Schoolbook 2x2x1 (A 2x2, B 2x1)
    def schoolbook_2x2x1():
        terms = []
        for i in range(2):
            for j in range(2):
                u = zero_u(); v = zero_v(); w = zero_w()
                set_u(u, i, j, 1)
                set_v(v, j, 0, 1) # B is 2x1, k=0
                set_w(w, 0, i, 1)
                terms.append((u, v, w))
        return terms

    # Schoolbook 2x1x1 (A 2x1, B 1x1)
    def schoolbook_2x1x1():
        terms = []
        for i in range(2):
            u = zero_u(); v = zero_v(); w = zero_w()
            set_u(u, i, 0, 1)
            set_v(v, 0, 0, 1)
            set_w(w, 0, i, 1)
            terms.append((u, v, w))
        return terms

    # Map local indices to global indices
    # A blocks:
    # A11: i in 0..1, j in 0..1 -> Global j = j_loc
    # A12: i in 0..1, j in 0..1 -> Global j = j_loc + 2
    # A13: i in 0..1, j in 0..0 -> Global j = 4
    
    # B blocks:
    # B11: j in 0..1, k in 0..1 -> Global j = j_loc, k = k_loc
    # B12: j in 0..1, k in 0..1 -> Global j = j_loc, k = k_loc + 2
    # B13: j in 0..1, k in 0..0 -> Global j = j_loc, k = 4
    # B21: j in 0..1, k in 0..1 -> Global j = j_loc + 2, k = k_loc
    # B22: j in 0..1, k in 0..1 -> Global j = j_loc + 2, k = k_loc + 2
    # B23: j in 0..1, k in 0..0 -> Global j = j_loc + 2, k = 4
    # B31: j in 0..0, k in 0..1 -> Global j = 4, k = k_loc
    # B32: j in 0..0, k in 0..1 -> Global j = 4, k = k_loc + 2
    # B33: j in 0..0, k in 0..0 -> Global j = 4, k = 4
    
    # C blocks (Output):
    # C1: i in 0..1, k in 0..1 -> Global k = k_loc
    # C2: i in 0..1, k in 0..1 -> Global k = k_loc + 2
    # C3: i in 0..1, k in 0..0 -> Global k = 4

    def map_term(u, v, w, a_j_offset, b_j_offset, b_k_offset, c_k_offset):
        # u, v, w are local vectors of size 4, 4, 4 (for 2x2) or smaller.
        # We need to expand them to global size and shift indices.
        
        # This is tricky because the local vectors are already flattened.
        # It's safer to generate the terms as (i, j, val) lists and then map.
        pass

    # Better approach: Generate terms as lists of (idx, val) and then map.
    
    def generate_block_terms(decomp_func, a_j_offset, b_j_offset, b_k_offset, c_k_offset, a_j_size, b_j_size, b_k_size, c_k_size):
        # This is getting complicated. Let's just instantiate the vectors directly.
        
        terms_out = []
        # We need to know the local dimensions to iterate.
        # For Strassen 2x2: local n=2, m=2, p=2.
        # For Schoolbook 2x1x2: local n=2, m=1, p=2.
        
        # Let's just hardcode the mapping for each block type.
        
        # Block 1: A11 B11 (Strassen 2x2)
        # A11: j 0..1. B11: j 0..1, k 0..1. C: k 0..1.
        for u, v, w in strassen_2x2():
            gu = zero_u(); gv = zero_v(); gw = zero_w()
            # Map u (size 4)
            for idx, val in enumerate(u):
                if val != 0:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    j_glob = j_loc + a_j_offset
                    set_u(gu, i_loc, j_glob, val)
            # Map v (size 4)
            for idx, val in enumerate(v):
                if val != 0:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    j_glob = j_loc + b_j_offset
                    k_glob = k_loc + b_k_offset
                    set_v(gv, j_glob, k_glob, val)
            # Map w (size 4)
            for idx, val in enumerate(w):
                if val != 0:
                    k_loc = idx // 2
                    i_loc = idx % 2
                    k_glob = k_loc + c_k_offset
                    set_w(gw, k_glob, i_loc, val)
            terms_out.append((gu, gv, gw))
        return terms_out

    # Add Block 1: A11 B11
    terms = generate_block_terms(strassen_2x2, 0, 0, 0, 0)
    for u, v, w in terms: add_term(u, v, w)
    
    # Add Block 2: A11 B12
    # A11: j 0..1. B12: j 0..1, k 2..3. C: k 2..3.
    terms = generate_block_terms(strassen_2x2, 0, 0, 2, 2)
    for u, v, w in terms: add_term(u, v, w)
    
    # Add Block 3: A12 B21
    # A12: j 2..3. B21: j 2..3, k 0..1. C: k 0..1.
    terms = generate_block_terms(strassen_2x2, 2, 2, 0, 0)
    for u, v, w in terms: add_term(u, v, w)
    
    # Add Block 4: A12 B22
    # A12: j 2..3. B22: j 2..3, k 2..3. C: k 2..3.
    terms = generate_block_terms(strassen_2x2, 2, 2, 2, 2)
    for u, v, w in terms: add_term(u, v, w)
    
    # Block 5: A13 B31 (Schoolbook 2x1x2)
    # A13: j 4. B31: j 4, k 0..1. C: k 0..1.
    for u, v, w in schoolbook_2x1x2():
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        for idx, val in enumerate(u):
            if val != 0:
                i_loc = idx // 1 # m_loc=1
                j_loc = 0
                j_glob = 4
                set_u(gu, i_loc, j_glob, val)
        for idx, val in enumerate(v):
            if val != 0:
                j_loc = 0
                k_loc = idx
                j_glob = 4
                k_glob = k_loc
                set_v(gv, j_glob, k_glob, val)
        for idx, val in enumerate(w):
            if val != 0:
                k_loc = idx // 2
                i_loc = idx % 2
                k_glob = k_loc
                set_w(gw, k_glob, i_loc, val)
        add_term(gu, gv, gw)

    # Block 6: A13 B32 (Schoolbook 2x1x2)
    # A13: j 4. B32: j 4, k 2..3. C: k 2..3.
    for u, v, w in schoolbook_2x1x2():
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        for idx, val in enumerate(u):
            if val != 0:
                i_loc = idx
                j_glob = 4
                set_u(gu, i_loc, j_glob, val)
        for idx, val in enumerate(v):
            if val != 0:
                j_glob = 4
                k_loc = idx
                k_glob = k_loc + 2
                set_v(gv, j_glob, k_glob, val)
        for idx, val in enumerate(w):
            if val != 0:
                k_loc = idx // 2
                i_loc = idx % 2
                k_glob = k_loc + 2
                set_w(gw, k_glob, i_loc, val)
        add_term(gu, gv, gw)

    # Block 7: A11 B13 (Schoolbook 2x2x1)
    # A11: j 0..1. B13: j 0..1, k 4. C: k 4.
    for u, v, w in schoolbook_2x2x1():
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        for idx, val in enumerate(u):
            if val != 0:
                i_loc = idx // 2
                j_loc = idx % 2
                j_glob = j_loc
                set_u(gu, i_loc, j_glob, val)
        for idx, val in enumerate(v):
            if val != 0:
                j_loc = idx
                k_loc = 0
                j_glob = j_loc
                k_glob = 4
                set_v(gv, j_glob, k_glob, val)
        for idx, val in enumerate(w):
            if val != 0:
                k_loc = 0
                i_loc = idx
                k_glob = 4
                set_w(gw, k_glob, i_loc, val)
        add_term(gu, gv, gw)

    # Block 8: A12 B23 (Schoolbook 2x2x1)
    # A12: j 2..3. B23: j 2..3, k 4. C: k 4.
    for u, v, w in schoolbook_2x2x1():
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        for idx, val in enumerate(u):
            if val != 0:
                i_loc = idx // 2
                j_loc = idx % 2
                j_glob = j_loc + 2
                set_u(gu, i_loc, j_glob, val)
        for idx, val in enumerate(v):
            if val != 0:
                j_loc = idx
                k_loc = 0
                j_glob = j_loc + 2
                k_glob = 4
                set_v(gv, j_glob, k_glob, val)
        for idx, val in enumerate(w):
            if val != 0:
                k_loc = 0
                i_loc = idx
                k_glob = 4
                set_w(gw, k_glob, i_loc, val)
        add_term(gu, gv, gw)

    # Block 9: A13 B33 (Schoolbook 2x1x1)
    # A13: j 4. B33: j 4, k 4. C: k 4.
    for u, v, w in schoolbook_2x1x1():
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        for idx, val in enumerate(u):
            if val != 0:
                i_loc = idx
                j_glob = 4
                set_u(gu, i_loc, j_glob, val)
        for idx, val in enumerate(v):
            if val != 0:
                j_glob = 4
                k_glob = 4
                set_v(gv, j_glob, k_glob, val)
        for idx, val in enumerate(w):
            if val != 0:
                k_glob = 4
                i_loc = idx
                set_w(gw, k_glob, i_loc, val)
        add_term(gu, gv, gw)

    # Convert lists of columns to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
