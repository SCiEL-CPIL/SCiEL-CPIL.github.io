import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
        col += 1
        
    # Define Strassen terms for 2x2 by 2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Note: Global B index is j*p + k.
    # Global A index is i*m + j.
    # Global C index is k*n + i.
    
    def get_strassen_terms():
        terms = []
        # M1 = (A00 + A11)(B00 + B11)
        # A: [1, 0, 0, 1], B: [1, 0, 0, 1]
        # C: C00 += 1, C11 += -1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Wait, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01+B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # u, v, w vectors for each M
        # u is length 4, v is length 4, w is length 4
        # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C00 += 1, C11 += 1? No.
        # C00 coeff 1, C11 coeff 1?
        # C00 = M1 ... C11 = M1 ...
        # So w = [1, 0, 0, 1]
        terms.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # C10 += 1, C11 += -1
        terms.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 0, 1, -1]
        w = [0, 1, 0, 1] # C01 += 1, C11 += 1
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 1, 0, 1]
        w = [1, 0, 1, 0] # C00 += 1, C10 += 1
        terms.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0] # C00 += -1, C01 += 1
        terms.append((u, v, w))
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C11 += 1
        terms.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C00 += 1
        terms.append((u, v, w))
        
        return terms

    def get_schoolbook_terms_221():
        # 2x2 by 2x1
        # A: 0,1,2,3
        # B: 0,1 (only col 0)
        # C: 0,1 (only col 0)
        terms = []
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        u1 = [1, 0, 0, 0]; v1 = [1, 0]; w1 = [1, 0]
        terms.append((u1, v1, w1))
        u2 = [0, 1, 0, 0]; v2 = [0, 1]; w2 = [1, 0]
        terms.append((u2, v2, w2))
        u3 = [0, 0, 1, 0]; v3 = [1, 0]; w3 = [0, 1]
        terms.append((u3, v3, w3))
        u4 = [0, 0, 0, 1]; v4 = [0, 1]; w4 = [0, 1]
        terms.append((u4, v4, w4))
        return terms

    # We need to map local terms to global U, V, W
    # Block 1: A1 (j=0,1), B1 (j=0,1)
    # Block 2: A2 (j=2,3), B2 (j=2,3)
    
    def process_block(a_offset, b_offset, terms_list):
        # a_offset: 0 or 2 (j index offset in A)
        # b_offset: 0 or 2 (j index offset in B)
        # terms_list: list of (u, v, w) local vectors
        # u is length 4 (2x2 A)
        # v is length 4 or 2 (2x2 or 2x1 B) -- Wait, B is 2x5 in the block?
        # No, the terms are for specific sub-blocks of B.
        
        # For Strassen (2x2 by 2x2), v is length 4.
        # For Schoolbook (2x2 by 2x1), v is length 2.
        
        # We need to handle the mapping carefully.
        pass

    # Let's just generate the vectors for the whole 2x2 by 2x5 block directly.
    # Block 1: A1 (j=0,1), B1 (j=0,1, k=0..4)
    # B1 is 2x5.
    # We split B1 into B1_0 (j=0,1, k=0,1), B1_1 (j=0,1, k=2,3), B1_2 (j=0,1, k=4).
    # B1_0 is 2x2. Use Strassen.
    # B1_1 is 2x2. Use Strassen.
    # B1_2 is 2x1. Use Schoolbook.
    
    def add_block_terms(a_j_start, b_j_start):
        nonlocal col
        # a_j_start: 0 or 2
        # b_j_start: 0 or 2
        
        # Sub-block 0: k=0,1
        strassen_terms = get_strassen_terms()
        for u, v, w in strassen_terms:
            # Map u to global A
            gu = np.zeros(n*m)
            for i in range(2):
                for j in range(2):
                    gu[i*m + (a_j_start + j)] = u[i*2 + j]
            
            # Map v to global B
            gv = np.zeros(m*p)
            for j in range(2):
                for k in range(2):
                    gv[(b_j_start + j)*p + k] = v[j*2 + k]
            
            # Map w to global C
            gw = np.zeros(p*n)
            for i in range(2):
                for k in range(2):
                    gw[k*n + i] = w[i*2 + k]
            
            add_term(gu, gv, gw)
            
        # Sub-block 1: k=2,3
        strassen_terms = get_strassen_terms()
        for u, v, w in strassen_terms:
            gu = np.zeros(n*m)
            for i in range(2):
                for j in range(2):
                    gu[i*m + (a_j_start + j)] = u[i*2 + j]
            
            gv = np.zeros(m*p)
            for j in range(2):
                for k in range(2):
                    gv[(b_j_start + j)*p + (2 + k)] = v[j*2 + k]
            
            gw = np.zeros(p*n)
            for i in range(2):
                for k in range(2):
                    gw[(2 + k)*n + i] = w[i*2 + k]
            
            add_term(gu, gv, gw)
            
        # Sub-block 2: k=4
        schoolbook_terms = get_schoolbook_terms_221()
        for u, v, w in schoolbook_terms:
            gu = np.zeros(n*m)
            for i in range(2):
                for j in range(2):
                    gu[i*m + (a_j_start + j)] = u[i*2 + j]
            
            gv = np.zeros(m*p)
            for j in range(2):
                # v is length 2: v[0] is B[j,0], v[1] is B[j,1]
                # Here k=4 is the only column.
                # In schoolbook 2x1, v[0] corresponds to B[0,0], v[1] to B[1,0].
                # So v[j] maps to B[b_j_start+j, 4]
                gv[(b_j_start + j)*p + 4] = v[j]
            
            gw = np.zeros(p*n)
            for i in range(2):
                # w is length 2: w[0] is C[0,0], w[1] is C[1,0]
                # Here k=4.
                gw[4*n + i] = w[i]
            
            add_term(gu, gv, gw)

    # Block 1: A1 (j=0,1), B1 (j=0,1)
    add_block_terms(0, 0)
    
    # Block 2: A2 (j=2,3), B2 (j=2,3)
    add_block_terms(2, 2)
    
    return {"U": U, "V": V, "W": W}
