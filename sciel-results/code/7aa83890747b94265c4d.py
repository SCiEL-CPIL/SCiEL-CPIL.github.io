import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing contributions from blocks.
    # Each contribution is a list of (u_idx, v_idx, w_idx, coeff)
    # But the interface requires U, V, W matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # So we will collect all terms and then assemble the matrices.
    
    terms = []
    
    # Helper to add a schoolbook term A[i,j] * B[j,k]
    def add_schoolbook(i, j, k, coeff=1):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    # Helper to add a Strassen term for 2x2x2
    # Indices are local (0,1) for the 2x2 block
    # We need to map local to global
    # Let local i, j, k be 0 or 1
    # Global i = i_offset + i
    # Global j = j_offset + j
    # Global k = k_offset + k
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # M1 = (A00+A11)(B00+B11)
        # A00: i=0, j=0. A11: i=1, j=1.
        # B00: j=0, k=0. B11: j=1, k=1.
        # (A00+A11) * (B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # We need to represent this as a single rank-1 term in the tensor decomposition?
        # No, Strassen works by defining intermediate products M.
        # The decomposition of the tensor T = sum M_l * U_l * V_l * W_l
        # For Strassen, the "multiplication" is the scalar product of the expanded terms.
        # So U_l has entries for A, V_l for B, W_l for C.
        # M1 corresponds to:
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1, C01=-1, C10=-1 ? No.
        
        # Let's recall the Strassen formulas for C:
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A11-A10)(B10+B11)
        
        # Each M_l is a rank-1 term in the decomposition of the 2x2x2 tensor.
        # We add 7 terms.
        
        # M1
        u = {}
        v = {}
        w = {}
        # A00, A11
        u[(i_off+0, j_off+0)] = 1
        u[(i_off+1, j_off+1)] = 1
        # B00, B11
        v[(j_off+0, k_off+0)] = 1
        v[(j_off+1, k_off+1)] = 1
        # C00, C11, C01, C10
        w[(k_off+0, i_off+0)] = 1
        w[(k_off+1, i_off+1)] = 1
        w[(k_off+1, i_off+0)] = -1
        w[(k_off+0, i_off+1)] = -1
        terms.append((u, v, w))
        
        # M2
        u = {}
        v = {}
        w = {}
        u[(i_off+1, j_off+0)] = 1
        u[(i_off+1, j_off+1)] = 1
        v[(j_off+0, k_off+0)] = 1
        w[(k_off+0, i_off+1)] = 1
        w[(k_off+1, i_off+1)] = 1
        terms.append((u, v, w))
        
        # M3
        u = {}
        v = {}
        w = {}
        u[(i_off+0, j_off+0)] = 1
        v[(j_off+0, k_off+1)] = 1
        v[(j_off+1, k_off+1)] = -1
        w[(k_off+1, i_off+0)] = 1
        w[(k_off+1, i_off+1)] = 1
        terms.append((u, v, w))
        
        # M4
        u = {}
        v = {}
        w = {}
        u[(i_off+1, j_off+1)] = 1
        v[(j_off+1, k_off+0)] = 1
        v[(j_off+0, k_off+0)] = -1
        w[(k_off+0, i_off+1)] = 1
        w[(k_off+1, i_off+1)] = 1
        terms.append((u, v, w))
        
        # M5
        u = {}
        v = {}
        w = {}
        u[(i_off+0, j_off+0)] = 1
        u[(i_off+0, j_off+1)] = 1
        v[(j_off+1, k_off+1)] = 1
        w[(k_off+1, i_off+0)] = 1
        w[(k_off+1, i_off+1)] = 1
        terms.append((u, v, w))
        
        # M6
        u = {}
        v = {}
        w = {}
        u[(i_off+1, j_off+0)] = 1
        u[(i_off+0, j_off+0)] = -1
        v[(j_off+0, k_off+0)] = 1
        v[(j_off+0, k_off+1)] = 1
        w[(k_off+0, i_off+1)] = 1
        w[(k_off+1, i_off+1)] = 1
        terms.append((u, v, w))
        
        # M7
        u = {}
        v = {}
        w = {}
        u[(i_off+1, j_off+1)] = 1
        u[(i_off+1, j_off+0)] = -1
        v[(j_off+1, k_off+0)] = 1
        v[(j_off+1, k_off+1)] = 1
        w[(k_off+0, i_off+1)] = 1
        w[(k_off+1, i_off+1)] = 1
        terms.append((u, v, w))

    # Decomposition Plan:
    # Split n=3 into 2+1 (i_off 0, 1)
    # Split m=5 into 2+3 (j_off 0, 2)
    # Split p=7 into 2+5 (k_off 0, 2)
    
    # Block 11: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # Use Strassen for A11*B11
    add_strassen_2x2x2(0, 0, 0)
    # Plus A12 (2x3) * B21 (3x2) -> C11 (2x2)
    # A12: i in 0..1, j in 2..4
    # B21: j in 2..4, k in 0..1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
                
    # Block 12: A11 (2x2) * B12 (2x5) -> C12 (2x5)
    # A11: i 0..1, j 0..1
    # B12: j 0..1, k 2..6
    for i in range(2):
        for j in range(2):
            for k in range(2, 7):
                add_schoolbook(i, j, k)
    # Plus A12 (2x3) * B22 (3x5) -> C12 (2x5)
    # A12: i 0..1, j 2..4
    # B22: j 2..4, k 2..6
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 7):
                add_schoolbook(i, j, k)
                
    # Block 21: A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # A21: i 2, j 0..1
    # B11: j 0..1, k 0..1
    for i in range(2, 3):
        for j in range(2):
            for k in range(2):
                add_schoolbook(i, j, k)
    # Plus A22 (1x3) * B21 (3x2) -> C21 (1x2)
    # A22: i 2, j 2..4
    # B21: j 2..4, k 0..1
    for i in range(2, 3):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
                
    # Block 22: A21 (1x2) * B12 (2x5) -> C22 (1x5)
    # A21: i 2, j 0..1
    # B12: j 0..1, k 2..6
    for i in range(2, 3):
        for j in range(2):
            for k in range(2, 7):
                add_schoolbook(i, j, k)
    # Plus A22 (1x3) * B22 (3x5) -> C22 (1x5)
    # A22: i 2, j 2..4
    # B22: j 2..4, k 2..6
    for i in range(2, 3):
        for j in range(2, 5):
            for k in range(2, 7):
                add_schoolbook(i, j, k)

    # Now, assemble U, V, W
    # We have a list of terms.
    # Some terms are dicts (Strassen), some are tuples (Schoolbook).
    # Let's convert all to a uniform format: list of (u_dict, v_dict, w_dict)
    
    unified_terms = []
    for t in terms:
        if isinstance(t, tuple) and len(t) == 3:
            # Schoolbook
            u_idx, v_idx, w_idx = t
            u = {u_idx: 1}
            v = {v_idx: 1}
            w = {w_idx: 1}
            unified_terms.append((u, v, w))
        else:
            # Strassen
            u, v, w = t
            unified_terms.append((u, v, w))
            
    rank = len(unified_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(unified_terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
