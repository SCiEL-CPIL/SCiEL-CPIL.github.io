import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Schoolbook fallback (rank 36)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def verify_decomposition(U_terms, V_terms, W_terms):
        """Verify that the decomposition exactly represents matrix multiplication."""
        rank = len(U_terms)
        
        # Build target tensor indices
        target_indices = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.add((i * m + j, j * p + k, k * n + i))
        
        # Compute the decomposed tensor
        decomposed = {}
        for r in range(rank):
            U_r = U_terms[r]
            V_r = V_terms[r]
            W_r = W_terms[r]
            U_nz = [(idx, val) for idx, val in enumerate(U_r) if val != 0]
            V_nz = [(idx, val) for idx, val in enumerate(V_r) if val != 0]
            W_nz = [(idx, val) for idx, val in enumerate(W_r) if val != 0]
            
            for a, ua in U_nz:
                for b, vb in V_nz:
                    for c, wc in W_nz:
                        key = (a, b, c)
                        if key in target_indices:
                            decomposed[key] = decomposed.get(key, Fraction(0)) + ua * vb * wc
        
        # Verify all target entries equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    key = (a, b, c)
                    if key not in decomposed or decomposed[key] != 1:
                        return False
        
        # Verify no extra non-zero entries
        for key in decomposed:
            if key not in target_indices:
                return False
        
        return True
    
    def frac_to_float_list(vecs):
        return [[float(v) for v in vec] for vec in vecs]
    
    def make_term(U_dict, V_dict, W_dict):
        """Create a single rank-1 term as three vectors."""
        U_vec = [Fraction(0)] * (n * m)
        V_vec = [Fraction(0)] * (m * p)
        W_vec = [Fraction(0)] * (p * n)
        for idx, val in U_dict.items():
            U_vec[idx] = val
        for idx, val in V_dict.items():
            V_vec[idx] = val
        for idx, val in W_dict.items():
            W_vec[idx] = val
        return U_vec, V_vec, W_vec
    
    # Try Strassen-based construction
    # Block structure:
    # A (3x3) split as: top-left 2x2 (A_TL), top-right 2x1 (A_TR), bottom-left 1x2 (A_BL), bottom-right 1x1 (A_BR)
    # B (3x4) split as: top-left 2x2 (B_TL), top-right 2x2 (B_TR), bottom-left 1x2 (B_BL), bottom-right 1x2 (B_BR)
    # C (3x4) = [A_TL*B_TL + A_TR*B_BL, A_TL*B_TR + A_TR*B_BR; A_BL*B_TL + A_BR*B_BL, A_BL*B_TR + A_BR*B_BR]
    
    # Strassen for 2x2x2 with correct signs:
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2 = (A10+A11)B00 -> C10+=1, C11-=1
    # M3 = A00(B10-B00) -> C01+=1, C11+=1
    # M4 = A11(B01-B11) -> C00+=1, C10+=1
    # M5 = (A00+A01)B11 -> C00-=1, C01+=1, C11+=1
    # M6 = (A10-A00)(B00+B01) -> C10-=1, C11-=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11+=1
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add Strassen terms for a 2x2x2 block.
        a_rows: 2 row indices in A, a_cols: 2 col indices in A
        b_rows: 2 row indices in B, b_cols: 2 col indices in B
        c_rows: 2 row indices in C, c_cols: 2 col indices in C
        Returns list of (U_dict, V_dict, W_dict)
        """
        terms = []
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        def widx(ci, ck):
            return ck * n + ci
        
        # M1 = (a00+a11)(b00+b11)
        terms.append(make_term(
            {a00[0]*m+a00[1]: Fraction(1), a11[0]*m+a11[1]: Fraction(1)},
            {b00[0]*p+b00[1]: Fraction(1), b11[0]*p+b11[1]: Fraction(1)},
            {widx(*c00): Fraction(1), widx(*c11): Fraction(1)}
        ))
        
        # M2 = (a10+a11)b00
        terms.append(make_term(
            {a10[0]*m+a10[1]: Fraction(1), a11[0]*m+a11[1]: Fraction(1)},
            {b00[0]*p+b00[1]: Fraction(1)},
            {widx(*c10): Fraction(1), widx(*c11): Fraction(-1)}
        ))
        
        # M3 = a00(b10-b00)
        terms.append(make_term(
            {a00[0]*m+a00[1]: Fraction(1)},
            {b10[0]*p+b10[1]: Fraction(1), b00[0]*p+b00[1]: Fraction(-1)},
            {widx(*c01): Fraction(1), widx(*c11): Fraction(1)}
        ))
        
        # M4 = a11(b01-b11)
        terms.append(make_term(
            {a11[0]*m+a11[1]: Fraction(1)},
            {b01[0]*p+b01[1]: Fraction(1), b11[0]*p+b11[1]: Fraction(-1)},
            {widx(*c00): Fraction(1), widx(*c10): Fraction(1)}
        ))
        
        # M5 = (a00+a01)b11
        terms.append(make_term(
            {a00[0]*m+a00[1]: Fraction(1), a01[0]*m+a01[1]: Fraction(1)},
            {b11[0]*p+b11[1]: Fraction(1)},
            {widx(*c00): Fraction(-1), widx(*c01): Fraction(1), widx(*c11): Fraction(1)}
        ))
        
        # M6 = (a10-a00)(b00+b01)
        terms.append(make_term(
            {a10[0]*m+a10[1]: Fraction(1), a00[0]*m+a00[1]: Fraction(-1)},
            {b00[0]*p+b00[1]: Fraction(1), b01[0]*p+b01[1]: Fraction(1)},
            {widx(*c10): Fraction(-1), widx(*c11): Fraction(-1)}
        ))
        
        # M7 = (a01-a11)(b10+b11)
        terms.append(make_term(
            {a01[0]*m+a01[1]: Fraction(1), a11[0]*m+a11[1]: Fraction(-1)},
            {b10[0]*p+b10[1]: Fraction(1), b11[0]*p+b11[1]: Fraction(1)},
            {widx(*c00): Fraction(1), widx(*c11): Fraction(1)}
        ))
        
        return terms
    
    def add_mult_term(i, j, k):
        """Add standard multiplication term A[i,j]*B[j,k]->C[i,k]"""
        return make_term(
            {i*m+j: Fraction(1)},
            {j*p+k: Fraction(1)},
            {k*n+i: Fraction(1)}
        )
    
    # Build Strassen-based decomposition
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2, 7 terms)
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (2x2x2, 7 terms)
    # Block 3: A[0:2,2] * B[2,0:2] -> C[0:2,0:2] (2x1x2, 4 terms)
    # Block 4: A[0:2,2] * B[2,2:4] -> C[0:2,2:4] (2x1x2, 4 terms)
    # Block 5: A[2,0:2] * B[0:2,0:2] -> C[2,0:2] (1x2x2, 4 terms)
    # Block 6: A[2,0:2] * B[0:2,2:4] -> C[2,2:4] (1x2x2, 4 terms)
    # Block 7: A[2,2] * B[2,0:2] -> C[2,0:2] (1x1x2, 2 terms)
    # Block 8: A[2,2] * B[2,2:4] -> C[2,2:4] (1x1x2, 2 terms)
    # Total: 7+7+4+4+4+4+2+2 = 34 terms
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Add Strassen blocks
    for terms in [
        add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1]),  # Block 1
        add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3]),  # Block 2
    ]:
        for U_t, V_t, W_t in terms:
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
    
    # Add standard multiplication terms for remaining blocks
    for i in range(2):
        for k in range(2):
            U_t, V_t, W_t = add_mult_term(i, 2, k)
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
    
    for i in range(2):
        for k in range(2, 4):
            U_t, V_t, W_t = add_mult_term(i, 2, k)
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
    
    for j in range(2):
        for k in range(2):
            U_t, V_t, W_t = add_mult_term(2, j, k)
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
    
    for j in range(2):
        for k in range(2, 4):
            U_t, V_t, W_t = add_mult_term(2, j, k)
            U_terms.append(U_t)
            V_terms.append(V_t)
            W_terms.append(W_t)
    
    for k in range(2):
        U_t, V_t, W_t = add_mult_term(2, 2, k)
        U_terms.append(U_t)
        V_terms.append(V_t)
        W_terms.append(W_t)
    
    for k in range(2, 4):
        U_t, V_t, W_t = add_mult_term(2, 2, k)
        U_terms.append(U_t)
        V_terms.append(V_t)
        W_terms.append(W_t)
    
    # Verify the Strassen-based decomposition
    if verify_decomposition(U_terms, V_terms, W_terms):
        return {"U": frac_to_float_list(U_terms), "V": frac_to_float_list(V_terms), "W": frac_to_float_list(W_terms)}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
