import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    U_all = []
    V_all = []
    W_all = []
    
    def add_term(u, v, w):
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
    
    # Strassen 2x2x2 decomposition
    # Terms: (u, v, w) where u,v,w are length-4 vectors
    # u index: i*2+j, v index: j*2+k, w index: k*2+i
    # T[u_idx, v_idx, w_idx] = 1 for consistent (i,j,k)
    
    # Standard Strassen 7 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M4 + M7
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2: (A21+A22)B11
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # M3: A11(B12-B22)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # M4: A22(B21-B11)
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        # M5: (A11+A12)B22
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6: (A21-A11)(B11+B12)
        ([0,0,1,-1], [1,1,0,0], [0,0,1,0]),
        # M7: (A12-A22)(B21+B22)
        ([0,1,0,-1], [0,0,1,1], [0,0,0,1]),
    ]
    
    # Verify Strassen 2x2x2
    T_2x2 = np.zeros((4,4,4), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_2x2[i*2+j, j*2+k, k*2+i] = 1.0
    
    T_recon = np.zeros((4,4,4), dtype=float)
    for u, v, w in strassen_terms:
        T_recon += np.einsum('a,b,c->abc', np.array(u), np.array(v), np.array(w))
    
    strassen_ok = np.allclose(T_2x2, T_recon, atol=1e-10)
    
    # For 4x4 x 4x9, we use a block-based approach.
    # Split the 9 output columns into groups: [4, 4, 1]
    # For each group of size q, we do 4x4 x 4xq.
    # For q=4: use Strassen-style decomposition of 4x4x4 (rank 49 via two-level Strassen)
    # For q=1: schoolbook rank 16
    
    # Two-level Strassen for 4x4x4:
    # Outer level: treat 4x4x4 as 2x2x2 where each entry is a 2x2x2 tensor
    # Inner level: each 2x2x2 uses Strassen (rank 7)
    # Total: 7 * 7 = 49 terms
    
    # The outer Strassen operates on blocks:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2
    #
    # Outer Strassen terms (same as above but at block level):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M7
    #
    # Each Mx is a 2x2x2 tensor (product of two 2x2 matrices giving a 2x2 result)
    # Each Mx can be decomposed with rank 7 using inner Strassen.
    
    # For the full 4x4x4 tensor:
    # U index: i*4+j where i,j in 0..3
    # V index: j*4+k where j,k in 0..3
    # W index: k*4+i where k,i in 0..3
    
    # Block mapping:
    # i = 2*i0 + i1, j = 2*j0 + j1, k = 2*k0 + k1
    # U_idx = i*4+j = (2*i0+i1)*4 + (2*j0+j1) = 8*i0 + 4*i1 + 2*j0 + j1
    # V_idx = j*4+k = (2*j0+j1)*4 + (2*k0+k1) = 8*j0 + 4*j1 + 2*k0 + k1
    # W_idx = k*4+i = (2*k0+k1)*4 + (2*i0+i1) = 8*k0 + 4*k1 + 2*i0 + i1
    
    # For each outer Strassen term (t=0..6), we have coefficients for A blocks and B blocks.
    # For each inner Strassen term (s=0..6), we have the decomposition of the 2x2x2.
    
    # Let's define the outer Strassen more carefully.
    # Outer term t has:
    #   A coefficients: a_dict mapping (i0,j0) to coeff
    #   B coefficients: b_dict mapping (j0,k0) to coeff
    #   C coefficients: c_dict mapping (k0,i0) to coeff
    
    # Actually, let me just directly construct the 49 terms.
    
    # For a 4x4x4 multiplication at column offset co:
    # We generate 49 terms using two-level Strassen.
    
    def generate_4x4x4_terms(col_offset):
        """Generate 49 terms for 4x4 x 4x4 at given column offset."""
        terms = []
        
        # Outer Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict {(i0,j0): coeff} for 2x2 block of A
        # b_coeffs: dict {(j0,k0): coeff} for 2x2 block of B
        # c_coeffs: dict {(k0,i0): coeff} for 2x2 block of C
        
        outer_terms = [
            # M1: (A00+A11)(B00+B11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2: (A10+A11)B00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
            # M3: A00(B01-B11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}),
            # M4: A11(B10-B00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5: (A00+A01)B11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
            # M6: (A10-A00)(B00+B01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            # M7: (A01-A11)(B10+B11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}),
        ]
        
        # Inner Strassen terms for 2x2x2
        inner_terms = strassen_terms  # 7 terms
        
        for ot in outer_terms:
            a_coeffs, b_coeffs, c_coeffs = ot
            for it in inner_terms:
                u_inner, v_inner, w_inner = it
                
                # Build full u, v, w vectors
                u = np.zeros(U_rows, dtype=float)
                v = np.zeros(V_rows, dtype=float)
                w = np.zeros(W_rows, dtype=float)
                
                # For each block (i0,j0) in a_coeffs:
                for (i0, j0), ca in a_coeffs.items():
                    for ii in range(2):
                        for jj in range(2):
                            # u_inner index: ii*2+jj
                            c = ca * u_inner[ii*2+jj]
                            if c != 0:
                                u[(2*i0+ii)*4 + (2*j0+jj)] += c
                
                # For each block (j0,k0) in b_coeffs:
                for (j0, k0), cb in b_coeffs.items():
                    for jj in range(2):
                        for kk in range(2):
                            # v_inner index: jj*2+kk
                            c = cb * v_inner[jj*2+kk]
                            if c != 0:
                                v[(2*j0+jj)*4 + (2*k0+kk) + col_offset*4] += c
                
                # For each block (k0,i0) in c_coeffs:
                for (k0, i0), cc in c_coeffs.items():
                    for kk in range(2):
                        for ii in range(2):
                            # w_inner index: kk*2+ii
                            c = cc * w_inner[kk*2+ii]
                            if c != 0:
                                w[(2*k0+kk)*4 + (2*i0+ii) + col_offset*4] += c
                
                terms.append((u, v, w))
        
        return terms
    
    # Generate terms for each column group
    col_offset = 0
    col_groups = [4, 4, 1]
    
    for q in col_groups:
        if q == 4 and strassen_ok:
            # Use two-level Strassen: 49 terms
            terms = generate_4x4x4_terms(col_offset)
            for u, v, w in terms:
                add_term(u, v, w)
        else:
            # Schoolbook for 4x4 x 4xq
            for i in range(n):
                for j in range(m):
                    for k in range(q):
                        u = np.zeros(U_rows, dtype=float)
                        v = np.zeros(V_rows, dtype=float)
                        w = np.zeros(W_rows, dtype=float)
                        u[i * m + j] = 1.0
                        v[j * p + (col_offset + k)] = 1.0
                        w[(col_offset + k) * n + i] = 1.0
                        add_term(u, v, w)
        
        col_offset += q
    
    # Verify the full decomposition
    T = np.zeros((U_rows, V_rows, W_rows), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_recon = np.zeros((U_rows, V_rows, W_rows), dtype=float)
    for t in range(len(U_all)):
        T_recon += np.einsum('a,b,c->abc', U_all[t], V_all[t], W_all[t])
    
    if not np.allclose(T, T_recon, atol=1e-8):
        # Fallback: pure schoolbook
        U_all = []
        V_all = []
        W_all = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(U_rows, dtype=float)
                    v = np.zeros(V_rows, dtype=float)
                    w = np.zeros(W_rows, dtype=float)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    add_term(u, v, w)
    
    # Convert to list format
    rank = len(U_all)
    U = [[float(x) for x in row] for row in np.array(U_all).T]
    V = [[float(x) for x in row] for row in np.array(V_all).T]
    W = [[float(x) for x in row] for row in np.array(W_all).T]
    
    return {"U": U, "V": V, "W": W}
