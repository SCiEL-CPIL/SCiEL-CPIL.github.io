import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition using block structure
    # Split A (4x5) into two 2x5 blocks, each computed via 2+2+1 split of middle dim
    
    terms = []  # Each term is (u_vec, v_vec, w_vec) where these are sparse dicts
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying X(2x2) * Y(2x2) = Z(2x2)
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y10)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y01+y11)
    # Z00 = M1 - M5 + M3 + M7
    # Z01 = M4 + M5
    # Z10 = M2 + M6
    # Z11 = M1 - M2 + M4 + M6
    
    # For 2x2x2, the tensor indices:
    # U index: i*m+j = i*2+j (i,j in {0,1})
    # V index: j*p+k = j*2+k (j,k in {0,1})
    # W index: k*n+i = k*2+i (k,i in {0,1})
    
    # Strassen terms for 2x2x2:
    # Each term s has U entries (for i,j), V entries (for j,k), W entries (for k,i)
    # Term 1: M1 = (x00+x11)(y00+y11)
    #   U: (0,0)->1, (1,1)->1  => U[0]=1, U[3]=1
    #   V: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->1  => V[0]=1, V[1]=1, V[2]=1, V[3]=1
    #   W: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->1  => W[0]=1, W[1]=1, W[2]=1, W[3]=1
    #   But wait, this contributes to ALL of Z00, Z01, Z10, Z11? No...
    
    # Actually, in the tensor decomposition, each rank-1 term contributes to specific (i,j,k) entries.
    # For Strassen, the 7 products are intermediate, and the final C entries are linear combinations.
    # In the tensor decomposition T[i,j,k] = sum_s U[i,j,s]*V[j,k,s]*W[k,i,s],
    # each s corresponds to one "multiplication" in the algorithm.
    
    # Let me re-derive. For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # A rank-7 decomposition means: T[i,j,k] = sum_{s=0}^{6} U[i*2+j, s] * V[j*2+k, s] * W[k*2+i, s]
    
    # Strassen's algorithm computes 7 products M1..M7, and C entries are linear combos.
    # In tensor terms, each M_s corresponds to a rank-1 term where:
    # U and V encode the linear combinations of A and B entries
    # W encodes how M_s contributes to C entries
    
    # M1 = (A00+A11)(B00+B11)
    #   A part: A00 + A11 => U[0,0]=1, U[1,1]=1 (i=0,j=0 and i=1,j=1)
    #   B part: B00 + B11 => V[0,0]=1, V[1,1]=1 (j=0,k=0 and j=1,k=1)
    #   C part: contributes to C00 (+1), C01 (+1), C10 (+1), C11 (-1)
    #   W[0,0]=1 (k=0,i=0), W[1,0]=1 (k=1,i=0), W[0,1]=1 (k=0,i=1), W[1,1]=-1 (k=1,i=1)
    
    # Hmm wait, W index is k*n+i = k*2+i.
    # W[0*2+0]=W[0], W[1*2+0]=W[2], W[0*2+1]=W[1], W[1*2+1]=W[3]
    
    # Let me be very careful with indices.
    # U index: i*2+j
    # V index: j*2+k  
    # W index: k*2+i
    
    # M1: A part = A[0,0] + A[1,1] => U[0*2+0]=1, U[1*2+1]=1
    #      B part = B[0,0] + B[1,1] => V[0*2+0]=1, V[1*2+1]=1
    #      C: C[0,0]+=M1, C[0,1]+=M1, C[1,0]+=M1, C[1,1]-=M1
    #      W[0*2+0]=1, W[1*2+0]=1, W[0*2+1]=1, W[1*2+1]=-1
    
    # This is getting very tedious to do by hand for all blocks. Let me just implement it programmatically.
    
    # I'll build the decomposition by composing smaller decompositions.
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2, each shape (4, 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1  # A[0,0], A[1,1]
        V[0,0] = 1; V[3,0] = 1  # B[0,0], B[1,1]
        W[0,0] = 1; W[2,0] = 1; W[1,0] = 1; W[3,0] = -1  # C[0,0]+=, C[1,0]+=, C[0,1]+=, C[1,1]-=
        # W index: k*2+i. C[0,0]: k=0,i=0 => W[0]. C[1,0]: k=1,i=0 => W[2]. C[0,1]: k=0,i=1 => W[1]. C[1,1]: k=1,i=1 => W[3].
        
        # Term 1: M2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1  # A[1,0], A[1,1]
        V[0,1] = 1  # B[0,0]
        W[2,1] = 1; W[3,1] = -1  # C[1,0]+=, C[1,1]-=
        
        # Term 2: M3 = A00(B10-B11)
        U[0,2] = 1  # A[0,0]
        V[2,2] = 1; V[3,2] = -1  # B[1,0], B[1,1]
        W[2,2] = 1; W[0,2] = 1  # C[1,0]+=, C[0,0]+=
        # Wait: C00 = M1 - M5 + M3 + M7, C10 = M2 + M6. M3 only in C00.
        # Let me recheck: C00 = M1 - M5 + M3 + M7. So M3 contributes +1 to C00.
        # C00: k=0, i=0 => W[0]. So W[0,2] = 1.
        
        # Let me redo this more carefully.
        # C00 = M1 - M5 + M3 + M7
        # C01 = M4 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M4 + M6
        
        # Reset and redo:
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1   # C00 += M1
        W[1,0]=0   # C01: no M1
        W[2,0]=1   # C10 += M1? No! C10 = M2+M6, no M1.
        
        # I'm confusing myself. Let me just look up the standard Strassen tensor decomposition.
        
        # Standard Strassen for 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        #
        # c00 = M1 - M5 + M3 + M7
        # c01 = M4 + M5
        # c10 = M2 + M6
        # c11 = M1 - M2 + M4 + M6
        
        # In tensor form, for each Ms, we have U_s, V_s, W_s vectors:
        # U_s[i*2+j] = coefficient of A[i,j] in the "A part" of Ms
        # V_s[j*2+k] = coefficient of B[j,k] in the "B part" of Ms
        # W_s[k*2+i] = coefficient of Ms in C[i,k]
        
        # M1: A part = A00 + A11, B part = B00 + B11
        # C: c00+=M1, c11+=M1 (from c11 = M1 - M2 + M4 + M6)
        # Wait, c00 = M1 - M5 + M3 + M7, so M1 contributes +1 to c00.
        # c11 = M1 - M2 + M4 + M6, so M1 contributes +1 to c11.
        # c01, c10: no M1.
        
        U[0,0]=1; U[3,0]=1  # A[0,0], A[1,1]
        V[0,0]=1; V[3,0]=1  # B[0,0], B[1,1]
        W[0,0]=1   # c00: k=0,i=0
        W[3,0]=1   # c11: k=1,i=1
        # W[1,0]=0, W[2,0]=0
        
        # M2: A part = A10 + A11, B part = B00
        # C: c10+=M2, c11-=M2
        U[2,1]=1; U[3,1]=1  # A[1,0], A[1,1]
        V[0,1]=1  # B[0,0]
        W[2,1]=1   # c10: k=1,i=0 => W[1*2+0]=W[2]
        W[3,1]=-1  # c11: k=1,i=1 => W[1*2+1]=W[3]
        
        # M3: A part = A00, B part = B10 - B11
        # C: c00+=M3
        U[0,2]=1  # A[0,0]
        V[2,2]=1; V[3,2]=-1  # B[1,0], B[1,1]
        W[0,2]=1   # c00: k=0,i=0
        
        # M4: A part = A11, B part = B01 - B10
        # C: c01+=M4, c11+=M4
        U[3,3]=1  # A[1,1]
        V[1,3]=1; V[2,3]=-1  # B[0,1], B[1,0]
        W[1,3]=1   # c01: k=1,i=0 => W[1]
        W[3,3]=1   # c11: k=1,i=1 => W[3]
        
        # M5: A part = A00 + A01, B part = B11
        # C: c00-=M5, c01+=M5
        U[0,4]=1; U[1,4]=1  # A[0,0], A[0,1]
        V[3,4]=1  # B[1,1]
        W[0,4]=-1  # c00: k=0,i=0
        W[1,4]=1   # c01: k=1,i=0
        
        # M6: A part = A10 - A00, B part = B00 + B10
        # C: c10+=M6, c11+=M6
        U[2,5]=1; U[0,5]=-1  # A[1,0], A[0,0]
        V[0,5]=1; V[2,5]=1  # B[0,0], B[1,0]
        W[2,5]=1   # c10: k=1,i=0
        W[3,5]=1   # c11: k=1,i=1
        
        # M7: A part = A01 - A11, B part = B01 + B11
        # C: c00+=M7
        U[1,6]=1; U[3,6]=-1  # A[0,1], A[1,1]
        V[1,6]=1; V[3,6]=1  # B[0,1], B[1,1]
        W[0,6]=1   # c00: k=0,i=0
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    # Check: T[i,j,k] should be 1 for all i,j,k in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for s in range(7):
                    val += U_s[i*2+j, s] * V_s[j*2+k, s] * W_s[k*2+i, s]
                assert abs(val - 1) < 1e-10, f"Strassen check failed at i={i},j={j},k={k}, val={val}"
    
    # Now build 2x5x5 decomposition using 2+2+1 split
    def build_2x5x5():
        """Build rank-46 decomposition for 2x5 x 5x5 = 2x5."""
        n, m, p = 2, 5, 5
        # Split middle dim: 5 = 2+2+1
        # A (2x5) = [A0(2x2), A1(2x2), A2(2x1)]
        # B (5x5) split into 9 blocks
        # C (2x5) cols 0-1: A0*B00 + A1*B10 + A2*B20
        # C cols 2-3: A0*B01 + A1*B11 + A2*B21
        # C col 4: A0*B02 + A1*B12 + A2*B22
        
        terms = []  # list of (U_dict, V_dict, W_dict)
        
        def add_strassen_block(a_offset, b_row_offset, b_col_offset, c_col_offset):
            """Add a 2x2x2 Strassen block.
            a_offset: row offset in A (0 or 1 for the 2 rows)
            b_row_offset: row offset in B (0,2,4)
            b_col_offset: col offset in B (0,2,4)
            c_col_offset: col offset in C (0,2,4)
            """
            Us, Vs, Ws = strassen_2x2x2()
            for s in range(7):
                U_d = {}
                V_d = {}
                W_d = {}
                # U index in 2x2x2: i*2+j, maps to A row i, col b_row_offset+j
                # In full 2x5x5: U index = i*5 + (b_row_offset+j)
                for idx in range(4):
                    if Us[idx, s] != 0:
                        ii, jj = divmod(idx, 2)
                        full_idx = ii*5 + (b_row_offset + jj)
                        U_d[full_idx] = Us[idx, s]
                # V index in 2x2x2: j*2+k, maps to B row b_row_offset+j, col b_col_offset+k
                # In full: V index = (b_row_offset+j)*5 + (b_col_offset+k)
                for idx in range(4):
                    if Vs[idx, s] != 0:
                        jj, kk = divmod(idx, 2)
                        full_idx = (b_row_offset+jj)*5 + (b_col_offset+kk)
                        V_d[full_idx] = Vs[idx, s]
                # W index in 2x2x2: k*2+i, maps to C row i, col c_col_offset+k
                # In full 2x5x5: W index = k*2+i where k is col in C, i is row
                # Wait, for 2x5x5, W index = k*n+i = k*2+i
                for idx in range(4):
                    if Ws[idx, s] != 0:
                        kk, ii = divmod(idx, 2)
                        full_idx = (c_col_offset+kk)*2 + ii
                        W_d[full_idx] = Ws[idx, s]
                terms.append((U_d, V_d, W_d))
        
        def add_simple_block(a_cols, b_rows, b_cols, c_cols):
            """Add schoolbook terms for a small block.
            a_cols: list of A column indices
            b_rows: list of B row indices  
            b_cols: list of B col indices
            c_cols: list of C col indices
            For each (j in a_cols/b_rows, k in b_cols/c_cols): one term
            """
            for j_idx, j in enumerate(a_cols):
                for k_idx, k in enumerate(b_cols):
                    # B row is b_rows[j_idx], B col is k
                    U_d = {}
                    V_d = {}
                    W_d = {}
                    for i in range(2):
                        U_d[i*5 + j] = 1
                    V_d[b_rows[j_idx]*5 + k] = 1
                    W_d[k*2 + 0] = 1  # C[0,k]
                    W_d[k*2 + 1] = 1  # C[1,k]
                    terms.append((U_d, V_d, W_d))
        
        # C cols 0-1: A0*B00 + A1*B10 + A2*B20
        # A0 is cols 0-1, B00 is rows 0-1, cols 0-1
        add_strassen_block(0, 0, 0, 0)  # A0*B00: 2x2x2
        # A1 is cols 2-3, B10 is rows 2-3, cols 0-1
        add_strassen_block(0, 2, 0, 0)  # A1*B10: 2x2x2
        # A2 is col 4, B20 is row 4, cols 0-1: 2x1x1x2
        # Schoolbook: 2*1*2 = 4 terms
        for k in range(2):
            U_d = {0*5+4: 1, 1*5+4: 1}
            V_d = {4*5+k: 1}
            W_d = {k*2+0: 1, k*2+1: 1}
            terms.append((U_d, V_d, W_d))
        
        # C cols 2-3: A0*B01 + A1*B11 + A2*B21
        add_strassen_block(0, 0, 2, 2)  # A0*B01
        add_strassen_block(0, 2, 2, 2)  # A1*B11
        for k in range(2,4):
            U_d = {0*5+4: 1, 1*5+4: 1}
            V_d = {4*5+k: 1}
            W_d = {k*2+0: 1, k*2+1: 1}
            terms.append((U_d, V_d, W_d))
        
        # C col 4: A0*B02 + A1*B12 + A2*B22
        # A0*B02: 2x2x2x1, schoolbook 2*2*1=4
        for j in range(2):
            U_d = {0*5+j: 1, 1*5+j: 1}
            V_d = {j*5+4: 1}
            W_d = {4*2+0: 1, 4*2+1: 1}
            terms.append((U_d, V_d, W_d))
        # A1*B12: 2x2x2x1
        for j in range(2):
            U_d = {0*5+(2+j): 1, 1*5+(2+j): 1}
            V_d = {(2+j)*5+4: 1}
            W_d = {4*2+0: 1, 4*2+1: 1}
            terms.append((U_d, V_d, W_d))
        # A2*B22: 2x1x1x1
        U_d = {0*5+4: 1, 1*5+4: 1}
        V_d = {4*5+4: 1}
        W_d = {4*2+0: 1, 4*2+1: 1}
        terms.append((U_d, V_d, W_d))
        
        return terms
    
    # Build the full 4x5x5 by combining two 2x5x5 halves
    # Top half: rows 0-1 of A, produces rows 0-1 of C
    # Bottom half: rows 2-3 of A, produces rows 2-3 of C
    
    all_terms = []
    
    # Top half (rows 0-1)
    top_terms = build_2x5x5()
    for U_d, V_d, W_d in top_terms:
        # Remap: A rows 0-1 stay as is (indices 0-4 for cols)
        # V stays the same
        # W: C rows 0-1, so W index k*2+i stays the same (i in 0,1)
        all_terms.append((U_d, V_d, W_d))
    
    # Bottom half (rows 2-3)
    bot_terms = build_2x5x5()
    for U_d, V_d, W_d in bot_terms:
        new_U_d = {}
        new_W_d = {}
        for idx, val in U_d.items():
            # idx = i*5+j where i in {0,1} for the 2x5x5 subproblem
            # In full 4x5x5, these are rows 2,3, so new idx = (i+2)*5+j
            i, j = divmod(idx, 5)
            new_U_d[(i+2)*5 + j] = val
        for idx, val in W_d.items():
            # idx = k*2+i where i in {0,1}
            # In full, C rows are 2,3, so new i = i+2
            k, i = divmod(idx, 2)
            new_W_d[k*4 + (i+2)] = val
        all_terms.append((U_d and new_U_d, V_d, new_W_d))
    
    rank = len(all_terms)
    
    # Convert to matrices
    U = [[0]*rank for _ in range(20)]  # 4*5
    V = [[0]*rank for _ in range(25)]  # 5*5
    W = [[0]*rank for _ in range(20)]  # 5*4
    
    for s, (U_d, V_d, W_d) in enumerate(all_terms):
        for idx, val in U_d.items():
            U[idx][s] = val
        for idx, val in V_d.items():
            V[idx][s] = val
        for idx, val in W_d.items():
            W[idx][s] = val
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(4):
        for j in range(5):
            for k in range(5):
                val = 0
                for s in range(rank):
                    val += U_arr[i*5+j, s] * V_arr[j*5+k, s] * W_arr[k*4+i, s]
                if abs(val - 1) > 1e-9:
                    print(f"FAIL: i={i},j={j},k={k}, val={val}")
                    break
    
    return {"U": U, "V": V, "W": W}
