import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is sum of ranks of parts
    # Part 1: 2x2x5 -> 7+7+4 = 18
    # Part 2: 2x3x5 -> 11+11+6 = 28
    # Total 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # Helper to add Strassen 2x2x2
    def add_strassen(i_off, j_off, k_off):
        # i_off, j_off, k_off are global offsets for the 2x2x2 block
        # Local indices 0,1 map to global i_off+0, i_off+1 etc.
        
        # M1
        # (A11+A22)(B11+B22)
        # A11: i=0, j=0. A22: i=1, j=1.
        # B11: j=0, k=0. B22: j=1, k=1.
        # Term 1: U[0,0]=1, U[1,1]=1. V[0,0]=1, V[1,1]=1. W[0,0]=1, W[1,1]=1.
        # This is not a single rank-1 term. Strassen uses 7 rank-1 terms.
        
        # We need the explicit rank-1 terms for Strassen.
        # M1 = (A11+A22)(B11+B22)
        # A11+A22 involves A[0,0] and A[1,1].
        # B11+B22 involves B[0,0] and B[1,1].
        # The product is (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]).
        # This expands to A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1].
        # In the tensor decomposition, each term is $u_i v_j w_k$.
        # We need to represent the multiplication as sum of rank-1 tensors.
        
        # Strassen Rank 1 Terms:
        # 1. (A11+A22)(B11+B22)
        #    U: A11=1, A22=1. V: B11=1, B22=1. W: C11+=1, C22+=1? No.
        #    The term contributes to C11 and C22.
        #    Actually, the standard Strassen decomposition is:
        #    M1 = (A11+A22)(B11+B22)
        #    M2 = (A21+A22)B11
        #    M3 = A11(B12-B22)
        #    M4 = A22(B21-B11)
        #    M5 = (A11+A12)B22
        #    M6 = (A21-A11)(B11+B12)
        #    M7 = (A12-A22)(B21+B22)
        
        #    C11 = M1 + M4 - M5 + M7
        #    C12 = M3 + M5
        #    C21 = M2 + M4
        #    C22 = M1 - M2 + M3 + M6
        
        #    Each Mt is a scalar product of a linear combination of A and a linear combination of B.
        #    So each Mt corresponds to a rank-1 term in the tensor?
        #    Yes, $M_t = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
        #    This is a rank-1 tensor in the space of A, B, C?
        #    The contribution to C is $M_t \times 1$? No.
        #    $C_{ik} += M_t \times w_{ik}$.
        #    So the rank-1 term is $U_t \otimes V_t \otimes W_t$.
        #    $U_t[i,j] = u_{ij}$, $V_t[j,k] = v_{jk}$, $W_t[k,i] = w_{ik}$.
        
        #    For M1:
        #    U: A11=1, A22=1. (i=0,j=0; i=1,j=1)
        #    V: B11=1, B22=1. (j=0,k=0; j=1,k=1)
        #    W: C11=1, C22=1. (k=0,i=0; k=1,i=1)
        #    Wait, $C_{ik}$ index is $k \times n + i$.
        #    C11 is k=0, i=0. Index 0.
        #    C22 is k=1, i=1. Index 1*2+1=3.
        
        #    So for M1:
        #    U[0*m+0] = 1
        #    U[1*m+1] = 1
        #    V[0*p+0] = 1
        #    V[1*p+1] = 1
        #    W[0*n+0] = 1
        #    W[1*n+1] = 1
        
        #    This adds 1 to rank.
        
        #    We need to map local i,j,k to global i,j,k.
        
        #    Local i=0 -> Global i = i_off
        #    Local i=1 -> Global i = i_off + 1
        #    Local j=0 -> Global j = j_off
        #    Local j=1 -> Global j = j_off + 1
        #    Local k=0 -> Global k = k_off
        #    Local k=1 -> Global k = k_off + 1
        
        #    M1:
        #    U: (i_off, j_off)=1, (i_off+1, j_off+1)=1
        #    V: (j_off, k_off)=1, (j_off+1, k_off+1)=1
        #    W: (k_off, i_off)=1, (k_off+1, i_off+1)=1
        
        #    M2: (A21+A22)B11
        #    U: (i_off+1, j_off)=1, (i_off+1, j_off+1)=1
        #    V: (j_off, k_off)=1
        #    W: (k_off, i_off+1)=1
        
        #    M3: A11(B12-B22)
        #    U: (i_off, j_off)=1
        #    V: (j_off, k_off+1)=1, (j_off+1, k_off+1)=-1
        #    W: (k_off+1, i_off)=1
        
        #    M4: A22(B21-B11)
        #    U: (i_off+1, j_off+1)=1
        #    V: (j_off+1, k_off)=1, (j_off, k_off)=-1
        #    W: (k_off, i_off+1)=1
        
        #    M5: (A11+A12)B22
        #    U: (i_off, j_off)=1, (i_off, j_off+1)=1
        #    V: (j_off+1, k_off+1)=1
        #    W: (k_off+1, i_off)=1
        
        #    M6: (A21-A11)(B11+B12)
        #    U: (i_off+1, j_off)=1, (i_off, j_off)=-1
        #    V: (j_off, k_off)=1, (j_off, k_off+1)=1
        #    W: (k_off+1, i_off+1)=1
        
        #    M7: (A12-A22)(B21+B22)
        #    U: (i_off, j_off+1)=1, (i_off+1, j_off+1)=-1
        #    V: (j_off+1, k_off)=1, (j_off+1, k_off+1)=1
        #    W: (k_off, i_off)=1
        
        terms = [
            # M1
            { 'U': [(i_off, j_off, 1), (i_off+1, j_off+1, 1)],
              'V': [(j_off, k_off, 1), (j_off+1, k_off+1, 1)],
              'W': [(k_off, i_off, 1), (k_off+1, i_off+1, 1)] },
            # M2
            { 'U': [(i_off+1, j_off, 1), (i_off+1, j_off+1, 1)],
              'V': [(j_off, k_off, 1)],
              'W': [(k_off, i_off+1, 1)] },
            # M3
            { 'U': [(i_off, j_off, 1)],
              'V': [(j_off, k_off+1, 1), (j_off+1, k_off+1, -1)],
              'W': [(k_off+1, i_off, 1)] },
            # M4
            { 'U': [(i_off+1, j_off+1, 1)],
              'V': [(j_off+1, k_off, 1), (j_off, k_off, -1)],
              'W': [(k_off, i_off+1, 1)] },
            # M5
            { 'U': [(i_off, j_off, 1), (i_off, j_off+1, 1)],
              'V': [(j_off+1, k_off+1, 1)],
              'W': [(k_off+1, i_off, 1)] },
            # M6
            { 'U': [(i_off+1, j_off, 1), (i_off, j_off, -1)],
              'V': [(j_off, k_off, 1), (j_off, k_off+1, 1)],
              'W': [(k_off+1, i_off+1, 1)] },
            # M7
            { 'U': [(i_off, j_off+1, 1), (i_off+1, j_off+1, -1)],
              'V': [(j_off+1, k_off, 1), (j_off+1, k_off+1, 1)],
              'W': [(k_off, i_off, 1)] },
        ]
        
        for t in terms:
            for i, j, val in t['U']:
                U[i * m + j][col] = val
            for j, k, val in t['V']:
                V[j * p + k][col] = val
            for k, i, val in t['W']:
                W[k * n + i][col] = val
            col += 1

    # Helper to add Schoolbook 2x2x1
    def add_schoolbook_221(i_off, j_off, k_off):
        # i=0,1; j=0,1; k=0
        # C10 = A10B00 + A11B10
        # C20 = A20B00 + A21B10
        # Terms:
        # 1. A10 B00 -> C10
        # 2. A11 B10 -> C10
        # 3. A20 B00 -> C20
        # 4. A21 B10 -> C20
        
        terms = [
            { 'U': [(i_off, j_off, 1)], 'V': [(j_off, k_off, 1)], 'W': [(k_off, i_off, 1)] },
            { 'U': [(i_off, j_off+1, 1)], 'V': [(j_off+1, k_off, 1)], 'W': [(k_off, i_off, 1)] },
            { 'U': [(i_off+1, j_off, 1)], 'V': [(j_off, k_off, 1)], 'W': [(k_off, i_off+1, 1)] },
            { 'U': [(i_off+1, j_off+1, 1)], 'V': [(j_off+1, k_off, 1)], 'W': [(k_off, i_off+1, 1)] },
        ]
        for t in terms:
            for i, j, val in t['U']:
                U[i * m + j][col] = val
            for j, k, val in t['V']:
                V[j * p + k][col] = val
            for k, i, val in t['W']:
                W[k * n + i][col] = val
            col += 1

    # Helper to add Schoolbook 2x1x2
    def add_schoolbook_212(i_off, j_off, k_off):
        # i=0,1; j=0; k=0,1
        # C10 = A10 B00
        # C11 = A10 B01
        # C20 = A20 B00
        # C21 = A20 B01
        
        terms = [
            { 'U': [(i_off, j_off, 1)], 'V': [(j_off, k_off, 1)], 'W': [(k_off, i_off, 1)] },
            { 'U': [(i_off, j_off, 1)], 'V': [(j_off, k_off+1, 1)], 'W': [(k_off+1, i_off, 1)] },
            { 'U': [(i_off+1, j_off, 1)], 'V': [(j_off, k_off, 1)], 'W': [(k_off, i_off+1, 1)] },
            { 'U': [(i_off+1, j_off, 1)], 'V': [(j_off, k_off+1, 1)], 'W': [(k_off+1, i_off+1, 1)] },
        ]
        for t in terms:
            for i, j, val in t['U']:
                U[i * m + j][col] = val
            for j, k, val in t['V']:
                V[j * p + k][col] = val
            for k, i, val in t['W']:
                W[k * n + i][col] = val
            col += 1

    # Helper to add Schoolbook 2x3x1
    def add_schoolbook_231(i_off, j_off, k_off):
        # i=0,1; j=0,1,2; k=0
        # C10 = A10B00 + A11B10 + A12B20
        # C20 = A20B00 + A21B10 + A22B20
        
        for jj in range(3):
            terms = [
                { 'U': [(i_off, j_off+jj, 1)], 'V': [(j_off+jj, k_off, 1)], 'W': [(k_off, i_off, 1)] },
                { 'U': [(i_off+1, j_off+jj, 1)], 'V': [(j_off+jj, k_off, 1)], 'W': [(k_off, i_off+1, 1)] },
            ]
            for t in terms:
                for i, j, val in t['U']:
                    U[i * m + j][col] = val
                for j, k, val in t['V']:
                    V[j * p + k][col] = val
                for k, i, val in t['W']:
                    W[k * n + i][col] = val
                col += 1

    # Part 1: A1 B1 (2x2x5)
    # i_off=0, j_off=0
    # k=0,1 -> Strassen
    add_strassen(0, 0, 0)
    # k=2,3 -> Strassen
    add_strassen(0, 0, 2)
    # k=4 -> Schoolbook 2x2x1
    add_schoolbook_221(0, 0, 4)

    # Part 2: A2 B2 (2x3x5)
    # i_off=0, j_off=2
    # k=0,1 -> 2x3x2
    # Split j=2,3,4 into j=2,3 (2x2x2) and j=4 (2x1x2)
    add_strassen(0, 2, 0) # j=2,3
    add_schoolbook_212(0, 4, 0) # j=4
    
    # k=2,3 -> 2x3x2
    add_strassen(0, 2, 2) # j=2,3
    add_schoolbook_212(0, 4, 2) # j=4
    
    # k=4 -> 2x3x1
    add_schoolbook_231(0, 2, 4)

    return {"U": U, "V": V, "W": W}
