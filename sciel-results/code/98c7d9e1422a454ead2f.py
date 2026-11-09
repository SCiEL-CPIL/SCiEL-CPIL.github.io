DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    for i_b in range(3):
        for j_b in range(3):
            for k_b in range(3):
                i_sz = 2 if i_b < 2 else 1
                k_sz = 2 if k_b < 2 else 1
                j_sz = 2 if j_b < 2 else 1
                
                if i_sz == 2 and k_sz == 2 and j_sz == 2:
                    _strassen(i_b, k_b, j_b, add_term)
                else:
                    _schoolbook(i_b, k_b, j_b, add_term)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_e, v_e, w_e) in enumerate(terms):
        for row, val in u_e:
            U[row][c] = val
        for row, val in v_e:
            V[row][c] = val
        for row, val in w_e:
            W[row][c] = val
    
    return {"U": U, "V": V, "W": W}

def _strassen(i_b, k_b, j_b, add_term):
    i0, i1 = i_b * 2, i_b * 2 + 1
    k0, k1 = k_b * 2, k_b * 2 + 1
    j0, j1 = j_b * 2, j_b * 2 + 1
    
    # m1 = (A[i0,k0]+A[i1,k1]) * (B[k0,j0]+B[k1,j1])
    # C[i0,j0] += m1, C[i1,j1] += m1
    add_term(
        [(i0*5+k0, 1), (i1*5+k1, 1)],
        [(k0*5+j0, 1), (k1*5+j1, 1)],
        [(j0*5+i0, 1), (j1*5+i1, 1)]
    )
    
    # m2 = (A[i1,k0]+A[i1,k1]) * B[k0,j0]
    # C[i1,j0] += m2, C[i1,j1] -= m2
    add_term(
        [(i1*5+k0, 1), (i1*5+k1, 1)],
        [(k0*5+j0, 1)],
        [(j0*5+i1, 1), (j1*5+i1, -1)]
    )
    
    # m3 = A[i0,k0] * (B[k0,j1]-B[k1,j1])
    # C[i0,j1] += m3, C[i1,j1] += m3
    add_term(
        [(i0*5+k0, 1)],
        [(k0*5+j1, 1), (k1*5+j1, -1)],
        [(j1*5+i0, 1), (j1*5+i1, 1)]
    )
    
    # m4 = A[i1,k1] * (B[k1,j0]-B[k0,j0])
    # C[i0,j0] += m4, C[i1,j0] += m4
    add_term(
        [(i1*5+k1, 1)],
        [(k1*5+j0, 1), (k0*5+j0, -1)],
        [(j0*5+i0, 1), (j0*5+i1, 1)]
    )
    
    # m5 = (A[i0,k0]+A[i0,k1]) * B[k1,j1]
    # C[i0,j0] -= m5, C[i0,j1] += m5
    add_term(
        [(i0*5+k0, 1), (i0*5+k1, 1)],
        [(k1*5+j1, 1)],
        [(j0*5+i0, -1), (j1*5+i0, 1)]
    )
    
    # m6 = (A[i1,k0]-A[i0,k0]) * (B[k0,j0]+B[k0,j1])
    # C[i1,j1] += m6
    add_term(
        [(i1*5+k0, 1), (i0*5+k0, -1)],
        [(k0*5+j0, 1), (k0*5+j1, 1)],
        [(j1*5+i1, 1)]
    )
    
    # m7 = (A[i0,k1]-A[i1,k1]) * (B[k1,j0]+B[k1,j1])
    # C[i0,j0] += m7, C[i1,j1] -= m7
    add_term(
        [(i0*5+k1, 1), (i1*5+k1, -1)],
        [(k1*5+j0, 1), (k1*5+j1, 1)],
        [(j0*5+i0, 1), (j1*5+i1, -1)]
    )

def _schoolbook(i_b, k_b, j_b, add_term):
    i_rows = [i_b*2, i_b*2+1] if i_b < 2 else [4]
    k_rows = [k_b*2, k_b*2+1] if k_b < 2 else [4]
    j_cols = [j_b*2, j_b*2+1] if j_b < 2 else [4]
    
    for i in i_rows:
        for k in k_rows:
            for j in j_cols:
                add_term(
                    [(i*5+k, 1)],
                    [(k*5+j, 1)],
                    [(j*5+i, 1)]
                )
