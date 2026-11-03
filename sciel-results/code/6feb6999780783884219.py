import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_2x2x2(i0, j0, k0):
        """
        Standard Strassen 7-term decomposition for 2x2x2.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
        C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        M1 = (a+d)(e+h)
        M2 = (c+d)(e)
        M3 = (a)(g-e)
        M4 = (d)(f-h)
        M5 = (a+b)(h)
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C00 = M1 - M4 + M5 - M7
        C01 = M4 + M5
        C10 = M2 + M3
        C11 = M1 - M3 + M6 - M5
        """
        i1 = i0+1; j1 = j0+1; k1 = k0+1
        # M1 = (A00+A11)(B00+B11)
        add_rank1_term([i0*m+j0, i1*m+j1], [1,1],
                       [j0*p+k0, j1*p+k1], [1,1],
                       [k0*n+i0, k1*n+i1], [1,1])
        # M2 = (A10+A11)(B00)
        add_rank1_term([i1*m+j0, i1*m+j1], [1,1],
                       [j0*p+k0], [1],
                       [k0*n+i1], [1])
        # M3 = (A00)(B10-B00)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j0*p+k0], [1,-1],
                       [k1*n+i1], [1])
        # M4 = (A11)(B01-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j1*p+k1], [1,-1],
                       [k0*n+i0], [1])
        # M5 = (A00+A01)(B11)
        add_rank1_term([i0*m+j0, i0*m+j1], [1,1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i1], [-1,1])
        # M6 = (A10-A00)(B00+B01)
        add_rank1_term([i1*m+j0, i0*m+j0], [1,-1],
                       [j0*p+k0, j1*p+k0], [1,1],
                       [k1*n+i1], [1])
        # M7 = (A01-A11)(B10+B11)
        add_rank1_term([i0*m+j1, i1*m+j1], [1,-1],
                       [j1*p+k0, j1*p+k1], [1,1],
                       [k0*n+i0, k1*n+i0], [1,-1])

    def schoolbook(n_l, m_l, p_l, i_off, j_off, k_off):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+ii], [1.0])

    # 3x4x7 = 3x4x3 + 3x4x4
    # 3x4x3 = 2x4x3 + 1x4x3
    #   2x4x3 = 2x2x3 + 2x2x3
    #     2x2x3 = Strassen(2x2x2) + schoolbook(2x2x1) = 7 + 4 = 11
    #   1x4x3 = schoolbook = 12
    #   Total 3x4x3 = 22 + 12 = 34
    # 3x4x4 = 3x2x4 + 3x2x4
    #   3x2x4 = 2x2x4 + 1x2x4
    #     2x2x4 = Strassen(2x2x2) + Strassen(2x2x2) = 14
    #     1x2x4 = schoolbook = 8
    #     Total 3x2x4 = 22
    #   Total 3x4x4 = 44
    # Grand total = 78

    # 3x4x3 part (k=0,1,2)
    # 2x4x3 (rows 0,1; cols 0-3; k=0,1,2)
    #   2x2x3 (rows 0,1; cols 0,1; k=0,1,2)
    strassen_2x2x2(0, 0, 0)
    schoolbook(2, 2, 1, 0, 0, 2)
    #   2x2x3 (rows 0,1; cols 2,3; k=0,1,2)
    strassen_2x2x2(0, 2, 0)
    schoolbook(2, 2, 1, 0, 2, 2)
    # 1x4x3 (row 2; cols 0-3; k=0,1,2)
    schoolbook(1, 4, 3, 2, 0, 0)

    # 3x4x4 part (k=3,4,5,6)
    # 3x2x4 (rows 0-2; cols 0,1; k=3,4,5,6)
    #   2x2x4 (rows 0,1; cols 0,1; k=3,4,5,6)
    strassen_2x2x2(0, 0, 3)
    strassen_2x2x2(0, 0, 5)
    #   1x2x4 (row 2; cols 0,1; k=3,4,5,6)
    schoolbook(1, 2, 4, 2, 0, 3)
    # 3x2x4 (rows 0-2; cols 2,3; k=3,4,5,6)
    #   2x2x4 (rows 0,1; cols 2,3; k=3,4,5,6)
    strassen_2x2x2(0, 2, 3)
    strassen_2x2x2(0, 2, 5)
    #   1x2x4 (row 2; cols 2,3; k=3,4,5,6)
    schoolbook(1, 2, 4, 2, 2, 3)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
