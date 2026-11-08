import numpy as np
import time

def construct():
    n, m, p = 5, 5, 5
    t0 = time.time()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for (i, j), val in u_dict.items():
            u[i*m + j] = val
        for (j, k), val in v_dict.items():
            v[j*p + k] = val
        for (k, i), val in w_dict.items():
            w[k*n + i] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
        add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
        add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
        add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})
    
    def schoolbook(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    def block_3x3x3(i0, j0, k0):
        """3x3x3 using 2+1 split with Strassen for 2x2x2"""
        i1, i2 = i0+1, i0+2
        j1, j2 = j0+1, j0+2
        k1, k2 = k0+1, k0+2
        
        # C22 = A22*B22 + A21*B12
        strassen_2x2(i0, i1, j0, j1, k0, k1)  # A22*B22: 7
        schoolbook([i0,i1], [j2], [k0,k1])  # A21*B12: 2
        
        # C23 = A22*B23 + A21*B13
        schoolbook([i0,i1], [j0,j1], [k2])  # A22*B23: 4
        schoolbook([i0,i1], [j2], [k2])  # A21*B13: 2
        
        # C32 = A12*B22 + A11*B32
        schoolbook([i2], [j0,j1], [k0,k1])  # A12*B22: 4
        schoolbook([i2], [j2], [k0,k1])  # A11*B32: 2
        
        # C33 = A12*B23 + A11*B33
        schoolbook([i2], [j0,j1], [k2])  # A12*B23: 2
        schoolbook([i2], [j2], [k2])  # A11*B33: 1
        # Total: 7+2+4+2+4+2+2+1 = 24
    
    # 2+3 split for 5x5x5
    # C22 = A22*B22 + A23*B32
    strassen_2x2(0, 1, 0, 1, 0, 1)  # 7
    schoolbook([0,1], [2,3,4], [0,1])  # 6
    # C22: 13
    
    # C23 = A22*B23 + A23*B33
    strassen_2x2(0, 1, 0, 1, 2, 3)  # A22*B23a: 7
    schoolbook([0,1], [0,1], [4])  # A22*B23b: 4
    # A23*B33: 2x3 * 3x3. Use 2+1 split on the 3x3 part of B and 2x2+2x1 on A
    # A23a(2x2, j=2,3) * B33(3x3): use block approach
    # A23a*B33a(2x2, j=2,3;k=2,3): Strassen
    strassen_2x2(0, 1, 2, 3, 2, 3)  # 7
    # A23a*B33b(2x1, j=2,3;k=4): schoolbook
    schoolbook([0,1], [2,3], [4])  # 4
    # A23b(2x1, j=4) * B33(3x3): schoolbook
    schoolbook([0,1], [4], [2,3,4])  # 6
    # C23: 7+4+7+4+6 = 28
    
    # C32 = A32*B22 + A33*B32
    strassen_2x2(2, 3, 0, 1, 0, 1)  # A32a*B22: 7
    schoolbook([4], [0,1], [0,1])  # A32b*B22: 4
    # A33*B32: 3x3 * 3x2
    # A33a(2x2, i=2,3;j=2,3) * B32a(2x2, j=2,3;k=0,1): Strassen
    strassen_2x2(2, 3, 2, 3, 0, 1)  # 7
    # A33a * B32b(1x2, j=4;k=0,1): schoolbook
    schoolbook([2,3], [4], [0,1])  # 4
    # A33b(2x1, i=2,3;j=4) * B32a(2x2, j=2,3;k=0,1): schoolbook
    schoolbook([2,3], [4], [0,1])  # 4 -- wait, this is same as above?
    
    # I'm double counting. Let me be more careful.
    # A33 is 3x3 (i=2,3,4; j=2,3,4), B32 is 3x2 (j=2,3,4; k=0,1)
    # Split by j: j={2,3} and j={4}
    # A33a (i=2,3,4; j=2,3) * B32a (j=2,3; k=0,1): 3x2 * 2x2
    #   Split i: i={2,3} and i={4}
    #   A33a1 (i=2,3; j=2,3) * B32a: 2x2 * 2x2 = Strassen (7)
    #   A33a2 (i=4; j=2,3) * B32a: 1x2 * 2x2 = schoolbook (4)
    # A33b (i=2,3,4; j=4) * B32b (j=4; k=0,1): 3x1 * 1x2 = schoolbook (6)
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # C22 = A22*B22 + A23*B32
    strassen_2x2(0, 1, 0, 1, 0, 1)  # 7
    schoolbook([0,1], [2,3,4], [0,1])  # 6
    # C22: 13
    
    # C23 = A22*B23 + A23*B33
    strassen_2x2(0, 1, 0, 1, 2, 3)  # 7
    schoolbook([0,1], [0,1], [4])  # 4
    strassen_2x2(0, 1, 2, 3, 2, 3)  # 7
    schoolbook([0,1], [2,3], [4])  # 4
    schoolbook([0,1], [4], [2,3,4])  # 6
    # C23: 28
    
    # C32 = A32*B22 + A33*B32
    strassen_2x2(2, 3, 0, 1, 0, 1)  # 7
    schoolbook([4], [0,1], [0,1])  # 4
    strassen_2x2(2, 3, 2, 3, 0, 1)  # 7
    schoolbook([4], [2,3], [0,1])  # 4
    schoolbook([2,3,4], [4], [0,1])  # 6
    # C32: 28
    
    # C33 = A32*B23 + A33*B33
    strassen_2x2(2, 3, 0, 1, 2, 3)  # 7
    schoolbook([2,3], [0,1], [4])  # 4
    schoolbook([4], [0,1], [2,3])  # 4
    schoolbook([4], [0,1], [4])  # 2
    block_3x3x3(2, 2, 2)  # 24
    # C33: 41
    
    # Total: 13+28+28+41 = 110
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    if time.time() - t0 > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
