DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def set_term(col, u_entries, v_entries, w_entries):
        """u_entries: list of (idx, coeff), etc."""
        for idx, coeff in u_entries:
            U[idx][col] = coeff
        for idx, coeff in v_entries:
            V[idx][col] = coeff
        for idx, coeff in w_entries:
            W[idx][col] = coeff
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                a = 2*bi
                b = 2*bi+1
                c_idx = 4+2*bi
                d = 4+2*bi+1
                e = 2*bj*6 + 3*bk
                f = 2*bj*6 + 3*bk+1
                g = 2*bj*6 + 3*bk+2
                h = (2*bj+1)*6 + 3*bk
                i = (2*bj+1)*6 + 3*bk+1
                j = (2*bj+1)*6 + 3*bk+2
                c00 = 3*bk*2
                c01 = (3*bk+1)*2
                c02 = (3*bk+2)*2
                c10 = 3*bk*2+1
                c11 = (3*bk+1)*2+1
                c12 = (3*bk+2)*2+1
                
                # P1=(a+b)(e+i)
                set_term(col, [(a,1),(b,1)], [(e,1),(i,1)], [(c00,1),(c01,1),(c10,1),(c11,1)])
                col+=1
                # P2=(c+d)(e+i)
                set_term(col, [(c_idx,1),(d,1)], [(e,1),(i,1)], [(c00,1),(c01,1),(c10,1),(c11,1)])
                col+=1
                # P3=a(f-h)
                set_term(col, [(a,1)], [(f,1),(h,-1)], [(c00,1),(c01,1)])
                col+=1
                # P4=d(i-e)
                set_term(col, [(d,1)], [(i,1),(e,-1)], [(c00,1),(c10,1)])
                col+=1
                # P5=(a+d)(e+i)
                set_term(col, [(a,1),(d,1)], [(e,1),(i,1)], [(c00,1),(c01,1),(c10,1),(c11,1)])
                col+=1
                # P6=(b+c)(e+i)
                set_term(col, [(b,1),(c_idx,1)], [(e,1),(i,1)], [(c00,1),(c01,1),(c10,1),(c11,1)])
                col+=1
                # P7=(b+d)(f-h)
                set_term(col, [(b,1),(d,1)], [(f,1),(h,-1)], [(c00,1),(c01,1),(c10,1),(c11,1)])
                col+=1
                # P8=a*g
                set_term(col, [(a,1)], [(g,1)], [(c02,1)])
                col+=1
                # P9=c*j
                set_term(col, [(c_idx,1)], [(j,1)], [(c02,1)])
                col+=1
                # P10=b*g
                set_term(col, [(b,1)], [(g,1)], [(c12,1)])
                col+=1
                # P11=d*j
                set_term(col, [(d,1)], [(j,1)], [(c12,1)])
                col+=1
    
    return {"U": U, "V": V, "W": W}
