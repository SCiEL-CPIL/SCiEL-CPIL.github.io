"""Matrix multiplication 3x4x7 with Strassen block decomposition."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 7)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for (ii, jj), val in u_entries:
            u_col[ii * m + jj] = val
        for (jj, kk), val in v_entries:
            v_col[jj * p + kk] = val
        for (kk, ii), val in w_entries:
            w_col[kk * n + ii] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    for (i0, i1) in i_blocks:
        for (j0, j1) in j_blocks:
            for (k0, k1) in k_blocks:
                si, sj, sk = i1-i0, j1-j0, k1-k0
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen for 2x2x2
                    # a=A[i0,j0], b=A[i0,j0+1], c=A[i0+1,j0], d=A[i0+1,j0+1]
                    # e=B[j0,k0], f=B[j0,k0+1], g=B[j0+1,k0], h=B[j0+1,k0+1]
                    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
                    # C[i0,k0]=M1+M4-M5+M7, C[i0,k0+1]=M3+M5, C[i0+1,k0]=M2+M4, C[i0+1,k0+1]=M1-M2+M3+M6
                    
                    # M1: U:{(i0,j0):1,(i0+1,j0+1):1}, V:{(j0,k0):1,(j0+1,k0+1):1}
                    #   W: C[i0,k0]+=1, C[i0+1,k0+1]+=1
                    add_term(
                        u_entries=[((i0,j0),1),((i0+1,j0+1),1)],
                        v_entries=[((j0,k0),1),((j0+1,k0+1),1)],
                        w_entries=[((k0,i0),1),((k0+1,i0+1),1)]
                    )
                    # M2: U:{(i0+1,j0):1,(i0+1,j0+1):1}, V:{(j0,k0):1}
                    #   W: C[i0+1,k0]+=1, C[i0+1,k0+1]+=-1
                    add_term(
                        u_entries=[((i0+1,j0),1),((i0+1,j0+1),1)],
                        v_entries=[((j0,k0),1)],
                        w_entries=[((k0,i0+1),1),((k0+1,i0+1),-1)]
                    )
                    # M3: U:{(i0,j0):1}, V:{(j0,k0+1):1,(j0+1,k0+1):-1}
                    #   W: C[i0,k0+1]+=1, C[i0+1,k0+1]+=1
                    add_term(
                        u_entries=[((i0,j0),1)],
                        v_entries=[((j0,k0+1),1),((j0+1,k0+1),-1)],
                        w_entries=[((k0+1,i0),1),((k0+1,i0+1),1)]
                    )
                    # M4: U:{(i0+1,j0+1):1}, V:{(j0+1,k0):1,(j0,k0):-1}
                    #   W: C[i0,k0]+=1, C[i0+1,k0]+=1
                    add_term(
                        u_entries=[((i0+1,j0+1),1)],
                        v_entries=[((j0+1,k0),1),((j0,k0),-1)],
                        w_entries=[((k0,i0),1),((k0,i0+1),1)]
                    )
                    # M5: U:{(i0,j0):1,(i0,j0+1):1}, V:{(j0+1,k0+1):1}
                    #   W: C[i0,k0]+=-1, C[i0,k0+1]+=1
                    add_term(
                        u_entries=[((i0,j0),1),((i0,j0+1),1)],
                        v_entries=[((j0+1,k0+1),1)],
                        w_entries=[((k0,i0),-1),((k0+1,i0),1)]
                    )
                    # M6: U:{(i0+1,j0):1,(i0,j0):-1}, V:{(j0,k0):1,(j0,k0+1):1}
                    #   W: C[i0+1,k0+1]+=1
                    add_term(
                        u_entries=[((i0+1,j0),1),((i0,j0),-1)],
                        v_entries=[((j0,k0),1),((j0,k0+1),1)],
                        w_entries=[((k0+1,i0+1),1)]
                    )
                    # M7: U:{(i0,j0+1):1,(i0+1,j0+1):-1}, V:{(j0+1,k0):1,(j0+1,k0+1):1}
                    #   W: C[i0,k0]+=1, C[i0+1,k0+1]+=-1
                    add_term(
                        u_entries=[((i0,j0+1),1),((i0+1,j0+1),-1)],
                        v_entries=[((j0+1,k0),1),((j0+1,k0+1),1)],
                        w_entries=[((k0,i0),1),((k0+1,i0+1),-1)]
                    )
                else:
                    for ii in range(i0, i1):
                        for jj in range(j0, j1):
                            for kk in range(k0, k1):
                                add_term(
                                    u_entries=[((ii,jj),1)],
                                    v_entries=[((jj,kk),1)],
                                    w_entries=[((kk,ii),1)]
                                )
    
    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
# EVOLVE-BLOCK-END

