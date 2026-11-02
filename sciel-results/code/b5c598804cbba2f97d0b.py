"""Matrix multiplication 3,4,5 - rank 56 decomposition using Strassen on 2x2 blocks."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(A_idx, B_idx, C_idx, start):
        """Add Strassen 2x2x2. A_idx/B_idx: {'a':(r,c),...}, C_idx: {'c00':(r,c),...}"""
        a = A_idx['a']; b = A_idx['b']; c = A_idx['c']; d = A_idx['d']
        e = B_idx['e']; f = B_idx['f']; g = B_idx['g']; h = B_idx['h']
        c00 = C_idx['c00']; c01 = C_idx['c01']; c10 = C_idx['c10']; c11 = C_idx['c11']
        ci = start
        
        # M1=(a+d)(e+h): C00+=1,C01+=1,C10-=1,C11-=1
        U[a[0]*m+a[1]][ci]+=1; U[d[0]*m+d[1]][ci]+=1
        V[e[0]*p+e[1]][ci]+=1; V[h[0]*p+h[1]][ci]+=1
        W[c00[1]*n+c00[0]][ci]+=1; W[c01[1]*n+c01[0]][ci]+=1
        W[c10[1]*n+c10[0]][ci]-=1; W[c11[1]*n+c11[0]][ci]-=1
        ci+=1
        # M2=(c+d)e: C10+=1,C11-=1
        U[c[0]*m+c[1]][ci]+=1; U[d[0]*m+d[1]][ci]+=1
        V[e[0]*p+e[1]][ci]+=1
        W[c10[1]*n+c10[0]][ci]+=1; W[c11[1]*n+c11[0]][ci]-=1
        ci+=1
        # M3=a(f-h): C01+=1,C11-=1
        U[a[0]*m+a[1]][ci]+=1
        V[f[0]*p+f[1]][ci]+=1; V[h[0]*p+h[1]][ci]-=1
        W[c01[1]*n+c01[0]][ci]+=1; W[c11[1]*n+c11[0]][ci]-=1
        ci+=1
        # M4=d(g-c): C00+=1,C10-=1
        U[d[0]*m+d[1]][ci]+=1
        V[g[0]*p+g[1]][ci]+=1; V[c[0]*p+c[1]][ci]-=1
        W[c00[1]*n+c00[0]][ci]+=1; W[c10[1]*n+c10[0]][ci]-=1
        ci+=1
        # M5=(a+b)h: C00-=1,C01+=1
        U[a[0]*m+a[1]][ci]+=1; U[b[0]*m+b[1]][ci]+=1
        V[h[0]*p+h[1]][ci]+=1
        W[c00[1]*n+c00[0]][ci]-=1; W[c01[1]*n+c01[0]][ci]+=1
        ci+=1
        # M6=(c-a)(e+g): C10+=1,C11-=1
        U[c[0]*m+c[1]][ci]+=1; U[a[0]*m+a[1]][ci]-=1
        V[e[0]*p+e[1]][ci]+=1; V[g[0]*p+g[1]][ci]+=1
        W[c10[1]*n+c10[0]][ci]+=1; W[c11[1]*n+c11[0]][ci]-=1
        ci+=1
        # M7=(b-d)(f+h): C01-=1,C10+=1
        U[b[0]*m+b[1]][ci]+=1; U[d[0]*m+d[1]][ci]-=1
        V[f[0]*p+f[1]][ci]+=1; V[h[0]*p+h[1]][ci]+=1
        W[c01[1]*n+c01[0]][ci]-=1; W[c10[1]*n+c10[0]][ci]+=1
        ci+=1
        return ci
    
    def add_schoolbook(ar, ac, br, bc, cr, cc, start):
        U[ar*m+ac][start]=1; V[br*p+bc][start]=1; W[cc*n+cr][start]=1
        return start+1
    
    # Block 1: C[0:2,0:2] via two Strassen
    col = add_strassen({'a':(0,0),'b':(0,2),'c':(1,0),'d':(1,2)},
                       {'e':(0,0),'f':(0,1),'g':(2,0),'h':(2,1)},
                       {'c00':(0,0),'c01':(0,1),'c10':(1,0),'c11':(1,1)}, col)
    col = add_strassen({'a':(0,2),'b':(0,3),'c':(1,2),'d':(1,3)},
                       {'e':(2,0),'f':(2,1),'g':(3,0),'h':(3,1)},
                       {'c00':(0,0),'c01':(0,1),'c10':(1,0),'c11':(1,1)}, col)
    # Block 2: C[0:2,2:4] via two Strassen
    col = add_strassen({'a':(0,0),'b':(0,1),'c':(1,0),'d':(1,1)},
                       {'e':(0,2),'f':(0,3),'g':(1,2),'h':(1,3)},
                       {'c00':(0,2),'c01':(0,3),'c10':(1,2),'c11':(1,3)}, col)
    col = add_strassen({'a':(0,2),'b':(0,3),'c':(1,2),'d':(1,3)},
                       {'e':(2,2),'f':(2,3),'g':(3,2),'h':(3,3)},
                       {'c00':(0,2),'c01':(0,3),'c10':(1,2),'c11':(1,3)}, col)
    # Block 3: C[0:2,4] schoolbook 8
    for ar in range(2):
        for ac in range(4):
            for br in range(4):
                if br == ac:  # only diagonal? No, all pairs
                    pass
    # Actually for 2x4*4x1, we need A[ar,ac]*B[ac,4] for all ar,ac
    for ar in range(2):
        for ac in range(4):
            col = add_schoolbook(ar, ac, ac, 4, ar, 4, col)
    # Block 4: C[2,0:2] schoolbook 8
    for ac in range(4):
        for bc in range(2):
            col = add_schoolbook(2, ac, ac, bc, 2, bc, col)
    # Block 5: C[2,2:4] schoolbook 8
    for ac in range(4):
        for bc in range(2,4):
            col = add_schoolbook(2, ac, ac, bc, 2, bc, col)
    # Block 6: C[2,4] schoolbook 4
    for ac in range(4):
        col = add_schoolbook(2, ac, ac, 4, 2, 4, col)
    
    assert col == rank, f"Got {col}, expected {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
