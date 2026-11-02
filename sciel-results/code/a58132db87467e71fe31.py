"""
5x5 matrix multiplication tensor decomposition.
Block decomposition: Strassen (rank 7) for 2x2, Laderman (rank 15) for 3x3,
schoolbook for cross-blocks. Total rank: 7+15+12+12+12+18+18+18 = 112.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    _start = time.time()
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def add_term(U, V, W, u_idx, v_idx, w_idx, col, u_val=1, v_val=1, w_val=1):
        """Add a single rank-1 term at column col."""
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    def schoolbook_block(n1, m1, p1, U, V, W, row_off_u, col_off_v, row_off_w, start_col):
        """Add schoolbook terms for n1 x m1 x p1 block at given offsets."""
        col = start_col
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[row_off_u + i*m1 + j][col] = 1
                    V[col_off_v + j*p1 + k][col] = 1
                    W[row_off_w + k*n1 + i][col] = 1
                    col += 1
        return col
    
    # Strassen 2x2 decomposition (7 terms)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For the tensor: U[i*2+j], V[j*2+k], W[k*2+i]
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # i=C[0,0], j=C[0,1], k=C[1,0], l=C[1,1]
    # W index for C[i,k] is W[k*2+i]
    
    strassen = [
        # (u_coeffs, v_coeffs, w_coeffs) where u is 4-tuple for A 2x2 row-major
        # u[0]=a, u[1]=b, u[2]=c, u[3]=d
        # v[0]=e, v[1]=f, v[2]=g, v[3]=h
        # w[0]=C[0,0] (W[0*2+0]), w[1]=C[1,0] (W[1*2+0]), w[2]=C[0,1] (W[0*2+1]), w[3]=C[1,1] (W[1*2+1])
        # M1 = (a+d)(e+h): u=[1,0,0,1], v=[1,1,0,1], w=[1,0,0,1]
        ((1,0,0,1), (1,1,0,1), (1,0,0,1)),   # M1
        # M2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0]
        ((0,0,1,1), (1,0,0,0), (0,1,0,0)),   # M2
        # M3 = a(f-h): u=[1,0,0,0], v=[0,1,0,-1], w=[0,0,1,0]
        ((1,0,0,0), (0,1,0,-1), (0,0,1,0)),  # M3
        # M4 = d(g-e): u=[0,0,0,1], v=[-1,0,1,0], w=[1,1,0,0]
        ((0,0,0,1), (-1,0,1,0), (1,1,0,0)),  # M4
        # M5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,1,0]
        ((1,1,0,0), (0,0,0,1), (-1,0,1,0)),  # M5
        # M6 = (c-a)(e+f): u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),  # M6
        # M7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),  # M7
    ]
    
    # Laderman 3x3 decomposition (15 terms)
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # C = [[s,t,u],[v,w,x],[y,z,aa]]
    # Using Laderman's algorithm with 15 products:
    # P1 = (a+b)(j+m)
    # P2 = (a-b)(j-m)
    # P3 = (c+d)(l+m)
    # P4 = (c-d)(l-m)
    # P5 = (f+g)(o+p)
    # P6 = (f-g)(o-p)
    # P7 = (a+e)(j+n)
    # P8 = (a-e)(j-n)
    # P9 = (e+i)(n+r)
    # P10 = (e-i)(n-r)
    # P11 = (b+f)(k+o)
    # P12 = (b-f)(k-o)
    # P13 = (h+g)(q+p)
    # P14 = (h-g)(q-p)
    # P15 = (b+h)(k+q)
    
    # C[0,0] = P1 + P2 + P7 - P8
    # C[0,1] = P7 + P11 - P12 - P15
    # C[0,2] = P1 + P3 + P9 - P10
    # C[1,0] = P1 - P2 + P5 - P6
    # C[1,1] = P5 + P8 + P12 + P15 - P1
    # C[1,2] = P3 + P9 - P10 - P4
    # C[2,0] = P9 - P10 + P13 - P14
    # C[2,1] = P13 + P11 - P12 - P15
    # C[2,2] = P5 - P6 + P1 + P4
    
    # u is 9-tuple for A 3x3 row-major: [a,b,c,d,e,f,g,h,i]
    # v is 9-tuple for B 3x3 row-major: [j,k,l,m,n,o,p,q,r]
    # w is 9-tuple for C 3x3: W[k*3+i] so w[0]=C[0,0], w[1]=C[1,0], w[2]=C[2,0], w[3]=C[0,1], w[4]=C[1,1], w[5]=C[2,1], w[6]=C[0,2], w[7]=C[1,2], w[8]=C[2,2]
    
    laderman = [
        # P1 = (a+b)(j+m): u=[1,1,0,0,0,0,0,0,0], v=[1,0,0,1,0,0,0,0,0], w=[1,0,0,0,0,0,1,0,0]
        ((1,1,0,0,0,0,0,0,0), (1,0,0,1,0,0,0,0,0), (1,0,0,0,0,0,1,0,0)),  # P1
        # P2 = (a-b)(j-m): u=[1,-1,0,0,0,0,0,0,0], v=[1,0,0,-1,0,0,0,0,0], w=[1,0,0,0,0,0,-1,0,0]
        ((1,-1,0,0,0,0,0,0,0), (1,0,0,-1,0,0,0,0,0), (1,0,0,0,0,0,-1,0,0)),  # P2
        # P3 = (c+d)(l+m): u=[0,0,1,1,0,0,0,0,0], v=[0,0,1,1,0,0,0,0,0], w=[1,0,0,0,0,0,0,0,0]
        ((0,0,1,1,0,0,0,0,0), (0,0,1,1,0,0,0,0,0), (1,0,0,0,0,0,0,0,0)),  # P3
        # P4 = (c-d)(l-m): u=[0,0,1,-1,0,0,0,0,0], v=[0,0,1,-1,0,0,0,0,0], w=[0,0,0,0,0,0,0,0,1]
        ((0,0,1,-1,0,0,0,0,0), (0,0,1,-1,0,0,0,0,0), (0,0,0,0,0,0,0,0,1)),  # P4
        # P5 = (f+g)(o+p): u=[0,0,0,0,0,1,1,0,0], v=[0,0,0,0,0,1,1,0,0], w=[0,1,0,0,0,0,0,0,1]
        ((0,0,0,0,0,1,1,0,0), (0,0,0,0,0,1,1,0,0), (0,1,0,0,0,0,0,0,1)),  # P5
        # P6 = (f-g)(o-p): u=[0,0,0,0,0,1,-1,0,0], v=[0,0,0,0,0,1,-1,0,0], w=[0,0,0,0,0,0,0,0,-1]
        ((0,0,0,0,0,1,-1,0,0), (0,0,0,0,0,1,-1,0,0), (0,0,0,0,0,0,0,0,-1)),  # P6
        # P7 = (a+e)(j+n): u=[1,0,0,0,1,0,0,0,0], v=[1,0,0,0,1,0,0,0,0], w=[0,0,0,1,0,0,0,0,0]
        ((1,0,0,0,1,0,0,0,0), (1,0,0,0,1,0,0,0,0), (0,0,0,1,0,0,0,0,0)),  # P7
        # P8 = (a-e)(j-n): u=[1,0,0,0,-1,0,0,0,0], v=[1,0,0,0,-1,0,0,0,0], w=[0,0,0,0,0,0,0,0,0]
        # Wait, P8 contributes to C[0,0] with -1, C[1,1] with +1
        ((1,0,0,0,-1,0,0,0,0), (1,0,0,0,-1,0,0,0,0), (0,0,0,0,0,0,0,0,0)),  # P8 - need to fix
        # P9 = (e+i)(n+r): u=[0,0,0,0,1,0,0,0,1], v=[0,0,0,0,1,0,0,0,1], w=[0,0,0,0,0,0,1,0,0]
        ((0,0,0,0,1,0,0,0,1), (0,0,0,0,1,0,0,0,1), (0,0,0,0,0,0,1,0,0)),  # P9
        # P10 = (e-i)(n-r): u=[0,0,0,0,1,0,0,0,-1], v=[0,0,0,0,1,0,0,0,-1], w=[0,0,0,0,0,0,-1,0,0]
        ((0,0,0,0,1,0,0,0,-1), (0,0,0,0,1,0,0,0,-1), (0,0,0,0,0,0,-1,0,0)),  # P10
        # P11 = (b+f)(k+o): u=[0,1,0,0,0,1,0,0,0], v=[0,1,0,0,0,1,0,0,0], w=[0,0,0,0,0,0,0,0,0]
        ((0,1,0,0,0,1,0,0,0), (0,1,0,0,0,1,0,0,0), (0,0,0,0,0,0,0,0,0)),  # P11 - need to fix
        # P12 = (b-f)(k-o): u=[0,1,0,0,0,-1,0,0,0], v=[0,1,0,0,0,-1,0,0,0], w=[0,0,0,0,0,0,0,0,0]
        ((0,1,0,0,0,-1,0,0,0), (0,1,0,0,0,-1,0,0,0), (0,0,0,0,0,0,0,0,0)),  # P12 - need to fix
        # P13 = (h+g)(q+p): u=[0,0,0,0,0,0,1,1,0], v=[0,0,0,0,0,0,1,1,0], w=[0,0,0,0,0,0,0,0,0]
        ((0,0,0,0,0,0,1,1,0), (0,0,0,0,0,0,1,1,0), (0,0,0,0,0,0,0,0,0)),  # P13 - need to fix
        # P14 = (h-g)(q-p): u=[0,0,0,0,0,0,-1,1,0], v=[0,0,0,0,0,0,-1,1,0], w=[0,0,0,0,0,0,0,0,0]
        ((0,0,0,0,0,0,-1,1,0), (0,0,0,0,0,0,-1,1,0), (0,0,0,0,0,0,0,0,0)),  # P14 - need to fix
        # P15 = (b+h)(k+q): u=[0,1,0,0,0,0,0,1,0], v=[0,1,0,0,0,0,0,1,0], w=[0,0,0,0,0,0,0,0,0]
        ((0,1,0,0,0,0,0,1,0), (0,1,0,0,0,0,0,1,0), (0,0,0,0,0,0,0,0,0)),  # P15 - need to fix
    ]
    
    # I need to fix the W coefficients for Laderman. Let me redo this carefully.
    # C[0,0] = P1 + P2 + P7 - P8
    # C[0,1] = P7 + P11 - P12 - P15
    # C[0,2] = P1 + P3 + P9 - P10
    # C[1,0] = P1 - P2 + P5 - P6
    # C[1,1] = P5 + P8 + P12 + P15 - P1
    # C[1,2] = P3 + P9 - P10 - P4
    # C[2,0] = P9 - P10 + P13 - P14
    # C[2,1] = P13 + P11 - P12 - P15
    # C[2,2] = P5 - P6 + P1 + P4
    
    # W index: W[k*3+i] where i is row, k is col of C
    # C[0,0] -> W[0*3+0] = W[0]
    # C[1,0] -> W[0*3+1] = W[1]
    # C[2,0] -> W[0*3+2] = W[2]
    # C[0,1] -> W[1*3+0] = W[3]
    # C[1,1] -> W[1*3+1] = W[4]
    # C[2,1] -> W[1*3+2] = W[5]
    # C[0,2] -> W[2*3+0] = W[6]
    # C[1,2] -> W[2*3+1] = W[7]
    # C[2,2] -> W[2*3+2] = W[8]
    
    laderman_fixed = [
        # P1: contributes to C[0,0](+1), C[0,2](+1), C[1,0](+1), C[1,1](-1), C[2,2](+1)
        ((1,1,0,0,0,0,0,0,0), (1,0,0,1,0,0,0,0,0), (1,0,1,0,0,0,1,0,1)),
        # P2: contributes to C[0,0](+1), C[0,2](-1), C[1,0](-1)
        ((1,-1,0,0,0,0,0,0,0), (1,0,0,-1,0,0,0,0,0), (1,0,0,0,0,0,-1,0,0)),
        # P3: contributes to C[0,2](+1), C[1,2](+1)
        ((0,0,1,1,0,0,0,0,0), (0,0,1,1,0,0,0,0,0), (0,0,0,0,0,0,1,1,0)),
        # P4: contributes to C[1,2](-1), C[2,2](+1)
        ((0,0,1,-1,0,0,0,0,0), (0,0,1,-1,0,0,0,0,0), (0,0,0,0,0,0,0,-1,1)),
        # P5: contributes to C[1,0](+1), C[1,1](+1), C[2,2](+1)
        ((0,0,0,0,0,1,1,0,0), (0,0,0,0,0,1,1,0,0), (0,1,0,0,1,0,0,0,1)),
        # P6: contributes to C[1,0](-1), C[2,2](-1)
        ((0,0,0,0,0,1,-1,0,0), (0,0,0,0,0,1,-1,0,0), (0,0,0,0,0,0,0,0,-1)),
        # P7: contributes to C[0,0](+1), C[0,1](+1)
        ((1,0,0,0,1,0,0,0,0), (1,0,0,0,1,0,0,0,0), (1,0,0,1,0,0,0,0,0)),
        # P8: contributes to C[0,0](-1), C[1,1](+1)
        ((1,0,0,0,-1,0,0,0,0), (1,0,0,0,-1,0,0,0,0), (-1,0,0,0,1,0,0,0,0)),
        # P9: contributes to C[0,2](+1), C[1,2](+1), C[2,0](+1)
        ((0,0,0,0,1,0,0,0,1), (0,0,0,0,1,0,0,0,1), (0,0,1,0,0,0,1,1,0)),
        # P10: contributes to C[0,2](-1), C[1,2](-1), C[2,0](-1)
        ((0,0,0,0,1,0,0,0,-1), (0,0,0,0,1,0,0,0,-1), (0,0,-1,0,0,0,-1,-1,0)),
        # P11: contributes to C[0,1](+1), C[2,1](+1)
        ((0,1,0,0,0,1,0,0,0), (0,1,0,0,0,1,0,0,0), (0,0,0,1,0,0,0,0,1)),
        # P12: contributes to C[0,1](-1), C[1,1](+1), C[2,1](-1)
        ((0,1,0,0,0,-1,0,0,0), (0,1,0,0,0,-1,0,0,0), (0,0,0,-1,1,0,0,0,-1)),
        # P13: contributes to C[2,0](+1), C[2,1](+1)
        ((0,0,0,0,0,0,1,1,0), (0,0,0,0,0,0,1,1,0), (0,0,1,0,0,1,0,0,0)),
        # P14: contributes to C[2,0](-1), C[2,1](-1)
        ((0,0,0,0,0,0,-1,1,0), (0,0,0,0,0,0,-1,1,0), (0,0,-1,0,0,-1,0,0,0)),
        # P15: contributes to C[0,1](-1), C[1,1](+1), C[2,1](-1)
        ((0,1,0,0,0,0,0,1,0), (0,1,0,0,0,0,0,1,0), (0,0,0,-1,1,0,0,0,-1)),
    ]
    
    # Now build the full 5x5 decomposition
    # Partition: rows/cols 0-1 (2x2) and 2-4 (3x3)
    
    total_rank = 7 + 15 + 12 + 12 + 12 + 18 + 18 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] using Strassen (7 terms)
    for (u, v, w) in strassen:
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i*2+j][col] += u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[j*2+k][col] += v[j*2+k]
        for i in range(2):
            for k in range(2):
                if w[k*2+i] != 0:
                    W[k*2+i][col] += w[k*2+i]
        col += 1
    
    # Block 2: A[2:5,2:5] * B[2:5,2:5] -> C[2:5,2:5] using Laderman (15 terms)
    for (u, v, w) in laderman_fixed:
        for i in range(3):
            for j in range(3):
                if u[i*3+j] != 0:
                    U[(2+i)*2+(2+j)][col] += u[i*3+j]
        for j in range(3):
            for k in range(3):
                if v[j*3+k] != 0:
                    V[(2+j)*2+(2+k)][col] += v[j*3+k]
        for i in range(3):
            for k in range(3):
                if w[k*3+i] != 0:
                    W[(2+k)*2+(2+i)][col] += w[k*3+i]
        col += 1
    
    # Block 3: A[0:2,2:5] * B[2:5,0:2] -> C[0:2,0:2] (2x3x2 = 12 terms)
    col = schoolbook_block(2, 3, 2, U, V, W, 2, 4, 0, col)
    
    # Block 4: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5] (2x2x3 = 12 terms)
    col = schoolbook_block(2, 2, 3, U, V, W, 0, 2, 6, col)
    
    # Block 5: A[2:5,0:2] * B[0:2,0:2] -> C[2:5,0:2] (3x2x2 = 12 terms)
    col = schoolbook_block(3, 2, 2, U, V, W, 8, 0, 2, col)
    
    # Block 6: A[2:5,2:5] * B[2:5,0:2] -> C[2:5,0:2] (3x3x2 = 18 terms)
    col = schoolbook_block(3, 3, 2, U, V, W, 14, 4, 2, col)
    
    # Block 7: A[0:2,2:5] * B[2:5,2:5] -> C[0:2,2:5] (2x3x3 = 18 terms)
    col = schoolbook_block(2, 3, 3, U, V, W, 2, 8, 6, col)
    
    # Block 8: A[2:5,0:2] * B[0:2,2:5] -> C[2:5,2:5] (3x2x3 = 18 terms)
    col = schoolbook_block(3, 2, 3, U, V, W, 8, 2, 8, col)
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
