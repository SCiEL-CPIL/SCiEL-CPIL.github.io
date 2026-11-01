import numpy as np
import time

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    deadline = start_time + 130

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        error = np.max(np.abs(T - T_recon))
        return error < tol, error

    # Schoolbook fallback (rank 16)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Strassen-inspired rank-14 decomposition
    def strassen_14():
        # A indices: a = i*2+j, i∈{0,1}, j∈{0,1}
        # a=0: (i=0,j=0)=a00, a=1: (i=0,j=1)=a01, a=2: (i=1,j=0)=a10, a=3: (i=1,j=1)=a11
        # B indices: b = j*4+k, j∈{0,1}, k∈{0,1,2,3}
        # b=0:(0,0), b=1:(0,1), b=2:(0,2), b=3:(0,3), b=4:(1,0), b=5:(1,1), b=6:(1,2), b=7:(1,3)
        # C indices: c = k*2+i, k∈{0,1,2,3}, i∈{0,1}
        # c=0:(k=0,i=0), c=1:(k=0,i=1), c=2:(k=1,i=0), c=3:(k=1,i=1), c=4:(k=2,i=0), c=5:(k=2,i=1), c=6:(k=3,i=0), c=7:(k=3,i=1)
        
        rank = 14
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Left half: B[:, 0:2]
        # P1 = (a00+a11)(b00+b11) where b00=B[0,0], b11=B[1,1]
        # A: a00(a=0), a11(a=3)
        # B: B[0,0](b=0), B[1,1](b=5)
        # C: C[0,0](c=0) with +1, C[1,1](c=3) with +1
        d = 0
        U[0, d] = 1; U[3, d] = 1
        V[0, d] = 1; V[5, d] = 1
        W[0, d] = 1; W[3, d] = 1
        
        # P2 = (a10+a11)b00 where b00=B[0,0]
        # A: a10(a=2), a11(a=3)
        # B: B[0,0](b=0)
        # C: C[1,0](c=1) with +1, C[1,1](c=3) with -1
        d = 1
        U[2, d] = 1; U[3, d] = 1
        V[0, d] = 1
        W[1, d] = 1; W[3, d] = -1
        
        # P3 = a00(b10-b11) where b10=B[1,0], b11=B[1,1]
        # A: a00(a=0)
        # B: B[1,0](b=4) with +1, B[1,1](b=5) with -1
        # C: C[0,1](c=2) with +1, C[1,1](c=3) with +1
        d = 2
        U[0, d] = 1
        V[4, d] = 1; V[5, d] = -1
        W[2, d] = 1; W[3, d] = 1
        
        # P4 = a11(b01-b00) where b01=B[0,1], b00=B[0,0]
        # A: a11(a=3)
        # B: B[0,1](b=1) with +1, B[0,0](b=0) with -1
        # C: C[0,0](c=0) with -1, C[1,0](c=1) with +1
        d = 3
        U[3, d] = 1
        V[1, d] = 1; V[0, d] = -1
        W[0, d] = -1; W[1, d] = 1
        
        # P5 = (a00+a01)b11 where b11=B[1,1]
        # A: a00(a=0), a01(a=1)
        # B: B[1,1](b=5)
        # C: C[0,0](c=0) with +1, C[0,1](c=2) with +1
        d = 4
        U[0, d] = 1; U[1, d] = 1
        V[5, d] = 1
        W[0, d] = 1; W[2, d] = 1
        
        # P6 = (a10-a00)(b00+b01)
        # A: a10(a=2) with +1, a00(a=0) with -1
        # B: B[0,0](b=0), B[0,1](b=1)
        # C: C[1,1](c=3) with +1
        d = 5
        U[2, d] = 1; U[0, d] = -1
        V[0, d] = 1; V[1, d] = 1
        W[3, d] = 1
        
        # P7 = (a01-a11)(b10+b11)
        # A: a01(a=1) with +1, a11(a=3) with -1
        # B: B[1,0](b=4), B[1,1](b=5)
        # C: C[0,0](c=0) with -1
        d = 6
        U[1, d] = 1; U[3, d] = -1
        V[4, d] = 1; V[5, d] = 1
        W[0, d] = -1
        
        # Right half: B[:, 2:4]
        # bR_00=B[0,2](b=2), bR_01=B[0,3](b=3), bR_10=B[1,2](b=6), bR_11=B[1,3](b=7)
        
        # P8 = (a00+a11)(bR_00+bR_11)
        d = 7
        U[0, d] = 1; U[3, d] = 1
        V[2, d] = 1; V[7, d] = 1
        W[4, d] = 1; W[7, d] = 1  # C[0,2](c=4), C[1,3](c=7)
        
        # P9 = (a10+a11)bR_00
        d = 8
        U[2, d] = 1; U[3, d] = 1
        V[2, d] = 1
        W[5, d] = 1; W[7, d] = -1  # C[1,2](c=5), C[1,3](c=7)
        
        # P10 = a00(bR_10-bR_11)
        d = 9
        U[0, d] = 1
        V[6, d] = 1; V[7, d] = -1
        W[6, d] = 1; W[7, d] = 1  # C[0,3](c=6), C[1,3](c=7)
        
        # P11 = a11(bR_01-bR_00)
        d = 10
        U[3, d] = 1
        V[3, d] = 1; V[2, d] = -1
        W[4, d] = -1; W[5, d] = 1  # C[0,2](c=4), C[1,2](c=5)
        
        # P12 = (a00+a01)bR_11
        d = 11
        U[0, d] = 1; U[1, d] = 1
        V[7, d] = 1
        W[4, d] = 1; W[6, d] = 1  # C[0,2](c=4), C[0,3](c=6)
        
        # P13 = (a10-a00)(bR_00+bR_01)
        d = 12
        U[2, d] = 1; U[0, d] = -1
        V[2, d] = 1; V[3, d] = 1
        W[7, d] = 1  # C[1,3](c=7)
        
        # P14 = (a01-a11)(bR_10+bR_11)
        d = 13
        U[1, d] = 1; U[3, d] = -1
        V[6, d] = 1; V[7, d] = 1
        W[4, d] = -1  # C[0,2](c=4)
        
        return U, V, W

    # Try rank 14 Strassen-inspired
    if time.time() < deadline:
        U14, V14, W14 = strassen_14()
        ok, err = verify(U14, V14, W14)
        if ok:
            U_list = [[float(x) for x in row] for row in U14]
            V_list = [[float(x) for x in row] for row in V14]
            W_list = [[float(x) for x in row] for row in W14]
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            print(f"Strassen-14 error: {err}")
            # Print details for debugging
            T_recon = np.einsum('ad,bd,cd->abc', U14, V14, W14)
            diff = T - T_recon
            nonzero = np.where(np.abs(diff) > 1e-9)
            for idx in zip(*nonzero):
                print(f"  T{idx}={T[idx]}, recon={T_recon[idx]}, diff={diff[idx]}")

    # Fallback to schoolbook
    U, V, W = schoolbook()
    ok, err = verify(U, V, W)
    if ok:
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Should never get here
    return {"U": [[0.0]*16 for _ in range(4)], "V": [[0.0]*16 for _ in range(8)], "W": [[0.0]*16 for _ in range(8)]}
