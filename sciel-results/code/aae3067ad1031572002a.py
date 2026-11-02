"""Matrix multiplication 4x4x5 - rank minimization via block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110

    def strassen_2x2x2():
        """Rank-7 decomposition for 2x2*2x2."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # m0: (a00+a11)(b00+b11) -> c00, c11
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # m1: (a10+a11)b00 -> c10
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
        # m2: a00(b01-b11) -> c01, c11
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # m3: a11(b10-b00) -> c10, c00
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1; W[0,3]=1
        # m4: (a00+a01)b11 -> c00, c01
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # m5: (a10-a00)(b00+b01) -> c11
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # m6: (a01-a11)(b10+b11) -> c00, c01
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
        return U, V, W

    def strassen_2x2x3():
        """Rank-9 decomposition for 2x2*2x3."""
        U = np.zeros((4, 9))
        V = np.zeros((6, 9))
        W = np.zeros((6, 9))
        # m0: (a00+a11)(b00+b10)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # m1: (a10+a11)(b00+b10)
        U[2,1]=1; U[3,1]=1; V[0,1]=1; V[3,1]=1; W[3,1]=1
        # m2: a00(b01-b11)
        U[0,2]=1; V[1,2]=1; V[4,2]=-1; W[1,2]=1; W[4,2]=1
        # m3: a11(b10-b00)
        U[3,3]=1; V[3,3]=1; V[0,3]=-1; W[3,3]=1; W[0,3]=1
        # m4: (a00+a01)(b02+b12)
        U[0,4]=1; U[1,4]=1; V[2,4]=1; V[5,4]=1; W[2,4]=1; W[5,4]=1
        # m5: a10(b02-b12)
        U[2,5]=1; V[2,5]=1; V[5,5]=-1; W[5,5]=1
        # m6: a01(b12-b02)
        U[1,6]=1; V[5,6]=1; V[2,6]=-1; W[2,6]=1
        # m7: (a10-a00)(b11+b01)
        U[2,7]=1; U[0,7]=-1; V[1,7]=1; V[4,7]=1; W[4,7]=1
        # m8: (a01-a11)(b11+b01)
        U[1,8]=1; U[3,8]=-1; V[1,8]=1; V[4,8]=1; W[1,8]=1
        return U, V, W

    def naive_2x2x1():
        """Rank-4 decomposition for 2x2*2x1."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def check(U, V, W, rnk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def to_list(arr):
        return [[round(v * 2) / 2.0 for v in row] for row in arr]

    # Strategy: 4x4*4x5 via 2x2 blocks of A
    # Each 2x2 A-block multiplies 2x5 B-block
    # Split 2x5 into 2x3 + 2x2: rank 9 + 7 = 16 per block
    # Total: 4 * 16 = 64

    R = 64
    U_tot = np.zeros((n*m, R))
    V_tot = np.zeros((m*p, R))
    W_tot = np.zeros((p*n, R))

    s22 = strassen_2x2x2()
    s23 = strassen_2x2x3()

    off = 0
    for ar in (0, 2):
        for ac in (0, 2):
            br = ac
            # Part 1: 2x3 (cols 0,1,2)
            for t in range(9):
                for ii in range(2):
                    for jj in range(2):
                        U_tot[(ar+ii)*m + (ac+jj), off+t] = s23[1][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(3):
                        V_tot[(br+ii)*p + jj, off+t] = s23[1][ii*3+jj, t]
                for ii in range(2):
                    for jj in range(3):
                        W_tot[jj*n + (ar+ii), off+t] = s23[2][ii*3+jj, t]
            off += 9
            # Part 2: 2x2 (cols 3,4)
            for t in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U_tot[(ar+ii)*m + (ac+jj), off+t] = s22[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        V_tot[(br+ii)*p + (3+jj), off+t] = s22[1][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        W_tot[(3+jj)*n + (ar+ii), off+t] = s22[2][ii*2+jj, t]
            off += 7

    if check(U_tot, V_tot, W_tot, R):
        return {"U": to_list(U_tot), "V": to_list(V_tot), "W": to_list(W_tot)}

    # Fallback: schoolbook rank 80
    Rs = n * m * p
    Us = np.zeros((n*m, Rs))
    Vs = np.zeros((m*p, Rs))
    Ws = np.zeros((p*n, Rs))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j, c] = 1.0
                Vs[j*p+k, c] = 1.0
                Ws[k*n+i, c] = 1.0
                c += 1
    return {"U": to_list(Us), "V": to_list(Vs), "W": to_list(Ws)}
# EVOLVE-BLOCK-END
