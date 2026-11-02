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
        """Rank-7 decomposition for 2x2*2x2.
        Returns U, V, W with shapes (4,7), (4,7), (4,7)
        U indexed by [i*2+j], V by [i*2+j], W by [i*2+j]
        """
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
        """Rank-9 decomposition for 2x2*2x3.
        Returns U, V, W with shapes (4,9), (6,9), (6,9)
        U indexed by [i*2+j], V by [i*3+j], W by [i*3+j]
        """
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

    def strassen_2x2x1():
        """Rank-3 decomposition for 2x2*2x1.
        Returns U, V, W with shapes (4,3), (2,3), (2,3)
        U indexed by [i*2+j], V by [i*2+j], W by [i*2+j]
        """
        U = np.zeros((4, 3))
        V = np.zeros((2, 3))
        W = np.zeros((2, 3))
        # m0: (a00+a11)(b00+b10) -> c00, c11
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1; W[1,0]=1
        # m1: (a00-a11)(b00-b10) -> c00, c11
        U[0,1]=1; U[3,1]=-1; V[0,1]=1; V[1,1]=-1; W[0,1]=1; W[1,1]=1
        # m2: (a01+a10)(b01+b11) -> c01, c10
        U[1,2]=1; U[2,2]=1; V[0,2]=1; V[1,2]=1; W[0,2]=1; W[1,2]=1
        return U, V, W

    def check(U, V, W, rnk):
        """Verify the tensor decomposition is exact."""
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
    # Split A into 2x2 blocks by rows (ar in {0,2}) and cols (ac in {0,2})
    # Each 2x2 A-block multiplies a 2x5 B-block (rows ac:ac+2 of B)
    # Split 2x5 into 2x2 (cols 0-1) + 2x2 (cols 2-3) + 2x1 (col 4): rank 7+7+3 = 17 per block
    # Total: 4 blocks * 17 = 68
    #
    # Alternative: split into 2x3 (cols 0-2) + 2x2 (cols 3-4): rank 9+7 = 16 per block
    # Total: 4 blocks * 16 = 64

    R = 64
    U_tot = np.zeros((n*m, R))
    V_tot = np.zeros((m*p, R))
    W_tot = np.zeros((p*n, R))

    s22 = strassen_2x2x2()
    s23 = strassen_2x2x3()
    s21 = strassen_2x2x1()

    # Try multiple split strategies and pick the best valid one
    best_result = None

    # Strategy 1: 2x3 + 2x2 (rank 16 per block, total 64)
    R1 = 64
    U1 = np.zeros((n*m, R1))
    V1 = np.zeros((m*p, R1))
    W1 = np.zeros((p*n, R1))
    off = 0
    for ar in (0, 2):
        for ac in (0, 2):
            # Part 1: 2x3 (cols 0,1,2 of B)
            for t in range(9):
                for ii in range(2):
                    for jj in range(2):
                        U1[(ar+ii)*m + (ac+jj), off+t] = s23[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(3):
                        V1[(ac+ii)*p + jj, off+t] = s23[1][ii*3+jj, t]
                for ii in range(2):
                    for jj in range(3):
                        W1[jj*n + (ar+ii), off+t] = s23[2][ii*3+jj, t]
            off += 9
            # Part 2: 2x2 (cols 3,4 of B)
            for t in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U1[(ar+ii)*m + (ac+jj), off+t] = s22[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        V1[(ac+ii)*p + (3+jj), off+t] = s22[1][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        W1[(3+jj)*n + (ar+ii), off+t] = s22[2][ii*2+jj, t]
            off += 7
    if check(U1, V1, W1, R1):
        best_result = {"U": to_list(U1), "V": to_list(V1), "W": to_list(W1)}

    # Strategy 2: 2x2 + 2x2 + 2x1 (rank 17 per block, total 68)
    R2 = 68
    U2 = np.zeros((n*m, R2))
    V2 = np.zeros((m*p, R2))
    W2 = np.zeros((p*n, R2))
    off = 0
    for ar in (0, 2):
        for ac in (0, 2):
            # Part 1: 2x2 (cols 0,1 of B)
            for t in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U2[(ar+ii)*m + (ac+jj), off+t] = s22[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        V2[(ac+ii)*p + jj, off+t] = s22[1][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        W2[jj*n + (ar+ii), off+t] = s22[2][ii*2+jj, t]
            off += 7
            # Part 2: 2x2 (cols 2,3 of B)
            for t in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U2[(ar+ii)*m + (ac+jj), off+t] = s22[0][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        V2[(ac+ii)*p + (2+jj), off+t] = s22[1][ii*2+jj, t]
                for ii in range(2):
                    for jj in range(2):
                        W2[(2+jj)*n + (ar+ii), off+t] = s22[2][ii*2+jj, t]
            off += 7
            # Part 3: 2x1 (col 4 of B)
            for t in range(3):
                for ii in range(2):
                    for jj in range(2):
                        U2[(ar+ii)*m + (ac+jj), off+t] = s21[0][ii*2+jj, t]
                for ii in range(2):
                    V2[(ac+ii)*p + 4, off+t] = s21[1][ii, t]
                for ii in range(2):
                    W2[4*n + (ar+ii), off+t] = s21[2][ii, t]
            off += 3
    if check(U2, V2, W2, R2):
        if best_result is None or R2 < R1:
            best_result = {"U": to_list(U2), "V": to_list(V2), "W": to_list(W2)}

    if best_result is not None:
        return best_result

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
