import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    deadline = time.time() + 140

    def verify(U, V, W):
        r = U.shape[1]
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(Ur[i,j,:] * Vr[j,k,:] * Wr[k,i,:])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    # Rank-29: 4 Strassen blocks (28 terms) + 1 term for row 2
    rank29 = 29
    U29 = np.zeros((n*m, rank29))
    V29 = np.zeros((m*p, rank29))
    W29 = np.zeros((p*n, rank29))

    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # m1: (a+d)(e+h)
            U29[0*m + b_r*2+0, col] = 1
            U29[1*m + b_r*2+1, col] = 1
            V29[(b_c*2+0)*p + b_c*2+0, col] = 1
            V29[(b_c*2+1)*p + b_c*2+1, col] = 1
            W29[(b_c*2+0)*n + 0, col] = 1
            W29[(b_c*2+1)*n + 1, col] = 1
            col += 1
            # m2: (c+d)e
            U29[1*m + b_r*2+0, col] = 1
            U29[1*m + b_r*2+1, col] = 1
            V29[(b_c*2+0)*p + b_c*2+0, col] = 1
            W29[(b_c*2+0)*n + 1, col] = 1
            W29[(b_c*2+1)*n + 1, col] = -1
            col += 1
            # m3: a(f-h)
            U29[0*m + b_r*2+0, col] = 1
            V29[(b_c*2+0)*p + b_c*2+1, col] = 1
            V29[(b_c*2+1)*p + b_c*2+1, col] = -1
            W29[(b_c*2+1)*n + 0, col] = 1
            W29[(b_c*2+1)*n + 1, col] = 1
            col += 1
            # m4: d(g-e)
            U29[1*m + b_r*2+1, col] = 1
            V29[(b_c*2+1)*p + b_c*2+0, col] = 1
            V29[(b_c*2+0)*p + b_c*2+0, col] = -1
            W29[(b_c*2+0)*n + 0, col] = 1
            W29[(b_c*2+0)*n + 1, col] = 1
            col += 1
            # m5: (a+b)h
            U29[0*m + b_r*2+0, col] = 1
            U29[0*m + b_r*2+1, col] = 1
            V29[(b_c*2+1)*p + b_c*2+1, col] = 1
            W29[(b_c*2+0)*n + 0, col] = -1
            W29[(b_c*2+1)*n + 0, col] = 1
            col += 1
            # m6: (c-a)(e+f)
            U29[0*m + b_r*2+0, col] = -1
            U29[1*m + b_r*2+0, col] = 1
            V29[(b_c*2+0)*p + b_c*2+0, col] = 1
            V29[(b_c*2+0)*p + b_c*2+1, col] = 1
            W29[(b_c*2+1)*n + 1, col] = 1
            col += 1
            # m7: (b-d)(g+h)
            U29[0*m + b_r*2+1, col] = 1
            U29[1*m + b_r*2+1, col] = -1
            V29[(b_c*2+1)*p + b_c*2+0, col] = 1
            V29[(b_c*2+1)*p + b_c*2+1, col] = 1
            W29[(b_c*2+0)*n + 0, col] = 1
            col += 1

    # Row 2: single term
    for j in range(4):
        U29[2*m + j, col] = 1
    for j in range(4):
        for k in range(4):
            V29[j*p + k, col] = 1
    for k in range(4):
        W29[k*n + 2, col] = 1
    col += 1

    if not verify(U29, V29, W29):
        # Fallback to rank 44
        rank44 = 44
        U44 = np.zeros((n*m, rank44))
        V44 = np.zeros((m*p, rank44))
        W44 = np.zeros((p*n, rank44))
        col = 0
        for b_r in range(2):
            for b_c in range(2):
                U44[0*m + b_r*2+0, col] = 1; U44[1*m + b_r*2+1, col] = 1
                V44[(b_r*2+0)*p + b_c*2+0, col] = 1; V44[(b_r*2+1)*p + b_c*2+1, col] = 1
                W44[(b_c*2+0)*n + 0, col] = 1; W44[(b_c*2+1)*n + 1, col] = 1; col += 1
                U44[1*m + b_r*2+0, col] = 1; U44[1*m + b_r*2+1, col] = 1
                V44[(b_r*2+0)*p + b_c*2+0, col] = 1
                W44[(b_c*2+0)*n + 1, col] = 1; W44[(b_c*2+1)*n + 1, col] = -1; col += 1
                U44[0*m + b_r*2+0, col] = 1
                V44[(b_r*2+0)*p + b_c*2+1, col] = 1; V44[(b_r*2+1)*p + b_c*2+1, col] = -1
                W44[(b_c*2+1)*n + 0, col] = 1; W44[(b_c*2+1)*n + 1, col] = 1; col += 1
                U44[1*m + b_r*2+1, col] = 1
                V44[(b_r*2+1)*p + b_c*2+0, col] = 1; V44[(b_r*2+0)*p + b_c*2+0, col] = -1
                W44[(b_c*2+0)*n + 0, col] = 1; W44[(b_c*2+0)*n + 1, col] = 1; col += 1
                U44[0*m + b_r*2+0, col] = 1; U44[0*m + b_r*2+1, col] = 1
                V44[(b_r*2+1)*p + b_c*2+1, col] = 1
                W44[(b_c*2+0)*n + 0, col] = -1; W44[(b_c*2+1)*n + 0, col] = 1; col += 1
                U44[0*m + b_r*2+0, col] = -1; U44[1*m + b_r*2+0, col] = 1
                V44[(b_r*2+0)*p + b_c*2+0, col] = 1; V44[(b_r*2+0)*p + b_c*2+1, col] = 1
                W44[(b_c*2+1)*n + 1, col] = 1; col += 1
                U44[0*m + b_r*2+1, col] = 1; U44[1*m + b_r*2+1, col] = -1
                V44[(b_r*2+1)*p + b_c*2+0, col] = 1; V44[(b_r*2+1)*p + b_c*2+1, col] = 1
                W44[(b_c*2+0)*n + 0, col] = 1; col += 1
                for j_local in range(2):
                    for k_local in range(2):
                        U44[2*m + b_r*2+j_local, col] = 1
                        V44[(b_r*2+j_local)*p + b_c*2+k_local, col] = 1
                        W44[(b_c*2+k_local)*n + 2, col] = 1
                        col += 1
        return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}

    return {"U": U29.tolist(), "V": V29.tolist(), "W": W29.tolist()}
