import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 140  # safety margin before 150s kill

    def add_strassen_block(U, V, W, col_start, k_start):
        """Correct Strassen 2x2x2 for columns k_start, k_start+1.
        U indices: a11=0, a12=1, a21=2, a22=3
        V indices: j*p+k
        W indices: k*n+i
        """
        # M1 = (a11+a22)(b10+b21) -> w00 += 1, w11 += 1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0*p + k_start, col_start+0] += 1
        V[1*p + (k_start+1), col_start+0] += 1
        W[k_start*n + 0, col_start+0] += 1
        W[(k_start+1)*n + 1, col_start+0] += 1

        # M2 = (a21+a22)(b10) -> w10 += 1, w11 -= 1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0*p + k_start, col_start+1] += 1
        W[k_start*n + 1, col_start+1] += 1
        W[(k_start+1)*n + 1, col_start+1] -= 1

        # M3 = (a11)(b11+b21) -> w00 += 1, w01 += 1
        U[0, col_start+2] += 1
        V[0*p + (k_start+1), col_start+2] += 1
        V[1*p + (k_start+1), col_start+2] += 1
        W[k_start*n + 0, col_start+2] += 1
        W[(k_start+1)*n + 0, col_start+2] += 1

        # M4 = (a22)(b10+b20) -> w01 += 1, w11 += 1
        U[3, col_start+3] += 1
        V[0*p + k_start, col_start+3] += 1
        V[1*p + k_start, col_start+3] += 1
        W[k_start*n + 1, col_start+3] += 1
        W[(k_start+1)*n + 1, col_start+3] += 1

        # M5 = (a11+a12)(b20) -> w01 -= 1, w10 += 1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[1*p + k_start, col_start+4] += 1
        W[k_start*n + 1, col_start+4] -= 1
        W[(k_start+1)*n + 1, col_start+4] += 1

        # M6 = (a21-a11)(b21) -> w00 -= 1, w10 += 1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[1*p + (k_start+1), col_start+5] += 1
        W[k_start*n + 0, col_start+5] -= 1
        W[(k_start+1)*n + 0, col_start+5] += 1

        # M7 = (a12-a22)(b10+b20) -> w01 += 1, w11 -= 1
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[0*p + k_start, col_start+6] += 1
        V[1*p + k_start, col_start+6] += 1
        W[k_start*n + 1, col_start+6] += 1
        W[(k_start+1)*n + 1, col_start+6] -= 1

    def verify(U_mat, V_mat, W_mat, rank):
        """Verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Build rank-25: 3 Strassen blocks (cols 0-1, 2-3, 4-5) + schoolbook (col 6)
    rank = 25
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    add_strassen_block(U, V, W, 0, 0)   # columns 0, 1
    add_strassen_block(U, V, W, 7, 2)   # columns 2, 3
    add_strassen_block(U, V, W, 14, 4)  # columns 4, 5

    # Schoolbook for column 6
    U[0, 21] += 1
    V[0*p + 6, 21] += 1
    W[6*n + 0, 21] += 1

    U[1, 22] += 1
    V[1*p + 6, 22] += 1
    W[6*n + 0, 22] += 1

    U[2, 23] += 1
    V[0*p + 6, 23] += 1
    W[6*n + 1, 23] += 1

    U[3, 24] += 1
    V[1*p + 6, 24] += 1
    W[6*n + 1, 24] += 1

    # Verify
    if not verify(U, V, W, rank):
        # Fallback: pure schoolbook rank 28
        rank = 28
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
