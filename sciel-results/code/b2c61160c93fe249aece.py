import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = 2, 2, 7
    deadline = time.time() + 115
    
    rank = 25
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_strassen(col_start, k_start):
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0+k_start, col_start+0] += 1; V[7*1+k_start+1, col_start+0] += 1
        W[2*k_start+0, col_start+0] += 1; W[2*(k_start+1)+1, col_start+0] += 1
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0+k_start, col_start+1] += 1
        W[2*k_start+1, col_start+1] += 1; W[2*(k_start+1)+1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0+k_start+1, col_start+2] += 1; V[7*1+k_start+1, col_start+2] -= 1
        W[2*(k_start+1)+0, col_start+2] += 1; W[2*(k_start+1)+1, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
        W[2*k_start+0, col_start+3] += 1; W[2*k_start+1, col_start+3] += 1
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1+k_start+1, col_start+4] += 1
        W[2*k_start+0, col_start+4] -= 1; W[2*(k_start+1)+0, col_start+4] += 1
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0+k_start, col_start+5] += 1; V[7*0+k_start+1, col_start+5] += 1
        W[2*(k_start+1)+1, col_start+5] += 1
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1+k_start, col_start+6] += 1; V[7*1+k_start+1, col_start+6] += 1
        W[2*k_start+0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    cs = 21
    U[0, cs+0] += 1; V[0+6, cs+0] += 1; W[12+0, cs+0] += 1
    U[1, cs+1] += 1; V[7+6, cs+1] += 1; W[12+0, cs+1] += 1
    U[2, cs+2] += 1; V[0+6, cs+2] += 1; W[12+1, cs+2] += 1
    U[3, cs+3] += 1; V[7+6, cs+3] += 1; W[12+1, cs+3] += 1

    def verify(U, V, W, trials=50):
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for _ in range(trials):
            A = np.random.randint(-10, 11, size=(n, m)).astype(np.float64)
            B = np.random.randint(-10, 11, size=(m, p)).astype(np.float64)
            C_true = A @ B
            A_flat = A.reshape(n*m)
            B_flat = B.reshape(m*p)
            u_vals = Ua.T @ A_flat
            v_vals = Va.T @ B_flat
            prod = u_vals * v_vals
            C_flat = Wa.T @ prod
            C_calc = C_flat.reshape(p, n).T
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        return True

    if verify(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
