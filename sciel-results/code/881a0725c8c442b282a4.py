"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    
    for rank in range(20, 25):
        for trial in range(10):
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(6, rank)
            V = np.random.randn(12, rank)
            W = np.random.randn(8, rank)
            
            for iteration in range(200):
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, rank))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        v, _, _, _ = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        O = np.zeros((m, rank))
                        for j in range(m):
                            O[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        w, _, _, _ = np.linalg.lstsq(O, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
            
            # Verify
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-8:
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                
                max_err_r = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(Ur[i*m+j, :] * Vr[j*p+k, :] * Wr[k*n+i, :])
                            max_err_r = max(max_err_r, abs(val - 1))
                
                if max_err_r < 1e-8:
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    # Fallback: schoolbook
    U = [[0]*24 for _ in range(6)]
    V = [[0]*24 for _ in range(12)]
    W = [[0]*24 for _ in range(8)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
