import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 3, 6
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        r = U.shape[1]
        # Vectorized verification
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Reshape to tensor?
        # Let's stick to loops for clarity but optimize if needed.
        # For r=36, nm=6, mp=18, pn=12.
        # 36 terms per (i,j,k). 36 (i,j,k) pairs. 1296 ops. Fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    # Use lstsq for speed and stability
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    w = np.zeros(r)
                W[k * n + i, :] = w
        return W

    def solve_for_U(V, W, r):
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    u = np.zeros(r)
                U[i * m + j, :] = u
        return U

    def solve_for_V(U, W, r):
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    v = np.zeros(r)
                V[j * p + k, :] = v
        return V

    def compute_error(U, V, W):
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err

    # Search for lower rank
    # Try ranks from 35 down to 30 (or lower)
    for rank in range(35, 29, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Try multiple random initializations
        for init in range(200):
            if time.time() > deadline - 8:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = float('inf')
            for it in range(100):
                W = solve_for_W(U, V, rank)
                U = solve_for_U(V, W, rank)
                V = solve_for_V(U, W, rank)
                
                err = compute_error(U, V, W)
                
                if err < 1e-10:
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                        break
                
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            if found:
                break
        if found:
            break
            
    return best_result
