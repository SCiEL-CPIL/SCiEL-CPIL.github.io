import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    def build_rank44():
        rank = 44
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        col = 0
        def add_strassen(rA, cA, rB, cB):
            nonlocal col
            # M1 = (X11+X22)(Y11+Y22)
            U[(rA+0)*m + (cA+0), col] = 1
            U[(rA+1)*m + (cA+1), col] = 1
            V[(rB+0)*p + (cB+0), col] = 1
            V[(rB+1)*p + (cB+1), col] = 1
            W[(cB+0)*n + (rA+0), col] = 1
            W[(cB+1)*n + (rA+1), col] = 1
            col += 1
            # M2 = (X21+X22)Y11
            U[(rA+1)*m + (cA+0), col] = 1
            U[(rA+1)*m + (cA+1), col] = 1
            V[(rB+0)*p + (cB+0), col] = 1
            W[(cB+0)*n + (rA+1), col] = 1
            W[(cB+1)*n + (rA+1), col] = -1
            col += 1
            # M3 = X11(Y12-Y22)
            U[(rA+0)*m + (cA+0), col] = 1
            V[(rB+0)*p + (cB+1), col] = 1
            V[(rB+1)*p + (cB+1), col] = -1
            W[(cB+1)*n + (rA+0), col] = 1
            W[(cB+1)*n + (rA+1), col] = 1
            col += 1
            # M4 = X22(Y21-Y11)
            U[(rA+1)*m + (cA+1), col] = 1
            V[(rB+1)*p + (cB+0), col] = 1
            V[(rB+0)*p + (cB+0), col] = -1
            W[(cB+0)*n + (rA+0), col] = 1
            W[(cB+0)*n + (rA+1), col] = 1
            col += 1
            # M5 = (X11+X12)Y22
            U[(rA+0)*m + (cA+0), col] = 1
            U[(rA+0)*m + (cA+1), col] = 1
            V[(rB+1)*p + (cB+1), col] = 1
            W[(cB+1)*n + (rA+0), col] = 1
            W[(cB+0)*n + (rA+0), col] = -1
            col += 1
            # M6 = (X21-X11)(Y11+Y12)
            U[(rA+1)*m + (cA+0), col] = 1
            U[(rA+0)*m + (cA+0), col] = -1
            V[(rB+0)*p + (cB+0), col] = 1
            V[(rB+0)*p + (cB+1), col] = 1
            W[(cB+1)*n + (rA+1), col] = 1
            col += 1
            # M7 = (X12-X22)(Y21+Y22)
            U[(rA+0)*m + (cA+1), col] = 1
            U[(rA+1)*m + (cA+1), col] = -1
            V[(rB+1)*p + (cB+0), col] = 1
            V[(rB+1)*p + (cB+1), col] = 1
            W[(cB+0)*n + (rA+0), col] = 1
            col += 1
        
        # Four 2x2x2 Strassen blocks
        add_strassen(0, 0, 0, 0)
        add_strassen(0, 2, 2, 0)
        add_strassen(0, 0, 0, 2)
        add_strassen(0, 2, 2, 2)
        
        # 1x4x4 schoolbook for row 2
        for j in range(4):
            for k in range(4):
                U[2*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + 2, col] = 1
                col += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol, rtol=0)
    
    if not verify(U44, V44, W44, n, m, p):
        # Fallback to schoolbook
        rank = 48
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to reduce rank using ALS
    def als_decompose(target, n, m, p, rank, init_U=None, init_V=None, init_W=None, max_iter=500, tol=1e-8):
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            np.random.seed(42)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U
            for r in range(rank):
                num = np.zeros(n*m)
                den = np.zeros(n*m)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            num[i*m+j] += target[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                            den[i*m+j] += V[j*p+k, r]**2 * W[k*n+i, r]**2
                mask = den > 1e-12
                U[mask, r] = num[mask] / den[mask]
            
            # Update V
            for r in range(rank):
                num = np.zeros(m*p)
                den = np.zeros(m*p)
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            num[j*p+k] += target[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                            den[j*p+k] += U[i*m+j, r]**2 * W[k*n+i, r]**2
                mask = den > 1e-12
                V[mask, r] = num[mask] / den[mask]
            
            # Update W
            for r in range(rank):
                num = np.zeros(p*n)
                den = np.zeros(p*n)
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            num[k*n+i] += target[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                            den[k*n+i] += U[i*m+j, r]**2 * V[j*p+k, r]**2
                mask = den > 1e-12
                W[mask, r] = num[mask] / den[mask]
            
            # Check convergence
            if it > 0 and it % 10 == 0:
                T = np.zeros((n, m, p))
                for r in range(rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    T += np.einsum('ij,jk,ki->ijk', u, v, w)
                if np.allclose(T, target, atol=tol, rtol=0):
                    break
        
        return U, V, W
    
    target = np.ones((n, m, p))
    
    # Try rank 43: initialize from rank-44 by dropping one term and re-optimizing
    for drop_idx in range(44):
        if time.time() > deadline:
            break
        U_init = np.delete(U44, drop_idx, axis=1)
        V_init = np.delete(V44, drop_idx, axis=1)
        W_init = np.delete(W44, drop_idx, axis=1)
        
        U_new, V_new, W_new = als_decompose(target, n, m, p, 43, U_init, V_init, W_init, max_iter=200)
        
        if verify(U_new, V_new, W_new, n, m, p):
            best_rank = 43
            best_U, best_V, best_W = U_new, V_new, W_new
            break
    
    if best_rank == 44 and time.time() < deadline - 10:
        # Try a few random restarts at rank 43
        np.random.seed(0)
        for trial in range(5):
            if time.time() > deadline:
                break
            U_init = np.random.randn(n*m, 43) * 0.1
            V_init = np.random.randn(m*p, 43) * 0.1
            W_init = np.random.randn(p*n, 43) * 0.1
            
            U_new, V_new, W_new = als_decompose(target, n, m, p, 43, U_init, V_init, W_init, max_iter=300)
            
            if verify(U_new, V_new, W_new, n, m, p):
                best_rank = 43
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
