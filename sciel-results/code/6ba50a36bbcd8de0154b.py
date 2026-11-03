import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, np_), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def time_left():
        return DEADLINE - time.time()
    
    def verify(U, V, W):
        r = U.shape[1]
        result = np.zeros((nm, mp, np_), dtype=np.float64)
        for c in range(r):
            result += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-9, rtol=0)
    
    def to_half_integers(U, V, W):
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        return U2, V2, W2
    
    def als_refine(U, V, W, T, max_iter=30):
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for it in range(max_iter):
            if time_left() < 5:
                break
            # Update U
            for c in range(r):
                vw = np.outer(V[:,c], W[:,c])
                vw_flat = vw.ravel()
                norm = np.dot(vw_flat, vw_flat)
                if norm < 1e-12:
                    continue
                # residual = T - sum_{c' != c} U_{c'} V_{c'} W_{c'}
                # u[i] = sum_{j,k} residual[i,j,k] * vw[j,k] / norm
                # = sum_{j,k} T[i,j,k] * vw[j,k] / norm - sum_{c'!=c} U[i,c'] * (V[j,c']W[k,c'] * vw[j,k]) / norm
                # This is getting complicated. Let's just compute the residual explicitly.
                residual = T.copy()
                for c2 in range(r):
                    if c2 != c:
                        residual -= np.einsum('a,b,d->abd', U[:,c2], V[:,c2], W[:,c2])
                # u[i] = <residual[i,:,:], vw> / norm
                U[:,c] = np.einsum('ijk,jk->i', residual, vw) / norm
            # Update V
            for c in range(r):
                uw = np.outer(U[:,c], W[:,c])
                uw_flat = uw.ravel()
                norm = np.dot(uw_flat, uw_flat)
                if norm < 1e-12:
                    continue
                residual = T.copy()
                for c2 in range(r):
                    if c2 != c:
                        residual -= np.einsum('a,b,d->abd', U[:,c2], V[:,c2], W[:,c2])
                V[:,c] = np.einsum('ijk,ik->j', residual, uw) / norm
            # Update W
            for c in range(r):
                uv = np.outer(U[:,c], V[:,c])
                uv_flat = uv.ravel()
                norm = np.dot(uv_flat, uv_flat)
                if norm < 1e-12:
                    continue
                residual = T.copy()
                for c2 in range(r):
                    if c2 != c:
                        residual -= np.einsum('a,b,d->abd', U[:,c2], V[:,c2], W[:,c2])
                W[:,c] = np.einsum('ijk,ij->k', residual, uv) / norm
        return U, V, W
    
    # Schoolbook decomposition
    def schoolbook():
        r = n*m*p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((np_, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n*m*p
    
    rng = np.random.default_rng(42)
    
    # Try to reduce rank
    for r in range(n*m*p - 1, max(1, n*m*p - 15), -1):
        if time_left() < 10:
            break
        for attempt in range(2):
            if time_left() < 5:
                break
            U = rng.standard_normal((nm, r)) * 0.5
            V = rng.standard_normal((mp, r)) * 0.5
            W = rng.standard_normal((np_, r)) * 0.5
            U, V, W = als_refine(U, V, W, T, max_iter=20)
            U2, V2, W2 = to_half_integers(U, V, W)
            if verify(U2, V2, W2):
                best_U, best_V, best_W = U2, V2, W2
                best_rank = r
                break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
