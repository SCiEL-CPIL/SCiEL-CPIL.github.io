<<<<<<< SEARCH
def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-12, deadline=None):
    prev_err = np.inf
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        err = np.linalg.norm(T - np.einsum('ur,vr,wr->uvw', U, V, W))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err
=======
def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-12, deadline=None):
    prev_err = np.inf
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        err = np.linalg.norm(T - np.einsum('ur,vr,wr->uvw', U, V, W))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    # Final high-precision polish pass with tighter tolerance
    if deadline is None or time.time() < deadline - 0.5:
        for _ in range(10):
            if deadline is not None and time.time() > deadline:
                break
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n*m):
                b = T[u].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m*p):
                b = T[:, v, :].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p*n):
                b = T[:, :, w].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            err = np.linalg.norm(T - np.einsum('ur,vr,wr->uvw', U, V, W))
            if err < 1e-14:
                break
    return U, V, W, err
>>>>>>> REPLACE
