def solve_2x3x3(rank=13, max_iters=500):
    n, m, p = 2, 3, 3
    nm, mp, pn = 6, 9, 6
    
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        if np.max(np.abs(R - target)) > tol:
            return False
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True

    def optimize(U, V, W, max_iters=500, lr_init=0.1):
        # Same as before but local
        pass
        
    # Try random inits
    for attempt in range(10):
        np.random.seed(attempt)
        U = np.random.randn(nm, rank) * 0.3
        V = np.random.randn(mp, rank) * 0.3
        W = np.random.randn(pn, rank) * 0.3
        
        U_opt, V_opt, W_opt, err = optimize(U, V, W, max_iters=max_iters, lr_init=0.05)
        U_r = np.round(U_opt * 2) / 2.0
        V_r = np.round(V_opt * 2) / 2.0
        W_r = np.round(W_opt * 2) / 2.0
        
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
            
    return None
