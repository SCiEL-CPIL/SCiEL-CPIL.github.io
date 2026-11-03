    def error_func(params, r):
        nm, mp, pn = n * m, m * p, p * n
        U = params[:r * nm].reshape(r, nm).T
        V = params[r * nm:r * nm + r * mp].reshape(r, mp).T
        W = params[r * nm + r * mp:].reshape(r, pn).T
        
        T_comp = np.einsum('ac,bc,dc->abd', U, V, W)
        # indices are (a, b, d)
        # We need to flatten T_comp and pick? Or use advanced indexing.
        # T_comp is (n*m, m*p, p*n)
        # indices list contains (a, b, d)
        # We can convert indices to arrays
        a_idx = [x[0] for x in indices]
        b_idx = [x[1] for x in indices]
        d_idx = [x[2] for x in indices]
        
        vals = T_comp[a_idx, b_idx, d_idx]
        return np.sum((vals - 1.0) ** 2)
