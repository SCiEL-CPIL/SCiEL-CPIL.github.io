    def snap_to_rationals(U, V, W):
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
        
        # Try half-integers
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if verify(U_h, V_h, W_h):
            return U_h, V_h, W_h
        
        return None
