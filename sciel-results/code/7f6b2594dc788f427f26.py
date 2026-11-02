<<<<<<< SEARCH
                # Check convergence
                err = verify(U, V, W, rank)
                if err < 1e-8:
                    # Polish
                    # Flatten
                    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(params):
                        U_p = params[:n*m*rank].reshape(n, m, rank)
                        V_p = params[n*m*rank:2*n*m*rank].reshape(m, p, rank)
                        W_p = params[2*n*m*rank:].reshape(p, n, rank)
                        T = np.einsum('ijc,jkc,kic->ijk', U_p, V_p, W_p)
                        return (T - 1.0).flatten()
                    
                    try:
                        res = least_squares(objective, params, method='lm', max_nfev=100)
                        U_p = res.x[:n*m*rank].reshape(n, m, rank)
                        V_p = res.x[n*m*rank:2*n*m*rank].reshape(m, p, rank)
                        W_p = res.x[2*n*m*rank:].reshape(p, n, rank)
                        
                        U_r = round_to_half(U_p)
                        V_r = round_to_half(V_p)
                        W_r = round_to_half(W_p)
                        
                        err_r = verify(U_r, V_r, W_r, rank)
                        if err_r < 1e-8:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    except:
                        pass

                    # Fallback to direct rounding if polish fails
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    err_r = verify(U_r, V_r, W_r, rank)
                    if err_r < 1e-8:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        
        return None
=======
                # Check convergence
                err = verify(U, V, W, rank)
                if err < 1e-8:
                    # Direct rounding
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    
                    err_r = verify(U_r, V_r, W_r, rank)
                    if err_r < 1e-8:
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                    # Try rounding to integers if half-integers fail? 
                    # Or try a few more iterations with tighter tolerance
                    # For now, if rounding fails, we might try to snap to nearest integer
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    err_i = verify(U_i, V_i, W_i, rank)
                    if err_i < 1e-8:
                        return {
                            "U": U_i.tolist(),
                            "V": V_i.tolist(),
                            "W": W_i.tolist()
                        }
        
        return None
>>>>>>> REPLACE
