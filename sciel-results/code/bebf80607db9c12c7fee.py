            # Old:
            # U[col] = u_vec
            # V[col] = v_vec
            # W[col] = w_vec
            
            # New:
            for i in range(16):
                U[i][col] = u_vec[i]
                V[i][col] = v_vec[i]
                W[i][col] = w_vec[i]
