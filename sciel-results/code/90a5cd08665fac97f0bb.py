    def add_factors(U_list, V_list, W_list, I_vals, J_vals, K_vals, local_rank):
        nonlocal rank
        for c in range(local_rank):
            u = [0.0] * (n*m)
            v = [0.0] * (m*p)
            w = [0.0] * (p*n)
            
            # U_list is rows. li is local row index.
            for li in range(len(U_list)):
                val = U_list[li][c]
                if val != 0:
                    i_loc = li // len(J_vals)
                    j_loc = li % len(J_vals)
                    gi = I_vals[i_loc]
                    gj = J_vals[j_loc]
                    u[gi * m + gj] = val
                    
            for lj in range(len(V_list)):
                val = V_list[lj][c]
                if val != 0:
                    j_loc = lj // len(K_vals)
                    k_loc = lj % len(K_vals)
                    gj = J_vals[j_loc]
                    gk = K_vals[k_loc]
                    v[gj * p + gk] = val
                    
            for lk in range(len(W_list)):
                val = W_list[lk][c]
                if val != 0:
                    k_loc = lk // len(I_vals)
                    i_loc = lk % len(I_vals)
                    gk = K_vals[k_loc]
                    gi = I_vals[i_loc]
                    w[gk * n + gi] = val
                    
            U.append(u)
            V.append(v)
            W.append(w)
            rank += 1
