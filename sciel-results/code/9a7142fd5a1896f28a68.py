                # Let's inline the logic for clarity and correctness with coefficients
                for (li, lj, val_u) in u_list:
                    global_i = li
                    global_j = j_base + lj
                    U[global_i * m + global_j][col] += val_u
                for (lj, lk, val_v) in v_list:
                    global_j = j_base + lj
                    global_k = k_base + lk
                    V[global_j * p + global_k][col] += val_v
                for (li, lk, val_w) in w_list:
                    global_i = li
                    global_k = k_base + lk
                    W[global_k * n + global_i][col] += val_w
                    
                col += 1
