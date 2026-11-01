        for block_idx in range(3):
            k_off = block_idx * 2
            for local_i in range(2):
                for local_j in range(3):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        U[local_i*3+local_j][full_t] = block_u[local_i*3+local_j][t]
            for local_j in range(3):
                for local_k in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        # Correct global index for V
                        global_v_idx = local_j * p + (k_off + local_k)
                        V[global_v_idx][full_t] = block_v[local_j*2+local_k][t]
            for local_k in range(2):
                for local_i in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        # Correct global index for W
                        global_w_idx = (k_off + local_k) * n + local_i
                        W[global_w_idx][full_t] = block_w[local_k*2+local_i][t]
