        T_check = np.zeros((4, 4, 4), dtype=float)
        for r in range(rank):
            u_r = U_np[:, r]  # 16 entries
            v_r = V_np[:, r]  # 16 entries
            w_r = W_np[:, r]  # 16 entries
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        T_check[i, j, k] += u_r[i*4+j] * v_r[j*4+k] * w_r[k*4+i]
