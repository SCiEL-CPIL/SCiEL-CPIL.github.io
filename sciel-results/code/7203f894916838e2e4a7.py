    # Try to find rank 50 solution
    # 5 blocks of 2x2x3 (rank 10 each) = 50
    if time.time() < deadline - 20:
        sol = _find_m223_rank10(timeout=15)
        if sol is not None:
            u_m223, v_m223, w_m223 = sol
            rank_new = 50
            U_new = [[0] * rank_new for _ in range(n * m)]
            V_new = [[0] * rank_new for _ in range(m * p)]
            W_new = [[0] * rank_new for _ in range(p * n)]
            
            for block in range(5):
                col_offset = block * 10
                k_base = block * 3
                for r in range(10):
                    c = col_offset + r
                    for i in range(2):
                        for j in range(2):
                            U_new[i * m + j][c] = float(u_m223[i * 2 + j, r])
                    for j in range(2):
                        for local_k in range(3):
                            global_k = k_base + local_k
                            V_new[j * p + global_k][c] = float(v_m223[j * 3 + local_k, r])
                    for local_k in range(3):
                        global_k = k_base + local_k
                        for i in range(2):
                            W_new[global_k * n + i][c] = float(w_m223[local_k * 2 + i, r])
                            
            return {"U": U_new, "V": V_new, "W": W_new}
