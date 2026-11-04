                    for i in range(2):
                        for k in range(8):
                            u_idx = i * m + 2
                            v_idx = 2 * p + k
                            w_idx = k * n + i
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            col += 1
                    