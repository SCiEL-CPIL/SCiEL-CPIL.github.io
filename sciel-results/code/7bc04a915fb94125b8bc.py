                for u in range(6): # 6 blocks
                    for t in range(7): # 7 terms
                        c = u * 7 + t
                        # Fill U
                        for i in range(4):
                            U[i][c] = U_forms[t][i]
                