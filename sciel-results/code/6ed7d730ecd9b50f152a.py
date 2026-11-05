                            # Update V
                            v_val = vf[idx]
                            if v_val != 0:
                                for r in range(2):
                                    for c in range(2):
                                        row = jo + a * 2 + r
                                        ccol = ko + b * 2 + c
                                        V[row * p + ccol][col] += v_val
