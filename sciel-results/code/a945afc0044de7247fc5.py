            mask = ex_val >= cont_val
            if np.any(mask):
                last_idx = np.max(np.where(mask)[0])
                if last_idx < k: # There is a next node
                    # Interpolate
                    # f(S) = Ex(S) - Cont(S)
                    # We want root of f(S) = 0 between S[last] and S[last+1]
                    f0 = ex_val[last_idx] - cont_val[last_idx]
                    f1 = ex_val[last_idx + 1] - cont_val[last_idx + 1]
                    if f0 != f1:
                        # Linear interp
                        # S = S0 + (0 - f0) * (S1 - S0) / (f1 - f0)
                        S0_node = S_vals[last_idx]
                        S1_node = S_vals[last_idx + 1]
                        B = S0_node - f0 * (S1_node - S0_node) / (f1 - f0)
                    else:
                        B = S_vals[last_idx]
                else:
                    B = S_vals[last_idx]
                boundaries[k // steps_per_interval - 1] = B
            else:
                boundaries[k // steps_per_interval - 1] = 0.0
