        if current_N == 0:
            # Find any move to nonzero
            for i in range(30):
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val: continue
                    delta = (new_val - old_val) * weights[i]
                    new_N = current_N + delta
                    if new_N != 0:
                        coeffs[i] = new_val
                        current_N = new_N
                        break
                if current_N != 0:
                    break
            # Now proceed with minimization
