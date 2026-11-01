    if time.time() < deadline - 10:
        U_str, V_str, W_str = strassen()
        ok, val = _check_decomposition(U_str, V_str, W_str)
        if ok:
            # Verify half-integer property
            all_half = True
            for mat in [U_str, V_str, W_str]:
                for row in mat:
                    for x in row:
                        if not _is_half_integer(x):
                            all_half = False
                            break
                    if not all_half:
                        break
                if not all_half:
                    break
            if all_half:
                best_rank = 7
                best_solution = {"U": U_str, "V": V_str, "W": W_str}
                return best_solution
