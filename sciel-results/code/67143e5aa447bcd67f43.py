<<<<<<< SEARCH
    # Final attempt: try to polish the rank-46 solution to see if it can be snapped to a lower-rank form
    if time.time() < deadline - 10:
        U_pol, V_pol, W_pol = gauss_newton_polish(U46, V46, W46, max_iter=20, tol=1e-12)
        valid, U_s, V_s, W_s = try_snap_validate(U_pol, V_pol, W_pol)
        if valid:
            # Check if we can reduce rank by zeroing out columns
            for r_target in [43, 44, 45]:
                if time.time() > deadline - 8:
                    break
                # Try to find a subset of columns that still works
                for _ in range(5):
                    if time.time() > deadline - 7:
                        break
                    mask = list(range(r_target))
                    U_sub = U_s[:, mask].copy()
                    V_sub = V_s[:, mask].copy()
                    W_sub = W_s[:, mask].copy()
                    if validate(U_sub, V_sub, W_sub):
                        return {
                            "U": [[float(x) for x in row] for row in U_sub],
                            "V": [[float(x) for x in row] for row in V_sub],
                            "W": [[float(x) for x in row] for row in W_sub]
                        }
=======
    # Final attempt: try to polish the rank-46 solution to see if it can be snapped to a lower-rank form
    if time.time() < deadline - 15:
        U_pol, V_pol, W_pol = gauss_newton_polish(U46, V46, W46, max_iter=30, tol=1e-14)
        valid, U_s, V_s, W_s = try_snap_validate(U_pol, V_pol, W_pol)
        if valid:
            # Check if we can reduce rank by zeroing out columns
            for r_target in [43, 44, 45]:
                if time.time() > deadline - 10:
                    break
                # Try to find a subset of columns that still works
                for _ in range(8):
                    if time.time() > deadline - 8:
                        break
                    mask = list(range(r_target))
                    U_sub = U_s[:, mask].copy()
                    V_sub = V_s[:, mask].copy()
                    W_sub = W_s[:, mask].copy()
                    if validate(U_sub, V_sub, W_sub):
                        return {
                            "U": [[float(x) for x in row] for row in U_sub],
                            "V": [[float(x) for x in row] for row in V_sub],
                            "W": [[float(x) for x in row] for row in W_sub]
                        }
                    # Also try shuffled subsets
                    np.random.seed(1234 + r_target)
                    perm = np.random.permutation(46)
                    mask2 = list(perm[:r_target])
                    U_sub2 = U_s[:, mask2].copy()
                    V_sub2 = V_s[:, mask2].copy()
                    W_sub2 = W_s[:, mask2].copy()
                    if validate(U_sub2, V_sub2, W_sub2):
                        return {
                            "U": [[float(x) for x in row] for row in U_sub2],
                            "V": [[float(x) for x in row] for row in V_sub2],
                            "W": [[float(x) for x in row] for row in W_sub2]
                        }
>>>>>>> REPLACE
