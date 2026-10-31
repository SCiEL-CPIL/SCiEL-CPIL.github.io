<<<<<<< SEARCH
    # --- Phase 2: Polish best result ---
    if best_result is not None and time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Expand radii
        x, y, r = expand_radii(x, y, r, w, h, time_budget=min(5.0, deadline - time.time() - 8))
        if is_valid_fast(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                best_result = z.copy()

        # Local position search
        if time.time() < deadline - 8:
            x, y, r = local_position_search(x, y, r, w, h, time_budget=min(8.0, deadline - time.time() - 5))
            if is_valid_fast(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = x
                    z[1::3][:N] = y
                    z[2::3][:N] = r
                    z[-1] = w
                    best_result = z.copy()

        # Final expansion
        if time.time() < deadline - 4:
            x, y, r = expand_radii(x, y, r, w, h, time_budget=min(3.0, deadline - time.time() - 2))
            if is_valid_fast(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    z = np.empty(3 * N + 1)
                    z[0::3][:N] = x
                    z[1::3][:N] = y
                    z[2::3][:N] = r
                    z[-1] = w
                    best_result = z.copy()
=======
    # --- Phase 2: Polish best result with high-precision passes ---
    if best_result is not None and time.time() < deadline - 12:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Multiple polish rounds with alternating position/radius optimization
        for polish_round in range(3):
            if time.time() > deadline - 10:
                break

            # Expand radii
            budget_exp = min(4.0, deadline - time.time() - 8)
            if budget_exp > 1.0:
                x, y, r = expand_radii(x, y, r, w, h, time_budget=budget_exp)
                if is_valid_fast(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        z = np.empty(3 * N + 1)
                        z[0::3][:N] = x
                        z[1::3][:N] = y
                        z[2::3][:N] = r
                        z[-1] = w
                        best_result = z.copy()

            # Local position search
            budget_pos = min(6.0, deadline - time.time() - 6)
            if budget_pos > 1.0:
                x, y, r = local_position_search(x, y, r, w, h, time_budget=budget_pos)
                if is_valid_fast(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        z = np.empty(3 * N + 1)
                        z[0::3][:N] = x
                        z[1::3][:N] = y
                        z[2::3][:N] = r
                        z[-1] = w
                        best_result = z.copy()

            # SLSQP refinement if time permits
            budget_slsqp = min(6.0, deadline - time.time() - 5)
            if budget_slsqp > 2.0:
                z = np.empty(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w
                z_refined = slsqp_refine(z, time_budget=budget_slsqp)
                if z_refined is not None:
                    xr = z_refined[0::3][:N]
                    yr = z_refined[1::3][:N]
                    rr = z_refined[2::3][:N]
                    wr = z_refined[-1]
                    hr = 2.0 - wr
                    if is_valid_fast(xr, yr, rr, wr, hr):
                        val_r = float(np.sum(rr))
                        if val_r > best_value:
                            best_value = val_r
                            best_result = z_refined.copy()
                            x, y, r, w, h = xr, yr, rr, wr, hr

            # Final expansion after SLSQP
            budget_final = min(3.0, deadline - time.time() - 3)
            if budget_final > 0.5:
                x, y, r = expand_radii(x, y, r, w, h, time_budget=budget_final)
                if is_valid_fast(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        z = np.empty(3 * N + 1)
                        z[0::3][:N] = x
                        z[1::3][:N] = y
                        z[2::3][:N] = r
                        z[-1] = w
                        best_result = z.copy()
>>>>>>> REPLACE

