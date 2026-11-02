    # Phase 2: Systematic rotation search to expose hidden dependencies
    if time.time() < deadline - 5:
        current_rank = U.shape[1]
        # We want to try rotations on the CURRENT U,V,W
        # If we find a reduction, we update U,V,W and restart the search from the new rank?
        # Or just continue?
        # Let's just do one pass of rotations on the current state.
        
        # To avoid excessive runtime, we limit the number of pairs if rank is large
        # But 36 is small.
        
        found_reduction = False
        while not found_reduction and time.time() < deadline - 5:
            found_reduction = True
            current_rank = U.shape[1]
            for col_a in range(current_rank):
                if time.time() > deadline - 5:
                    break
                for col_b in range(current_rank):
                    if col_a == col_b:
                        continue
                    if time.time() > deadline - 5:
                        break
                    for t in [-2, -1, -0.5, 0.5, 1, 2]:
                        if time.time() > deadline - 5:
                            break
                        U_rot, V_rot, W_rot = _try_column_rotation(U, V, W, col_a, col_b, t)
                        
                        # Check if any column is now removable
                        for col in range(current_rank):
                            if time.time() > deadline - 5:
                                break
                            redundant, _ = _check_column_redundant(U_rot, V_rot, W_rot, col)
                            if redundant:
                                mask = [i for i in range(current_rank) if i != col]
                                U_new = U_rot[:, mask]
                                V_new = V_rot[:, mask]
                                W_new = W_rot[:, mask]
                                if _verify_decomposition(U_new, V_new, W_new):
                                    U, V, W = U_new, V_new, W_new
                                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                    found_reduction = False
                                    break
                        if not found_reduction:
                            break
                    if not found_reduction:
                        break
                if not found_reduction:
                    break
