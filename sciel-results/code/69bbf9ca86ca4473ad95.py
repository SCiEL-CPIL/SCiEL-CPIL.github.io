<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    # Budget: ~40s for SA phase
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 40:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=25000, seed=init_idx * 7 + 1,
            deadline=deadline - 38
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 38:
            break

        # Best-fit local search (targeted moves)
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=4000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 36
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 34:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=2000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 32
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 30:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=8000,
            seed=777, deadline=deadline - 28
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 24:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=4000, step_size=0.002,
            seed=999, deadline=deadline - 22
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Bubble relaxation
    if time.time() < deadline - 22:
        bubble_pos, bubble_obj = _bubble_relaxation(
            best_pos, deadline=deadline - 20,
            n_iter=2000, dt=0.001
        )
        if bubble_obj > best_obj:
            best_pos = bubble_pos
            best_obj = bubble_obj

    # Phase 5: Gradient-based polish (smooth surrogate) - multiple passes
    if time.time() < deadline - 18:
        polished = _gradient_polish(best_pos, deadline - 16, beta=40.0)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 6: High-precision final polish (progressive beta) - main refinement
    if time.time() < deadline - 10:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 4)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 7: Another gradient polish after high-precision
    if time.time() < deadline - 6:
        polished2 = _gradient_polish(best_pos, deadline - 5, beta=80.0)
        if len(polished2) == 3:
            p_pol2, r_pol2, obj_pol2 = polished2
            if obj_pol2 > best_obj:
                best_pos = p_pol2
                best_obj = obj_pol2

    # Phase 8: Best-fit pass after gradient polish
    if time.time() < deadline - 4:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=4000,
            seed=555, deadline=deadline - 3
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 9: Final fine-tuning local search with very small steps
    if time.time() < deadline - 2:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=2000, step_size=0.0006,
            seed=12345, deadline=deadline - 1
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 10: Ultra-fine polish
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(99999)
        for _ in range(400):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.0002, 0.0002)
            dy = rng.uniform(-0.0002, 0.0002)
            nx = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            ny = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_obj = _objective_single_move(best_pos, i, nx, ny)
            if new_obj > best_obj + 1e-13:
                best_pos[i, 0] = nx
                best_pos[i, 1] = ny
                best_obj = new_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    # Budget: ~35s for SA phase (reduced to allow more polish time)
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 35:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=20000, seed=init_idx * 7 + 1,
            deadline=deadline - 33
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 33:
            break

        # Best-fit local search (targeted moves)
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=3000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 31
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 29:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=1500, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 28
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 25:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=6000,
            seed=777, deadline=deadline - 23
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 20:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=3000, step_size=0.002,
            seed=999, deadline=deadline - 18
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Radius growth - explicitly increase radii
    if time.time() < deadline - 18:
        grow_pos, grow_r, grow_obj = _grow_radii(
            best_pos, deadline=deadline - 16
        )
        if grow_obj > best_obj:
            best_pos = grow_pos
            best_obj = grow_obj

    # Phase 5: Gradient-based polish (smooth surrogate)
    if time.time() < deadline - 16:
        polished = _gradient_polish(best_pos, deadline - 14, beta=40.0)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 6: High-precision final polish (progressive beta) - main refinement
    if time.time() < deadline - 10:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 4)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 7: Another gradient polish after high-precision
    if time.time() < deadline - 6:
        polished2 = _gradient_polish(best_pos, deadline - 5, beta=80.0)
        if len(polished2) == 3:
            p_pol2, r_pol2, obj_pol2 = polished2
            if obj_pol2 > best_obj:
                best_pos = p_pol2
                best_obj = obj_pol2

    # Phase 8: Best-fit pass after gradient polish
    if time.time() < deadline - 4:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=3000,
            seed=555, deadline=deadline - 3
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 9: Final fine-tuning local search with very small steps
    if time.time() < deadline - 2:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=1500, step_size=0.0006,
            seed=12345, deadline=deadline - 1
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 10: Ultra-fine polish
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(99999)
        for _ in range(300):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.0002, 0.0002)
            dy = rng.uniform(-0.0002, 0.0002)
            nx = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            ny = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_obj = _objective_single_move(best_pos, i, nx, ny)
            if new_obj > best_obj + 1e-13:
                best_pos[i, 0] = nx
                best_pos[i, 1] = ny
                best_obj = new_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result
>>>>>>> REPLACE
