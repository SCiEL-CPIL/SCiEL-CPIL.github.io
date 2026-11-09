<<<<<<< SEARCH
    # --- Strategy 1: Physics Simulation with Hexagonal Lattice Start ---
    for spacing in [1.7, 1.6, 1.5]:
        if time.time() > deadline - 30:
            break
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline - 25:
                break
            
            init_inner = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, spacing)
                init_inner.append([x, y, 0.0])
            
            # Center cluster
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            
            # Estimate initial outer side
            max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in centered_init)
            est_side = max_r * 1.5 + 2.0
            
            # Run simulation
            sim_result = simulate(centered_init, (0.0, 0.0), est_side, outer_angle, dt=0.02, steps=300)
            
            # Re-center result
            cx_mean = sum(p[0] for p in sim_result) / len(sim_result)
            cy_mean = sum(p[1] for p in sim_result) / len(sim_result)
            centered_sim = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in sim_result]
            
            side, valid = evaluate(centered_sim, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_sim, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 2: Random Starts ---
    if time.time() < deadline - 40:
        rng = np.random.default_rng(42)
        for trial in range(10):
            if time.time() > deadline - 20:
                break
            
            init_inner = []
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                a = rng.uniform(0, 360)
                init_inner.append([x, y, a])
            
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            
            max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in centered_init)
            est_side = max_r * 1.5 + 2.0
            
            sim_result = simulate(centered_init, (0.0, 0.0), est_side, 0.0, dt=0.02, steps=300)
            
            cx_mean = sum(p[0] for p in sim_result) / len(sim_result)
            cy_mean = sum(p[1] for p in sim_result) / len(sim_result)
            centered_sim = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in sim_result]
            
            side, valid = evaluate(centered_sim, (0.0, 0.0), 0.0)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_sim, (0.0, 0.0), side, 0.0)
=======
    # --- Strategy 1: Physics Simulation with Hexagonal Lattice Start ---
    for spacing in [1.7, 1.6, 1.5]:
        if time.time() > deadline - 30:
            break
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline - 25:
                break
            
            init_inner = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, spacing)
                init_inner.append([x, y, 0.0])
            
            # Center cluster
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            
            # Estimate initial outer side
            max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in centered_init)
            est_side = max_r * 1.5 + 2.0
            
            # Run simulation
            sim_result = simulate(centered_init, (0.0, 0.0), est_side, outer_angle, dt=0.02, steps=300)
            
            # Re-center result
            cx_mean = sum(p[0] for p in sim_result) / len(sim_result)
            cy_mean = sum(p[1] for p in sim_result) / len(sim_result)
            centered_sim = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in sim_result]
            
            # Refine with gradient optimization
            refined = optimize_positions(centered_sim, (0.0, 0.0), outer_angle, max_iters=200, lr=0.03)
            cx_mean = sum(p[0] for p in refined) / len(refined)
            cy_mean = sum(p[1] for p in refined) / len(refined)
            centered_refined = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in refined]
            
            side, valid = evaluate(centered_refined, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_refined, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 2: Random Starts with Gradient Refinement ---
    if time.time() < deadline - 40:
        rng = np.random.default_rng(42)
        for trial in range(8):
            if time.time() > deadline - 20:
                break
            
            init_inner = []
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                a = rng.uniform(0, 360)
                init_inner.append([x, y, a])
            
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            
            max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in centered_init)
            est_side = max_r * 1.5 + 2.0
            
            sim_result = simulate(centered_init, (0.0, 0.0), est_side, 0.0, dt=0.02, steps=200)
            
            cx_mean = sum(p[0] for p in sim_result) / len(sim_result)
            cy_mean = sum(p[1] for p in sim_result) / len(sim_result)
            centered_sim = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in sim_result]
            
            # Refine with gradient optimization
            refined = optimize_positions(centered_sim, (0.0, 0.0), 0.0, max_iters=200, lr=0.03)
            cx_mean = sum(p[0] for p in refined) / len(refined)
            cy_mean = sum(p[1] for p in refined) / len(refined)
            centered_refined = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in refined]
            
            side, valid = evaluate(centered_refined, (0.0, 0.0), 0.0)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_refined, (0.0, 0.0), side, 0.0)
>>>>>>> REPLACE
