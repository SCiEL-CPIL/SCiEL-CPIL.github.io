<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    # Build lattice positions
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Lattice enumeration with multiple outer orientations
    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0,
                      7.5, -7.5, 22.5, -22.5, 37.5, -37.5]
    
    # Try all combinations of 4 from ring2
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if time.time() > deadline:
                break
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _compute_outer_for_positions(full_xy, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    n_hex = len(best_positions)

    # Phase 2: Nonlinear optimization
    if time.time() < deadline - 10:
        def objective(state):
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / SQRT3

            # Penalty for overlaps
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation(
                        (state[3*i], state[3*i+1]), state[3*i+2],
                        (state[3*j], state[3*j+1]), state[3*j+2])
                    if v > 0:
                        penalty += v ** 2
            return side + 300.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 1200, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'initial_simplex': None})
            if result.fun < best_side + 50:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 3: Optimize outer angle
    if time.time() < deadline - 15:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0,
                       2.5, 7.5, 12.5, 17.5, 22.5, 27.5,
                       -2.5, -7.5, -12.5, -17.5, -22.5, -27.5]:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Phase 4: High-precision polish
    if time.time() < deadline - 8:
        def polish_objective(state):
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / SQRT3
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation(
                        (state[3*i], state[3*i+1]), state[3*i+2],
                        (state[3*j], state[3*j+1]), state[3*j+2])
                    if v > 0:
                        penalty += v ** 2
            return side + 800.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(polish_objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_side + 10:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 5: Second optimization round with refined outer angle
    if time.time() < deadline - 10:
        def objective2(state):
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
                        A_ub.append([-nx, -ny, -1])
                        b_ub.append(-vx * nx - vy * ny)
            A_ub.append([0, 0, -1])
            b_ub.append(0)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if not res.success:
                return 1e6
            t = res.x[2]
            side = t * 2.0 / SQRT3
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation(
                        (state[3*i], state[3*i+1]), state[3*i+2],
                        (state[3*j], state[3*j+1]), state[3*j+2])
                    if v > 0:
                        penalty += v ** 2
            return side + 400.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(objective2, init_state, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_side + 20:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Final verification
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(
                    best_positions, best_angles, best_oangle)
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        for _ in range(300):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
=======
def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    # Build lattice positions
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Lattice enumeration with multiple outer orientations
    # Use a small gap factor to ensure strict disjointness
    GAP = 1.0005
    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0,
                      7.5, -7.5, 22.5, -22.5, 37.5, -37.5,
                      3.0, -3.0, 10.0, -10.0, 20.0, -20.0]

    # Try all combinations of 4 from ring2
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        # Apply gap factor
        full_xy_gapped = [(x * GAP, y * GAP) for x, y in full_xy]

        for oangle in oangles_phase1:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(full_xy_gapped, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy_gapped[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = [(x * GAP, y * GAP) for x, y in full_xy]
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    n_hex = len(best_positions)

    # Phase 2: Smooth surrogate optimization using log-sum-exp
    # This is much faster than LP-based objective since we can compute
    # the smooth objective without solving an LP each time
    if time.time() < deadline - 20:
        def smooth_objective(state):
            """Smooth objective: enclosing side + overlap penalty."""
            positions_local = [(state[3*i], state[3*i+1]) for i in range(n_hex)]
            angles_local = [state[3*i+2] for i in range(n_hex)]
            
            # Smooth enclosing side
            side_smooth, _ = _smooth_enclosing_side(positions_local, angles_local, best_oangle, temperature=0.05)
            
            # Overlap penalty
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation(
                        positions_local[i], angles_local[i],
                        positions_local[j], angles_local[j])
                    if v > 0:
                        penalty += v ** 2
            return side_smooth + 200.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(smooth_objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_side + 100:
                final_state = result.x
                positions = [(final_state[3*i], final_state[3*i+1]) for i in range(n_hex)]
                angles = [final_state[3*i+2] for i in range(n_hex)]
                hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(n_hex)]
                if _all_disjoint(hexes):
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 3: Fine outer angle search with minimize_scalar
    if time.time() < deadline - 25:
        def outer_angle_obj(oangle):
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            return side
        
        # Coarse search
        best_ao = best_oangle
        for trial in range(30):
            if time.time() > deadline:
                break
            ao = best_oangle + (trial - 15) * 1.0
            s = outer_angle_obj(ao)
            if s < best_side:
                best_side = s
                best_oangle = ao
                center, _ = _compute_outer_for_positions(best_positions, best_angles, ao)
                best_center = center
        
        # Fine search around best
        if time.time() < deadline - 10:
            try:
                res = minimize_scalar(outer_angle_obj, bounds=(best_oangle - 5, best_oangle + 5),
                                      method='bounded', options={'xatol': 1e-6, 'maxiter': 50})
                if res.success:
                    ao = res.x
                    center, side = _compute_outer_for_positions(best_positions, best_angles, ao)
                    if side < best_side:
                        best_side = side
                        best_oangle = ao
                        best_center = center
            except Exception:
                pass

    # Phase 4: High-precision LP-based polish
    if time.time() < deadline - 15:
        def lp_objective(state):
            positions_local = [(state[3*i], state[3*i+1]) for i in range(n_hex)]
            angles_local = [state[3*i+2] for i in range(n_hex)]
            center, side = _compute_outer_for_positions(positions_local, angles_local, best_oangle)
            if math.isinf(side):
                return 1e6
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation(
                        positions_local[i], angles_local[i],
                        positions_local[j], angles_local[j])
                    if v > 0:
                        penalty += v ** 2
            return side + 500.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(lp_objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_side + 5:
                final_state = result.x
                positions = [(final_state[3*i], final_state[3*i+1]) for i in range(n_hex)]
                angles = [final_state[3*i+2] for i in range(n_hex)]
                hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(n_hex)]
                if _all_disjoint(hexes):
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 5: Joint optimization of positions + outer angle
    if time.time() < deadline - 20:
        def joint_objective(state):
            # First 3*n_hex values: positions and angles
            # Last 3 values: outer center x, y, angle
            positions_local = [(state[3*i], state[3*i+1]) for i in range(n_hex)]
            angles_local = [state[3*i+2] for i in range(n_hex)]
            ox = state[3*n_hex]
            oy = state[3*n_hex+1]
            oangle = state[3*n_hex+2]
            
            # Compute enclosing side with given outer center
            # Use LP with fixed center
            all_verts = []
            for i in range(n_hex):
                verts = _hex_vertices(positions_local[i][0], positions_local[i][1], 1.0, angles_local[i])
                all_verts.extend(verts)
            
            normals = []
            for k in range(6):
                a = math.radians(oangle + 30 + 60 * k)
                normals.append((math.cos(a), math.sin(a)))
            
            # t = max over all (v - center) . n
            max_t = 0.0
            for vx, vy in all_verts:
                for nx, ny in normals:
                    val = (vx - ox) * nx + (vy - oy) * ny
                    if val > max_t:
                        max_t = val
            
            side = max_t * 2.0 / SQRT3
            
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_separation(
                        positions_local[i], angles_local[i],
                        positions_local[j], angles_local[j])
                    if v > 0:
                        penalty += v ** 2
            return side + 300.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        init_state.extend([best_center[0], best_center[1], best_oangle])

        try:
            result = minimize(joint_objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_side + 20:
                final_state = result.x
                positions = [(final_state[3*i], final_state[3*i+1]) for i in range(n_hex)]
                angles = [final_state[3*i+2] for i in range(n_hex)]
                ox = final_state[3*n_hex]
                oy = final_state[3*n_hex+1]
                oangle = final_state[3*n_hex+2]
                hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(n_hex)]
                if _all_disjoint(hexes):
                    center, side = _compute_outer_for_positions(positions, angles, oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
                        best_oangle = oangle
        except Exception:
            pass

    # Final verification
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(
                    best_positions, best_angles, best_oangle)
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        for _ in range(300):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
>>>>>>> REPLACE
