<<<<<<< SEARCH
    # ---------- Final polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 5:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
        # Extra micro-polish: coordinate descent on radii only
        if time.time() < deadline - 2:
            for _ in range(3):
                improved = False
                for i in range(N):
                    if time.time() > deadline - 1:
                        break
                    # Try increasing radius i slightly
                    for growth in [1.002, 1.001, 1.0005]:
                        new_r = radii_p[i] * growth
                        max_r = min(best_pos[i, 0], 1.0 - best_pos[i, 0],
                                    best_pos[i, 1], 1.0 - best_pos[i, 1])
                        if new_r > max_r:
                            new_r = max_r * 0.999
                        radii_p[i] = new_r
                        if _is_valid(best_pos, radii_p):
                            improved = True
                            break
                        else:
                            radii_p[i] /= growth
                if not improved:
                    break
            if _is_valid(best_pos, radii_p):
                s = np.sum(radii_p)
                if s > best_score:
                    best_score = s
                    best_radii = radii_p.copy()
=======
    # ---------- Final polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 8:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()
        # Extra micro-polish: aggressive coordinate descent
        if time.time() < deadline - 1:
            p_mp = best_pos.copy()
            r_mp = best_radii.copy()
            for _ in range(10):
                improved = False
                for i in range(N):
                    if time.time() > deadline - 0.3:
                        break
                    # Try growing radius i
                    for growth in [1.005, 1.002, 1.001, 1.0005, 1.0001]:
                        new_r = r_mp[i] * growth
                        max_r = min(p_mp[i, 0], 1.0 - p_mp[i, 0],
                                    p_mp[i, 1], 1.0 - p_mp[i, 1])
                        if new_r > max_r:
                            new_r = max_r * 0.9999
                        old_r = r_mp[i]
                        r_mp[i] = new_r
                        if _is_valid(p_mp, r_mp):
                            improved = True
                            break
                        else:
                            r_mp[i] = old_r
                    # Try nudging position to create room
                    for dx in [0.002, -0.002, 0.001, -0.001]:
                        for dy in [0.002, -0.002, 0.001, -0.001]:
                            ox, oy = p_mp[i, 0], p_mp[i, 1]
                            p_mp[i, 0] = ox + dx
                            p_mp[i, 1] = oy + dy
                            if p_mp[i, 0] < r_mp[i] or p_mp[i, 0] > 1 - r_mp[i]:
                                p_mp[i, 0] = ox
                                continue
                            if p_mp[i, 1] < r_mp[i] or p_mp[i, 1] > 1 - r_mp[i]:
                                p_mp[i, 1] = oy
                                continue
                            if _is_valid(p_mp, r_mp):
                                # Try to grow all circles
                                any_grown = False
                                for j in range(N):
                                    max_rj = min(p_mp[j, 0], 1.0 - p_mp[j, 0],
                                                 p_mp[j, 1], 1.0 - p_mp[j, 1])
                                    if r_mp[j] < max_rj * 0.9999:
                                        r_mp[j] = max_rj * 0.9999
                                        any_grown = True
                                improved = True
                                break
                            else:
                                p_mp[i, 0] = ox
                                p_mp[i, 1] = oy
                        if improved:
                            break
                if not improved:
                    break
            if _is_valid(p_mp, r_mp):
                s = np.sum(r_mp)
                if s > best_score:
                    best_score = s
                    best_pos = p_mp.copy()
                    best_radii = r_mp.copy()
>>>>>>> REPLACE

