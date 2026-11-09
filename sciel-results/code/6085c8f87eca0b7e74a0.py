<<<<<<< SEARCH
    # Polish
    polished = polish(best_overall_positions, 15.0)

    # Fix overlaps
    fixed = fix_overlaps(polished, 5.0)

    # Optimize outer angle
    best_side, best_angle = optimize_outer_angle(fixed, 3.0)

    # Final validity check
    final_pen = sum(sat_overlap(fixed[i], fixed[j]) for i in range(N) for j in range(i + 1, N))
    if final_pen > 1e-4:
        # Fallback
        safe = make_honeycomb()
        best_side, best_angle = optimize_outer_angle(safe, 1.0)
        fixed = safe

    return fixed, (0.0, 0.0), best_side, best_angle
=======
    # Polish with more time
    polished = polish(best_overall_positions, 18.0)

    # Fix overlaps
    fixed = fix_overlaps(polished, 4.0)

    # Second polish after fixing overlaps
    if time.time() < start_time + (deadline - start_time) - 8.0:
        fixed = polish(fixed, 6.0)
        fixed = fix_overlaps(fixed, 2.0)

    # Optimize outer angle
    best_side, best_angle = optimize_outer_angle(fixed, 3.0)

    # Final exact local refinement: try small perturbations to reduce side
    if time.time() < start_time + (deadline - start_time) - 2.0:
        for _ in range(50):
            if time.time() > start_time + (deadline - start_time) - 2.0:
                break
            trial = [p[:] for p in fixed]
            idx = random.randint(0, N - 1)
            trial[idx][0] += random.gauss(0, 0.02)
            trial[idx][1] += random.gauss(0, 0.02)
            trial[idx][2] += random.gauss(0, 2.0)
            # Check validity
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap(trial[i], trial[j]) > 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                s, a = optimize_outer_angle(trial, 0.5)
                if s < best_side:
                    best_side, best_angle = s, a
                    fixed = trial

    # Final validity check
    final_pen = sum(sat_overlap(fixed[i], fixed[j]) for i in range(N) for j in range(i + 1, N))
    if final_pen > 1e-4:
        # Fallback
        safe = make_honeycomb()
        best_side, best_angle = optimize_outer_angle(safe, 1.0)
        fixed = safe

    return fixed, (0.0, 0.0), best_side, best_angle
>>>>>>> REPLACE
