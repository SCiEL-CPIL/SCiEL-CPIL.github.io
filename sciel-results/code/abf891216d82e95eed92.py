<<<<<<< SEARCH
    # Local search on best candidate
    if best_points is not None and elapsed() < DEADLINE - 10:
        refined_pts, refined_s = local_search(best_points, max_iters=300)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    if best_points is None:
=======
    # Family H: Triangular lattice arrangements
    for scale in [1, 2, 3]:
        if elapsed() > DEADLINE:
            break
        pts = []
        for row in range(6):
            for col in range(5):
                x = col * scale + (row % 2) * scale // 2
                y = row * scale * 3 // 4
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) >= 30:
            pts = pts[:30]
            if len(set(pts)) == 30:
                s = score_of(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts[:]

    # Family I: Concentric patterns
    for r in range(1, 10):
        if elapsed() > DEADLINE:
            break
        pts = [(0, 0)]
        for angle in range(0, 360, 360 // min(8, 30 - len(pts))):
            import math
            x = int(round(r * math.cos(math.radians(angle))))
            y = int(round(r * math.sin(math.radians(angle))))
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.append((x, y))
        # Fill remaining with nearby points
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if len(pts) >= 30:
                    break
                if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            if len(pts) >= 30:
                break
        if len(pts) == 30 and len(set(pts)) == 30:
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Local search on best candidate
    if best_points is not None and elapsed() < DEADLINE - 15:
        refined_pts, refined_s = local_search(best_points, max_iters=800, step_range=3)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    # Targeted search for specific areas
    if best_points is not None and elapsed() < DEADLINE - 10:
        # Find the most common area in current best
        cnt = doubled_areas_fast(best_points)
        if cnt:
            target = max(cnt, key=cnt.get)
            targeted_pts, targeted_count = targeted_search(target, best_points, max_iters=300)
            if targeted_count > best_score:
                best_score = targeted_count
                best_points = targeted_pts

    # More local search iterations if time permits
    if best_points is not None and elapsed() < DEADLINE - 5:
        refined_pts, refined_s = local_search(best_points, max_iters=500, step_range=2)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    if best_points is None:
>>>>>>> REPLACE
