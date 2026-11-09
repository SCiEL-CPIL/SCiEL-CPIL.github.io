<<<<<<< SEARCH
def _local_search(points, max_iterations=5000, time_limit=60.0):
    """Hill-climbing local search with incremental scoring and target-area focus."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    best_counter = _area_counter(best_points)

    if best_score == 0:
        return best_points

    target_area = best_counter.most_common(1)[0][0]

    occupied = set(best_points)

    # Neighborhood sizes to try in sequence
    neighborhoods = [
        [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-15, 16) for dy in range(-15, 16) if (dx, dy) != (0, 0)],
    ]

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 3:
            break

        improved = False
        nh_size = iteration % 3
        if time.time() - start_time > time_limit - 10:
            nh_size = min(nh_size, 1)

        for idx in range(30):
            if time.time() - start_time > time_limit - 3:
                break

            px, py = best_points[idx]
            for dx, dy in neighborhoods[nh_size]:
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue

                new_point = (nx, ny)
                if new_point in occupied:
                    continue

                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)

                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    target_area = best_counter.most_common(1)[0][0]
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try a larger random restart within the current neighborhood
            for _ in range(10):
                if time.time() - start_time > time_limit - 3:
                    break
                idx = np.random.randint(0, 30)
                px, py = best_points[idx]
                dx = np.random.randint(-15, 16)
                dy = np.random.randint(-15, 16)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    target_area = best_counter.most_common(1)[0][0]
                    improved = True
                    break
            if not improved:
                break

    return best_points
=======
def _local_search(points, max_iterations=5000, time_limit=60.0):
    """Hill-climbing local search with incremental scoring and target-area focus."""
    start_time = time.time()
    best_points = list(points)
    best_score = _score(best_points)
    best_counter = _area_counter(best_points)

    if best_score == 0:
        return best_points

    target_area = best_counter.most_common(1)[0][0]
    target_count = best_counter[target_area]

    occupied = set(best_points)

    # Neighborhood sizes to try in sequence
    neighborhoods = [
        [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-15, 16) for dy in range(-15, 16) if (dx, dy) != (0, 0)],
    ]

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 3:
            break

        improved = False
        nh_size = iteration % 3
        if time.time() - start_time > time_limit - 10:
            nh_size = min(nh_size, 1)

        for idx in range(30):
            if time.time() - start_time > time_limit - 3:
                break

            px, py = best_points[idx]
            for dx, dy in neighborhoods[nh_size]:
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue

                new_point = (nx, ny)
                if new_point in occupied:
                    continue

                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)

                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    target_area = best_counter.most_common(1)[0][0]
                    target_count = best_counter[target_area]
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try a larger random restart within the current neighborhood
            for _ in range(10):
                if time.time() - start_time > time_limit - 3:
                    break
                idx = np.random.randint(0, 30)
                px, py = best_points[idx]
                dx = np.random.randint(-15, 16)
                dy = np.random.randint(-15, 16)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                new_points = list(best_points)
                new_points[idx] = new_point
                s, new_counter = _fast_score(new_points, best_counter, idx, (px, py), new_point)
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    best_counter = new_counter
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    target_area = best_counter.most_common(1)[0][0]
                    target_count = best_counter[target_area]
                    improved = True
                    break
            if not improved:
                break

    # Phase 2: Target-area focused search
    if time.time() - start_time < time_limit - 5:
        best_points = _target_area_search(best_points, best_counter, target_area, target_count,
                                          time_limit - (time.time() - start_time) - 3)
        new_score = _score(best_points)
        if new_score > best_score:
            best_score = new_score
            best_counter = _area_counter(best_points)
            target_area = best_counter.most_common(1)[0][0]
            target_count = best_counter[target_area]

    return best_points


def _target_area_search(points, counter, target_area, target_count, time_limit):
    """Focused search to maximize the count of a specific target area."""
    start_time = time.time()
    best_points = list(points)
    best_target_count = target_count
    best_counter = dict(counter)
    occupied = set(best_points)

    # Try moves that specifically increase target area count
    small_nh = [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)]
    med_nh = [(dx, dy) for dx in range(-10, 11) for dy in range(-10, 11) if (dx, dy) != (0, 0)]

    for iteration in range(3000):
        if time.time() - start_time > time_limit:
            break

        improved = False
        nh = small_nh if iteration % 3 != 2 else med_nh

        for idx in range(30):
            if time.time() - start_time > time_limit:
                break

            px, py = best_points[idx]
            for dx, dy in nh:
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue

                new_point = (nx, ny)
                if new_point in occupied:
                    continue

                new_points = list(best_points)
                new_points[idx] = new_point
                new_target_count, new_counter = _target_area_score(
                    new_points, best_counter, idx, (px, py), new_point, target_area
                )

                # Accept if target count improves, or if max multiplicity improves
                new_max = max(new_counter.values()) if new_counter else 0
                if new_target_count > best_target_count or (
                    new_target_count == best_target_count and new_max > max(best_counter.values())
                ):
                    best_points = new_points
                    best_counter = new_counter
                    best_target_count = new_target_count
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Random perturbation
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                idx = random.randint(0, 29)
                px, py = best_points[idx]
                dx = random.randint(-12, 12)
                dy = random.randint(-12, 12)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = px + dx, py + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                new_points = list(best_points)
                new_points[idx] = new_point
                new_target_count, new_counter = _target_area_score(
                    new_points, best_counter, idx, (px, py), new_point, target_area
                )
                new_max = max(new_counter.values()) if new_counter else 0
                if new_target_count > best_target_count or (
                    new_target_count == best_target_count and new_max > max(best_counter.values())
                ):
                    best_points = new_points
                    best_counter = new_counter
                    best_target_count = new_target_count
                    occupied.discard((px, py))
                    occupied.add(new_point)
                    improved = True
                    break
            if not improved:
                break

    return best_points
>>>>>>> REPLACE

