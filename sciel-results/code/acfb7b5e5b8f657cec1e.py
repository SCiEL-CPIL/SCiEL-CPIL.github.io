def construct():
    points = [(0, 0)]
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    step_len = 1
    dir_idx = 0
    while len(points) < 32:
        # Two segments of length step_len
        for _ in range(2):
            dx, dy = dirs[dir_idx]
            for _ in range(step_len):
                if len(points) >= 32:
                    break
                last_a, last_b = points[-1]
                points.append((last_a + dx, last_b + dy))
            dir_idx = (dir_idx + 1) % 6
        step_len += 1
    return [[a, b] for a, b in points]
