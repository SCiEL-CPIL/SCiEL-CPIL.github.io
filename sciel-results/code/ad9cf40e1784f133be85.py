# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_7[alpha], alpha^2 = 3).
    Returns an integer array of shape (k, 2, 2) listing points of N.
    """
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3

    # Field element index: idx = a + b * p, represents a + b*alpha
    # Point index: pidx = x_idx * q + y_idx

    # Precompute field multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for a1 in range(p):
        for b1 in range(p):
            i = a1 + b1 * p
            for a2 in range(p):
                for b2 in range(p):
                    j = a2 + b2 * p
                    ca = (a1 * a2 + b1 * b2 * k_const) % p
                    cb = (a1 * b2 + a2 * b1) % p
                    mul_table[i, j] = ca + cb * p

    # Build all lines
    # Vertical lines: x = const, indexed 0..48
    # Slanted lines: y = x*m + b, indexed 49 + m*49 + b
    total_lines = q + q * q  # 2450
    total_points = q * q  # 2401

    # line_points[L] = list of point indices on line L
    line_points = [[] for _ in range(total_lines)]
    # point_lines[x] = list of line indices through point x
    point_lines = [[] for _ in range(total_points)]

    # Vertical lines: x = const
    for x_val in range(q):
        line_id = x_val
        for y_val in range(q):
            pidx = x_val * q + y_val
            line_points[line_id].append(pidx)
            point_lines[pidx].append(line_id)

    # Slanted lines: y = x * m + b
    for m in range(q):
        for b in range(q):
            line_id = q + m * q + b
            for t in range(q):
                x_val = t
                y_val = (mul_table[m, t] + b) % q
                pidx = x_val * q + y_val
                line_points[line_id].append(pidx)
                point_lines[pidx].append(line_id)

    # Each point is on exactly 50 lines (1 vertical + 49 slanted)
    # Each line has exactly 49 points

    # Greedy algorithm to find large excluded set S
    # safe_count[y] = number of lines L through y such that (L \ {y}) ∩ S = ∅
    # Initially all 50 lines are safe for each point.
    # When we add x to S, for each line L through x, L becomes unsafe for all
    # other points y on L (since x ∈ S and x ≠ y).

    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # We'll try multiple orderings and keep the best result
    best_S = set()
    best_score = 0

    # Try natural order first
    orderings = [list(range(total_points))]
    # Add reverse order
    orderings.append(list(range(total_points - 1, -1, -1)))

    # Add random orderings
    rng = np.random.RandomState(42)
    for _ in range(20):
        orderings.append(rng.permutation(total_points).tolist())

    for order in orderings:
        if time.time() > deadline - 10:
            break

        S = set()
        safe_count = [50] * total_points
        # For efficiency, track which lines are safe for each point using a set
        safe_lines = [set(pl) for pl in point_lines]

        for x in order:
            if time.time() > deadline - 10:
                break

            # Check if adding x to S would violate any point
            # A violation occurs if there exists y ≠ x such that:
            # - safe_count[y] == 1, and
            # - the only safe line for y is a line L that contains x
            valid = True
            for L in point_lines[x]:
                for y in line_points[L]:
                    if y == x:
                        continue
                    if safe_count[y] == 1 and L in safe_lines[y]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                S.add(x)
                # Update: for each line L through x, L is no longer safe for
                # any other point y on L
                for L in point_lines[x]:
                    for y in line_points[L]:
                        if y == x:
                            continue
                        if L in safe_lines[y]:
                            safe_lines[y].discard(L)
                            safe_count[y] -= 1

        score = total_points - len(S)
        if score > best_score:
            best_score = score
            best_S = S.copy()
            # Print progress for debugging
            # print(f"Ordering {len(orderings)}: |S| = {len(S)}, score = {score}")

    # Build the output: N = all points not in S
    points = []
    for x in range(total_points):
        if x not in best_S:
            x_idx = x // q
            y_idx = x % q
            xa_a = x_idx % p
            xa_b = x_idx // p
            ya_a = y_idx % p
            ya_b = y_idx // p
            points.append([[xa_a, xa_b], [ya_a, ya_b]])

    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
