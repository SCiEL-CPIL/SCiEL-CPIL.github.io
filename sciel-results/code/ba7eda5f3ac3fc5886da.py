# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy approach.
    
    For each point x, we need at least one direction v such that all other 48 points
    on the line through x with direction v are in N.
    
    We greedily add points to N, always choosing the point that satisfies the most
    currently-unsatisfied point-direction constraints.
    """
    p = 7
    q = 49
    alpha_sq = 3

    def pt_to_idx(xa, xb, ya, yb):
        return xa * q * q + xb * q + ya * q + yb

    def idx_to_pt(idx):
        yb = idx % q
        ya = (idx // q) % q
        xb = (idx // (q * q)) % q
        xa = (idx // (q * q * q)) % q
        return (xa, xb, ya, yb)

    # Directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    directions.append((0, 1, 0, 0))
    num_dirs = len(directions)

    # Precompute line points
    line_points = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = pt_to_idx(xa, xb, ya, yb)
                    for di, (vxa, vxb, vya, vyb) in enumerate(directions):
                        others = []
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0:
                                    continue
                                tvxa = (ta * vxa + tb * vxb * alpha_sq) % p
                                tvxb = (ta * vxb + tb * vxa) % p
                                tvya = (ta * vya + tb * vyb * alpha_sq) % p
                                tvyb = (ta * vyb + tb * vya) % p
                                new_xa = (xa + tvxa) % p
                                new_xb = (xb + tvxb) % p
                                new_ya = (ya + tvya) % p
                                new_yb = (yb + tvyb) % p
                                others.append(pt_to_idx(new_xa, new_xb, new_ya, new_yb))
                        line_points[(x_idx, di)] = others

    # Build reverse map
    reverse_map = [[] for _ in range(2401)]
    for (x_idx, di), others in line_points.items():
        for y in others:
            reverse_map[y].append((x_idx, di))

    in_N = np.zeros(2401, dtype=bool)
    constraint_count = np.zeros((2401, num_dirs), dtype=np.int32)
    x_satisfied = np.zeros(2401, dtype=bool)
    unsatisfied_count = 2401

    deadline = time.time() + 580

    # Greedy: for each step, find the point that maximizes the number of newly satisfied constraints
    # A constraint (x, di) is satisfied when constraint_count[x, di] == 48
    # Adding point y satisfies constraint (x, di) if y in line_points[(x, di)] and constraint_count[x, di] == 47

    while unsatisfied_count > 0:
        if time.time() > deadline:
            break

        # Compute marginal gain for each candidate point
        best_y = -1
        best_gain = -1
        for y in range(2401):
            if in_N[y]:
                continue
            gain = 0
            for (x_idx, di) in reverse_map[y]:
                if constraint_count[x_idx, di] == 47 and not x_satisfied[x_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_y = y

        if best_y == -1:
            break

        in_N[best_y] = True

        # Update constraint counts
        for (x_idx, di) in reverse_map[best_y]:
            constraint_count[x_idx, di] += 1
            if constraint_count[x_idx, di] == 48 and not x_satisfied[x_idx]:
                x_satisfied[x_idx] = True
                unsatisfied_count -= 1

    # Build result
    points = []
    for idx in range(2401):
        if in_N[idx]:
            xa, xb, ya, yb = idx_to_pt(idx)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
