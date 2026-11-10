import numpy as np
import time

def nikodym_set() -> np.ndarray:
    n_points = 2401
    start_time = time.time()
    deadline = 580  # safety margin before 630s kill

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def idx_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)

    # Precompute all 50 direction line structures
    # For direction d: 49 lines, each with 49 points
    # line_id[d][point] = which of the 49 lines the point is on
    # line_points[d][line_id] = array of 49 point indices

    line_id = np.zeros((50, n_points), dtype=np.int32)
    line_points = []  # line_points[d][l] = array of 49 int64

    for d in range(50):
        if d < 49:
            ma, mb = divmod(d, 7)
        else:
            ma, mb = -1, -1  # vertical

        # Compute line ID for each point
        # For (1,ma,mb): c1 = (ya - xa*ma - 3*xb*mb) % 7, c2 = (yb - xa*mb - xb*ma) % 7
        # For vertical: c1 = xa, c2 = xb
        lids = np.zeros(n_points, dtype=np.int32)
        for idx in range(n_points):
            xa, xb, ya, yb = idx_to_point(idx)
            if d < 49:
                c1 = (ya - xa * ma - 3 * xb * mb) % 7
                c2 = (yb - xa * mb - xb * ma) % 7
            else:
                c1 = xa
                c2 = xb
            lids[idx] = c1 * 7 + c2
        line_id[d] = lids

        # Group points by line
        groups = [[] for _ in range(49)]
        for idx in range(n_points):
            groups[lids[idx]].append(idx)
        line_points.append([np.array(g, dtype=np.int64) for g in groups])

    # Witness-based greedy removal
    # State: in_N (bool array), n_in_N[d][l] = count of N-points on line l in direction d
    # For each point, track witness direction (one direction where the line is clean)
    # A point q is "witnessed" by direction d if:
    #   in_N[q] and n_in_N[d][line_id[d][q]] == 49, or
    #   not in_N[q] and n_in_N[d][line_id[d][q]] == 48

    in_N = np.ones(n_points, dtype=bool)

    # Start with a structured initial set: remove points on a conic
    # Use the unital: 2*xa + ya^2 - 3*yb^2 = 0 in F_7
    # Remove half of those points (where ya is a QR)
    QR = {0, 1, 2, 4}
    for xa in range(7):
        for xb in range(7):
            for ya in range(7):
                for yb in range(7):
                    if (2 * xa + ya * ya - 3 * yb * yb) % 7 == 0 and ya in QR:
                        in_N[point_to_idx(xa, xb, ya, yb)] = False

    # Compute n_in_N
    n_in_N = np.zeros((50, 49), dtype=np.int32)
    for d in range(50):
        for l in range(49):
            n_in_N[d, l] = np.sum(in_N[line_points[d][l]])

    # For each point, find a witness direction
    # witness[q] = direction d that witnesses q, or -1 if none
    witness = np.full(n_points, -1, dtype=np.int32)

    def find_witness(q):
        lid = line_id[:, q]
        for d in range(50):
            if in_N[q]:
                if n_in_N[d, lid[d]] == 49:
                    return d
            else:
                if n_in_N[d, lid[d]] == 48:
                    return d
        return -1

    # Initialize witnesses
    for q in range(n_points):
        witness[q] = find_witness(q)

    # If any point has no witness, our initial set is invalid
    # (shouldn't happen with the unital construction, but check)
    if np.any(witness < 0):
        # Fallback: start from all points
        in_N[:] = True
        for d in range(50):
            for l in range(49):
                n_in_N[d, l] = 49
        for q in range(n_points):
            witness[q] = 0  # any direction works when all points are in N

    # Greedy removal with witness tracking
    # A point p can be removed if:
    # 1. After removal, p itself still has a witness (as non-N point)
    # 2. No other point loses its ONLY witness (if witness[q] was the direction whose line contained p,
    #    and n_in_N drops to a value that no longer witnesses q)
    #    More precisely: for each q whose witness direction d has line containing p,
    #    check if q still has a witness after the update.

    # For efficiency, precompute for each point p, which (d, line) pairs contain p
    # point_lines[p] = list of (d, line_id[d][p]) for all d
    # Actually line_id[d][p] gives the line of p in direction d.

    # When we remove p:
    #   For each d: n_in_N[d, line_id[d][p]] -= 1
    #   Points affected: all q on line line_id[d][p] in direction d (for each d)
    #   That's 50 * 49 = 2450 point-direction pairs, but unique points <= 2401

    # Strategy: try removing points in order of "safety"
    # A point is safer to remove if fewer points depend on it as their sole witness

    # For now, use a simple heuristic: try all points, in a randomized-ish order
    # Use multiple passes

    removed_count = 0
    max_passes = 200
    for pass_num in range(max_passes):
        if time.time() - start_time > deadline:
            break

        progress = False
        # Try to remove points
        # Order: try points that are in N
        candidates = np.where(in_N)[0]
        if len(candidates) == 0:
            break

        for p in candidates:
            if time.time() - start_time > deadline:
                break
            if not in_N[p]:
                continue

            # Simulate removal
            affected = set()
            old_nin = {}
            for d in range(50):
                lid = line_id[d, p]
                old_nin[d] = n_in_N[d, lid]
                n_in_N[d, lid] -= 1
                # All points on this line are affected
                for q in line_points[d][lid]:
                    affected.add(q)

            in_N[p] = False

            # Check if all affected points still have witnesses
            valid = True
            for q in affected:
                if q == p:
                    # p is now not in N; check if it has a witness
                    w = find_witness(q)
                    if w < 0:
                        valid = False
                        break
                else:
                    # Check if q still has a witness
                    w = find_witness(q)
                    if w < 0:
                        valid = False
                        break

            if valid:
                # Commit the removal
                in_N[p] = False  # already set
                # Update witnesses for affected points
                for q in affected:
                    witness[q] = find_witness(q)
                removed_count += 1
                progress = True
            else:
                # Rollback
                in_N[p] = True
                for d in range(50):
                    n_in_N[d, line_id[d, p]] = old_nin[d]

        if not progress:
            break

    # Build result
    points = []
    for idx in range(n_points):
        if in_N[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)
