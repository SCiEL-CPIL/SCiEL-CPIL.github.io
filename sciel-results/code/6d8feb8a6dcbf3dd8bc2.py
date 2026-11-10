# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

DEADLINE = 620.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    Points are [[x_a, x_b], [y_a, y_b]] representing x = x_a + x_b*alpha,
    y = y_a + y_b*alpha.
    
    Strategy: Use the unital construction to get a valid Nikodym set with
    complement size 148, then attempt to add more points to the complement
    S while maintaining validity using incremental witness tracking.
    """
    p = 7
    k = 3  # alpha^2 = 3

    # Precompute all field elements as (a, b) pairs
    # Field element: a + b*alpha, where a, b in 0..6
    
    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)
    
    def field_add(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 + a2) % p, (b1 + b2) % p)
    
    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)
    
    # Enumerate all points in F_49^2
    # Each point is ((xa, xb), (ya, yb))
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    n_total = len(all_points)  # 2401
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Directions: (1, m) for all m in F_49, plus (0, 1)
    # A direction v = (vx, vy) where vx, vy are field elements
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)
    
    n_dirs = len(directions)  # 50
    
    # Precompute line neighbors for each direction
    # line_neighbors[d][i] = list of other point indices on the same line as i in direction d
    # Also, for incremental updates, we need to know for a point p, which points q are on the same line as p in direction d.
    # This is symmetric.
    
    line_neighbors = []
    # Also store the line groups to quickly identify affected points
    # For a given direction d and point p, the affected points are line_neighbors[d][p]
    
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        
        # Map from line_id to list of points on that line
        line_map = {}
        
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            # Compute line invariant
            if vx == (1, 0):  # direction (1, m) where m = vy
                # Line: y - m*x = const
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:  # direction (0, 1)
                # Line: x = const
                c = x
            
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        
        # For each point, the other points on its line
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        
        line_neighbors.append(neigh)
    
    # Base construction: unital-based
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)
    # squares = {0, 1, 2, 4}
    
    # S_base: points in unital where y_a is NOT a square
    S = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S.add(pt_idx)
    
    # Initialize witness counts
    # witness[d][x] = 1 if line through x in direction d has no other points in S, else 0
    witness = [[0] * n_total for _ in range(n_dirs)]
    # sat_count[x] = number of directions d such that witness[d][x] == 1
    sat_count = [0] * n_total
    
    # Compute initial witness state
    for d_idx in range(n_dirs):
        for pt_idx in range(n_total):
            # Check if any neighbor in this direction is in S
            has_s = False
            for other_idx in line_neighbors[d_idx][pt_idx]:
                if other_idx in S:
                    has_s = True
                    break
            if not has_s:
                witness[d_idx][pt_idx] = 1
                sat_count[pt_idx] += 1
    
    # Verify base is valid (all sat_count > 0)
    # assert all(c > 0 for c in sat_count)
    
    # Candidates: points not currently in S
    candidates = [i for i in range(n_total) if i not in S]
    
    # Shuffle candidates for variety (but deterministic seed)
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    added_count = 0
    for cand in candidates:
        if _time_left() < 100:
            break
        
        # Try adding cand to S
        # 1. Check if cand itself remains satisfied (or becomes satisfied? No, it's in S, so it doesn't need to be satisfied? 
        # Wait, the definition: "every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N."
        # So x in S also needs a witness line.
        # If we add cand to S, cand is in S. It needs a line where all OTHER points are in N (not in S).
        # So cand needs sat_count[cand] > 0 after the update?
        # Actually, if cand is in S, the condition for cand is: exists v such that L(cand, v) \ {cand} subset N.
        # This is exactly witness[v][cand] == 1.
        # So yes, cand must have sat_count > 0.
        
        # We need to update the state for all points affected by adding cand to S.
        # Affected points: cand itself, and all neighbors of cand in all directions.
        # For a neighbor q of cand in direction d:
        #   The line L(q, d) is the same as L(cand, d).
        #   Before: L(q, d) \ {q} had some set of S points.
        #   After: L(q, d) \ {q} has {cand} added to the S set.
        #   So witness[d][q] might change from 1 to 0.
        
        # For cand itself:
        #   Before: cand was in N. witness[d][cand] indicated if L(cand, d) \ {cand} was clean.
        #   After: cand is in S. The condition for cand is still "L(cand, d) \ {cand} subset N".
        #   The set of other points on the line hasn't changed. So witness[d][cand] remains the same?
        #   Wait. If cand was in N, and we move it to S, does the "cleanliness" of the line for *other* points change? Yes.
        #   Does the "cleanliness" for *cand* change?
        #   The line is the set of points. The set of points in S on that line (excluding the point itself) changes.
        #   For q != cand on the line: S_line(q) = S_line(q) \cup {cand} (if cand not already in S_line(q)).
        #   For cand: S_line(cand) = S_line(cand) \cup {cand} (conceptually, but we exclude self).
        #   Actually, witness[d][cand] checks if L(cand, d) \ {cand} contains any S.
        #   Before: cand was not in S. So S_line(cand) = (S \cap L) \ {cand} = S \cap L.
        #   After: cand is in S. So S_line(cand) = (S \cup {cand}) \cap L \ {cand} = (S \cap L) \cup ({cand} \cap L) \ {cand} = S \cap L.
        #   So the set of *other* points in S on the line is unchanged!
        #   Therefore, witness[d][cand] does NOT change when cand is added to S.
        #   However, we must ensure that cand *has* a witness. i.e., sum(witness[d][cand]) > 0.
        
        # So the changes are:
        # 1. For each direction d, for each neighbor q of cand:
        #    If witness[d][q] was 1, it might become 0 because cand (which is now in S) is on the line.
        #    Specifically, if cand was NOT in S before, adding it means the line now has one more S point.
        #    So if witness[d][q] == 1, it means there were NO other S points. Now there is one (cand). So it becomes 0.
        #    If witness[d][q] == 0, it stays 0.
        
        # 2. For cand itself:
        #    witness[d][cand] is unchanged.
        #    But we need to check if cand is satisfied.
        
        # Let's track changes to revert if invalid.
        changes = []
        valid = True
        
        # Check cand's satisfaction first (it doesn't change, so just check current state)
        # Wait, if cand was in N, it was already satisfied (since base is valid).
        # So sat_count[cand] > 0 is already true.
        # But what if adding cand to S somehow invalidates cand?
        # As derived above, witness[d][cand] is unchanged. So sat_count[cand] is unchanged.
        # So cand remains satisfied.
        
        # Now check neighbors
        for d_idx in range(n_dirs):
            neighbors = line_neighbors[d_idx][cand]
            for q in neighbors:
                if witness[d_idx][q] == 1:
                    # This witness is lost
                    witness[d_idx][q] = 0
                    sat_count[q] -= 1
                    changes.append((d_idx, q))
                    if sat_count[q] == 0:
                        valid = False
                        # We can break early, but need to revert
                        # It's easier to just break and revert everything recorded so far
                        # But we might have modified others.
                        # Let's just break out of loops and revert.
                        break
            if not valid:
                break
        
        if valid:
            S.add(cand)
            added_count += 1
        else:
            # Revert changes
            for d_idx, q in changes:
                witness[d_idx][q] = 1
                sat_count[q] += 1
            # Note: cand is NOT added to S, so no other reverts needed.
    
    # Build final result
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])
    
    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END
