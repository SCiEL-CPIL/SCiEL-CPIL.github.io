import numpy as np
import time
import random


def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    total = q * q  # 2401 points
    
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin
    
    # Precompute all lines: for each slope m (49 slopes) and intercept c (49),
    # plus vertical lines (49). Total = 49*49 + 49 = 2401 lines.
    # Each line has exactly 49 points.
    # 
    # For each point, we need to know which lines it belongs to (50 lines).
    # For each line, we store the 49 point indices.
    
    # Points indexed 0..2400: point_id = x * 49 + y where x,y in 0..48
    # x = xa + xb*alpha, y = ya + yb*alpha
    
    # Precompute line membership:
    # lines: list of arrays, each of shape (49,) containing point indices
    # point_lines: for each point, list of line indices
    
    lines = []
    point_lines = [[] for _ in range(total)]
    
    for m in range(q):
        for c in range(q):
            line_pts = []
            for x in range(q):
                # y = m*x + c
                y = (m * x + c) % q
                pt = x * q + y
                line_pts.append(pt)
            line_idx = len(lines)
            lines.append(np.array(line_pts, dtype=np.int32))
            for pt in line_pts:
                point_lines[pt].append(line_idx)
    
    for c in range(q):
        # vertical line: x = c
        line_pts = []
        for y in range(q):
            pt = c * q + y
            line_pts.append(pt)
        line_idx = len(lines)
        lines.append(np.array(line_pts, dtype=np.int32))
        for pt in line_pts:
            point_lines[pt].append(line_idx)
    
    n_lines = len(lines)
    
    # Verify: each point should be on exactly 50 lines
    # (49 slopes + 1 vertical)
    
    def greedy_solve(seed, time_budget):
        rng = random.Random(seed)
        in_S = np.zeros(total, dtype=bool)
        line_counts = np.zeros(n_lines, dtype=np.int32)
        
        # Order: random permutation of all points
        order = list(range(total))
        rng.shuffle(order)
        
        for pt in order:
            # Check: can we add this point to S?
            # Adding pt to S means incrementing line_counts for all lines through pt.
            # We must ensure no line through pt exceeds count 48.
            # If a line already has count 48, adding pt would make it 49 -> bad.
            # So pt can be added iff all lines through it have count < 48.
            can_add = True
            for li in point_lines[pt]:
                if line_counts[li] >= 48:
                    can_add = False
                    break
            if can_add:
                in_S[pt] = True
                for li in point_lines[pt]:
                    line_counts[li] += 1
                # Optional: if any line just reached 48, we could try to
                # un-add other points on that line to "free up" capacity,
                # but this is complex. Keep simple for now.
        
        return in_S
    
    def local_search(in_S, rng, time_budget):
        """Try to improve by removing points from S that are 'expensive' 
        (i.e., their removal allows adding multiple new points)."""
        in_S = in_S.copy()
        line_counts = np.zeros(n_lines, dtype=np.int32)
        for li in range(n_lines):
            line_counts[li] = np.sum(in_S[lines[li]])
        
        # Try to add random points not in S
        not_in_S = np.where(~in_S)[0]
        if len(not_in_S) == 0:
            return in_S
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            idx = rng.randint(0, len(not_in_S) - 1)
            pt = not_in_S[idx]
            # Check if pt can be added
            can_add = True
            for li in point_lines[pt]:
                if line_counts[li] >= 48:
                    can_add = False
                    break
            if can_add:
                in_S[pt] = True
                for li in point_lines[pt]:
                    line_counts[li] += 1
                # Remove from not_in_S
                not_in_S = not_in_S[not_in_S != pt]
                improved = True
        
        # Try to remove points from S that block many additions
        # For each point in S, compute how many lines it "saturates"
        in_S_pts = np.where(in_S)[0]
        rng.shuffle(in_S_pts.tolist())
        
        for pt in in_S_pts:
            if time.time() >= time_budget:
                break
            # Count how many lines through pt have count == 48
            blocking_lines = 0
            for li in point_lines[pt]:
                if line_counts[li] == 48:
                    blocking_lines += 1
            if blocking_lines == 0:
                continue
            # Try removing pt
            in_S[pt] = False
            for li in point_lines[pt]:
                line_counts[li] -= 1
            # Now try to add as many points as possible
            old_count = np.sum(in_S)
            # Re-add pt temporarily? No, just try to fill
            # Try adding all non-S points that can now fit
            not_in_S = np.where(~in_S)[0]
            added = 0
            for cand in not_in_S:
                can_add = True
                for li in point_lines[cand]:
                    if line_counts[li] >= 48:
                        can_add = False
                        break
                if can_add:
                    in_S[cand] = True
                    for li in point_lines[cand]:
                        line_counts[li] += 1
                    added += 1
            if added > blocking_lines:
                # Net gain: we removed 1, added 'added'
                # Keep the removal
                improved = True
            else:
                # Revert
                in_S[pt] = True
                for li in point_lines[pt]:
                    line_counts[li] += 1
                # Revert the additions
                # This is getting complex; simplify by not doing local search
                # for now and rely on multiple restarts
        
        return in_S
    
    best_S = None
    best_size = -1
    
    # Run multiple greedy restarts
    seed = 42
    while time.time() < deadline:
        in_S = greedy_solve(seed, deadline)
        size = np.sum(in_S)
        if size > best_size:
            best_size = size
            best_S = in_S.copy()
        
        # Also try a few local improvements on the best found so far
        if seed % 3 == 0 and best_S is not None and time.time() < deadline - 10:
            improved_S = local_search(best_S, random.Random(seed), deadline - 5)
            improved_size = np.sum(improved_S)
            if improved_size > best_size:
                best_size = improved_size
                best_S = improved_S.copy()
        
        seed += 1
    
    if best_S is None:
        # Fallback: empty excluded set -> N is everything
        best_S = np.zeros(total, dtype=bool)
    
    # N = all points NOT in S
    in_N = ~best_S
    pts = np.where(in_N)[0]
    result = np.zeros((len(pts), 2, 2), dtype=np.int64)
    for i, pt in enumerate(pts):
        x = pt // q
        y = pt % q
        xa = x % p
        xb = x // p
        ya = y % p
        yb = y // p
        result[i] = [[xa, xb], [ya, yb]]
    
    return result
