import time
import numpy as np
from collections import Counter, defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 87.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def score_of(pts):
        """Compute max multiplicity of any non-zero doubled area among all C(30,3) triples."""
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        max_count = 0
        areas = []
        for i in range(n - 2):
            diffs = P[i+1:] - P[i]
            m = len(diffs)
            if m < 2:
                continue
            dx = diffs[:, 0]
            dy = diffs[:, 1]
            cross_mat = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            tri = np.triu(cross_mat, k=1)
            nz = tri[tri != 0]
            if len(nz) > 0:
                areas.append(nz)
        if not areas:
            return 0
        all_areas = np.concatenate(areas)
        if len(all_areas) == 0:
            return 0
        vals, cnts = np.unique(all_areas, return_counts=True)
        return int(cnts.max())

    def score_with_counts(pts):
        """Return (max_count, target_area, area_counter) for a point set."""
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        areas = []
        for i in range(n - 2):
            diffs = P[i+1:] - P[i]
            m = len(diffs)
            if m < 2:
                continue
            dx = diffs[:, 0]
            dy = diffs[:, 1]
            cross_mat = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            tri = np.triu(cross_mat, k=1)
            nz = tri[tri != 0]
            if len(nz) > 0:
                areas.append(nz)
        if not areas:
            return 0, 0, Counter()
        all_areas = np.concatenate(areas)
        if len(all_areas) == 0:
            return 0, 0, Counter()
        cnt = Counter(all_areas.tolist())
        max_area, max_count = cnt.most_common(1)[0]
        return max_count, max_area, cnt

    def delta_score_move(pts, idx, new_pt):
        """
        Compute the change in score (max multiplicity) when moving point idx to new_pt.
        Returns (delta, new_max_count, new_target_area, new_counter).
        Uses incremental update: remove triples involving idx, add new triples.
        """
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        
        # Compute all areas first to get current state
        # Actually, for speed, let's compute the full area list but incrementally
        # For a single point move, only triples involving that point change.
        # Triples NOT involving idx remain the same.
        
        # Strategy: compute areas of triples NOT involving idx (unchanged),
        # then add areas of triples involving idx at new position.
        
        # This is still O(n^2) for the changed part, which is fine.
        
        # Build the new point set
        new_P = P.copy()
        new_P[idx] = np.array(new_pt)
        
        # Compute areas of triples NOT involving idx
        # These are triples (i,j,k) where none equals idx
        other_indices = [i for i in range(n) if i != idx]
        unchanged_areas = []
        for ii in range(len(other_indices)):
            i = other_indices[ii]
            # Triples (i, j, k) with j,k > i and j,k != idx
            # Simpler: iterate over all triples from other_indices
            for jj in range(ii+1, len(other_indices)):
                j = other_indices[jj]
                for kk in range(jj+1, len(other_indices)):
                    k = other_indices[kk]
                    # doubled area = |det(P[j]-P[i], P[k]-P[i])|
                    v1 = P[j] - P[i]
                    v2 = P[k] - P[i]
                    cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if cross != 0:
                        unchanged_areas.append(cross)
        
        # Compute areas of triples involving idx (at new position)
        changed_areas = []
        for j in range(n):
            if j == idx:
                continue
            for k in range(j+1, n):
                if k == idx:
                    continue
                if j == idx or k == idx:
                    # One of them is idx
                    other = j if k == idx else k
                    # Triangle (idx, j, k) -> (idx, other, ...)
                    # Actually need to be careful with ordering
                    pass
            # Let me redo this more carefully
            break
        
        # Simpler approach: just compute all areas with the new point set
        # But only for triples involving idx, since others are unchanged
        # Let me just compute the full thing but use numpy for speed
        
        # Full recompute with numpy (fast enough for n=30)
        all_areas_new = []
        for i in range(n - 2):
            diffs = new_P[i+1:] - new_P[i]
            m = len(diffs)
            if m < 2:
                continue
            dx = diffs[:, 0]
            dy = diffs[:, 1]
            cross_mat = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            tri = np.triu(cross_mat, k=1)
            nz = tri[tri != 0]
            if len(nz) > 0:
                all_areas_new.append(nz)
        
        if not all_areas_new:
            return 0, 0, 0, Counter()
        
        all_areas_arr = np.concatenate(all_areas_new)
        if len(all_areas_arr) == 0:
            return 0, 0, 0, Counter()
        
        cnt_new = Counter(all_areas_arr.tolist())
        max_area_new, max_count_new = cnt_new.most_common(1)[0]
        return max_count_new, max_area_new, cnt_new

    # Generate structured grid candidates
    def generate_grid_candidates():
        candidates = []
        
        # Various grid configurations: rows x cols
        configs = [
            (5, 6), (6, 5), (3, 10), (10, 3), (4, 8), (8, 4),
            (2, 15), (15, 2), (1, 30), (30, 1),
            (5, 5), (4, 7), (7, 4), (3, 9), (9, 3),
        ]
        
        for rows, cols in configs:
            if rows * cols != 30:
                continue
            # Center the grid
            x_start = -(cols - 1) // 2
            y_start = -(rows - 1) // 2
            pts = []
            for r in range(rows):
                for c in range(cols):
                    x = x_start + c
                    y = y_start + r
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
            if len(pts) == 30:
                candidates.append(pts)
        
        # Stretched grids (non-unit spacing)
        for rows, cols in [(5, 6), (6, 5), (3, 10), (10, 3)]:
            for sx in [1, 2, 3]:
                for sy in [1, 2, 3]:
                    x_start = -(cols - 1) * sx // 2
                    y_start = -(rows - 1) * sy // 2
                    pts = []
                    valid = True
                    for r in range(rows):
                        for c in range(cols):
                            x = x_start + c * sx
                            y = y_start + r * sy
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                pts.append((x, y))
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(pts) == 30:
                        candidates.append(pts)
        
        # Diagonal grids
        for diag in range(-5, 6):
            pts = []
            for x in range(15):
                y1 = x + diag
                y2 = x + diag + 1
                if -30 <= y1 <= 30:
                    pts.append((x, y1))
                if -30 <= y2 <= 30:
                    pts.append((x, y2))
            if len(pts) >= 30:
                # Deduplicate
                seen = set()
                unique_pts = []
                for p in pts:
                    if p not in seen:
                        seen.add(p)
                        unique_pts.append(p)
                if len(unique_pts) >= 30:
                    candidates.append(unique_pts[:30])
        
        return candidates

    # Targeted local search: for a given target area A, try to maximize count of triples with area A
    def targeted_search(initial_pts, target_area, max_iters, deadline):
        """
        Greedily move points to increase the count of triples with area == target_area.
        """
        pts = list(initial_pts)
        n = len(pts)
        
        def count_target_area(point_set):
            """Count triples with doubled area == target_area."""
            P = np.array(point_set, dtype=np.int64)
            count = 0
            for i in range(n - 2):
                diffs = P[i+1:] - P[i]
                m = len(diffs)
                if m < 2:
                    continue
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                cross_mat = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
                tri = np.triu(cross_mat, k=1)
                count += int(np.sum(tri == target_area))
            return count
        
        current_count = count_target_area(pts)
        best_count = current_count
        best_pts = list(pts)
        
        used_set = set(pts)
        no_improve = 0
        
        for iteration in range(max_iters):
            if elapsed() > deadline:
                break
            
            idx = random.randint(0, n - 1)
            old_pt = pts[idx]
            
            # Try to find a new position for pts[idx] that increases target area count
            best_new_pt = None
            best_new_count = current_count
            
            # Sample candidate positions
            candidates_pos = []
            # Nearby positions
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_pt[0] + dx, old_pt[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates_pos.append((nx, ny))
            # Some random positions
            for _ in range(5):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                candidates_pos.append((nx, ny))
            
            random.shuffle(candidates_pos)
            tested = 0
            for new_pt in candidates_pos:
                if tested > 40:  # Limit per iteration
                    break
                if new_pt in used_set or new_pt == old_pt:
                    continue
                tested += 1
                
                # Try the move
                pts[idx] = new_pt
                new_count = count_target_area(pts)
                
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_new_pt = new_pt
                else:
                    pts[idx] = old_pt
            
            if best_new_pt is not None:
                used_set.discard(old_pt)
                used_set.add(best_new_pt)
                pts[idx] = best_new_pt
                current_count = best_new_count
                no_improve = 0
                if current_count > best_count:
                    best_count = current_count
                    best_pts = list(pts)
            else:
                no_improve += 1
                if no_improve > 500:
                    break
        
        return best_pts, best_count

    # Main search
    random.seed(123)
    
    # Phase 1: Evaluate structured candidates
    grid_cands = generate_grid_candidates()
    
    best_score = 0
    best_points = None
    best_target_area = 0
    
    all_cands = list(grid_cands)
    
    # Add some random initial candidates
    for seed in range(5):
        random.seed(seed * 100 + 7)
        used = set()
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        all_cands.append(pts)
    
    for cand in all_cands:
        if len(cand) != 30:
            continue
        s, ta, _ = score_with_counts(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            best_target_area = ta
    
    # Phase 2: Targeted search from best candidates
    # Try multiple target areas
    target_areas_to_try = [best_target_area]
    # Also try some common small areas
    for a in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if a not in target_areas_to_try:
            target_areas_to_try.append(a)
    
    # Also get top areas from the best candidate
    if best_points is not None:
        _, _, cnt = score_with_counts(best_points)
        top_areas = [a for a, c in cnt.most_common(10)]
        for a in top_areas:
            if a not in target_areas_to_try:
                target_areas_to_try.append(a)
    
    # Run targeted search for each target area
    for ta in target_areas_to_try:
        if elapsed() > DEADLINE - 5:
            break
        
        # Start from best points found so far
        start_pts = list(best_points) if best_points else list(all_cands[0])
        
        # Also try from a few other good candidates
        start_variants = [start_pts]
        for cand in all_cands[:5]:
            if cand != start_pts:
                start_variants.append(list(cand))
        
        for sv in start_variants:
            if elapsed() > DEADLINE - 3:
                break
            
            result_pts, result_count = targeted_search(sv, ta, 2000, DEADLINE - 2)
            
            # Evaluate the full score of the result
            full_score, full_ta, _ = score_with_counts(result_pts)
            if full_score > best_score:
                best_score = full_score
                best_points = list(result_pts)
                best_target_area = full_ta
    
    # Phase 3: Final local search refinement (small moves)
    if elapsed() < DEADLINE - 3 and best_points is not None:
        current_pts = list(best_points)
        current_score = best_score
        used_set = set(current_pts)
        
        for _ in range(3000):
            if elapsed() > DEADLINE - 1:
                break
            
            idx = random.randint(0, 29)
            old_pt = current_pts[idx]
            
            best_new = None
            best_new_score = current_score
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_pt[0] + dx, old_pt[1] + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    npt = (nx, ny)
                    if npt in used_set:
                        continue
                    
                    current_pts[idx] = npt
                    s = score_of(current_pts)
                    
                    if s > best_new_score:
                        best_new_score = s
                        best_new = npt
                    else:
                        current_pts[idx] = old_pt
            
            if best_new is not None:
                used_set.discard(old_pt)
                used_set.add(best_new)
                current_pts[idx] = best_new
                current_score = best_new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_pts)
            else:
                current_pts[idx] = old_pt
    
    return best_points

