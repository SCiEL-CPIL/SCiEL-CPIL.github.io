import math
import time
import random
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = A * x1 + B * y1
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _build_lines(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _score_from_lines(lines):
    score = 0
    for key, indices in lines.items():
        cnt = len(indices)
        if cnt >= 5:
            return -1, False
        elif cnt == 4:
            score += 1
    return score, True

def _score(pts):
    lines = _build_lines(pts)
    return _score_from_lines(lines)

def _delta_score(pts, p_idx, old_pos, new_pos, lines):
    """Compute new score after moving point p_idx from old_pos to new_pos.
    Only recompute lines involving p_idx."""
    # Collect lines that involve p_idx
    affected_keys = set()
    for j in range(16):
        if j == p_idx:
            continue
        key_old = _line_key(old_pos, pts[j])
        key_new = _line_key(new_pos, pts[j])
        affected_keys.add(key_old)
        affected_keys.add(key_new)
    
    # For each affected line, count how many points (other than p_idx) are on it
    # We need to rebuild counts for affected lines only
    # Strategy: for each affected key, count points on that line (excluding p_idx)
    new_lines = dict(lines)  # shallow copy of dict; sets will be mutated
    
    # Remove p_idx from all lines it was on (with old_pos)
    for j in range(16):
        if j == p_idx:
            continue
        key_old = _line_key(old_pos, pts[j])
        if key_old in new_lines:
            new_lines[key_old].discard(p_idx)
            if not new_lines[key_old]:
                del new_lines[key_old]
    
    # Add p_idx to all lines it's on (with new_pos)
    for j in range(16):
        if j == p_idx:
            continue
        key_new = _line_key(new_pos, pts[j])
        if key_new not in new_lines:
            new_lines[key_new] = set()
        new_lines[key_new].add(p_idx)
        new_lines[key_new].add(j)
    
    # Also need to handle: lines that p_idx was on with old_pos but not with new_pos
    # and lines that p_idx is on with new_pos but wasn't with old_pos
    # The above handles this via discard/add
    
    # Check for 5+ collinear
    for key, indices in new_lines.items():
        if len(indices) >= 5:
            return -1, False
    
    # Count 4-collinear
    score = 0
    for key, indices in new_lines.items():
        if len(indices) == 4:
            score += 1
    
    return score, True

def _generate_initial(seed=0):
    rng = random.Random(seed)
    # Start with 4x4 grid but perturb slightly to break symmetries
    pts = []
    for x in range(4):
        for y in range(4):
            px = x + rng.choice([0, 0, 0, 1, -1])
            py = y + rng.choice([0, 0, 0, 1, -1])
            pts.append([px, py])
    seen = set()
    unique_pts = []
    for p in pts:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append(p)
    while len(unique_pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        tp = (x, y)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append([x, y])
    return unique_pts[:16]

def _generate_structured_start(seed=0):
    """Generate a structured starting configuration from a 5x5 or 6x6 grid subset."""
    rng = random.Random(seed)
    # Pick 16 points from a larger grid to create many 4-point lines
    grid_size = rng.choice([5, 6])
    all_pts = []
    for x in range(grid_size):
        for y in range(grid_size):
            all_pts.append((x, y))
    # Shuffle and pick 16
    rng.shuffle(all_pts)
    pts = [list(p) for p in all_pts[:16]]
    return pts

def _local_search(cur_pts, cur_score, deadline, max_iters=200):
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        lines = _build_lines(cur_pts)
        cur_score, cur_valid = _score_from_lines(lines)
        if not cur_valid:
            # Try to fix by moving points that cause 5-collinear
            for key, indices in lines.items():
                if len(indices) >= 5:
                    # Move one of the points
                    for idx in indices:
                        if time.time() > deadline:
                            return cur_pts, cur_score, False
                        old_pos = tuple(cur_pts[idx])
                        # Try random nearby positions
                        found = False
                        for attempt in range(20):
                            xi = old_pos[0] + random.randint(-3, 3)
                            yi = old_pos[1] + random.randint(-3, 3)
                            if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                                cand = (xi, yi)
                                if cand != old_pos and cand not in set(map(tuple, cur_pts)):
                                    new_score, new_valid = _delta_score(cur_pts, idx, old_pos, cand, lines)
                                    if new_valid:
                                        cur_pts[idx] = list(cand)
                                        lines = _build_lines(cur_pts)
                                        cur_score, cur_valid = _score_from_lines(lines)
                                        found = True
                                        break
                        if found:
                            improved = True
                            break
                    break
        
        if time.time() > deadline:
            return cur_pts, cur_score, False
        
        three_lines = {k: v for k, v in lines.items() if len(v) == 3}
        
        for p_idx in range(16):
            if time.time() > deadline:
                return cur_pts, cur_score, False
            
            current_pos = tuple(cur_pts[p_idx])
            candidates = set()
            
            # Intersections of pairs of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            for i in range(len(tl_keys)):
                for j in range(i + 1, len(tl_keys)):
                    A1, B1, C1 = tl_keys[i]
                    A2, B2, C2 = tl_keys[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            # Points on 3-lines extended
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
                idxs = list(v)
                p1 = cur_pts[idxs[0]]
                p2 = cur_pts[idxs[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                for t in range(-4, 5):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # Random nearby
            for _ in range(12):
                xi = current_pos[0] + random.randint(-5, 5)
                yi = current_pos[1] + random.randint(-5, 5)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            # Also try intersections involving 2-lines to create new 4-lines
            two_lines = {k: v for k, v in lines.items() if len(v) == 2 and p_idx not in v}
            tl_keys2 = [k for k in two_lines.keys() if p_idx not in two_lines[k]]
            # Limit to avoid too many combinations
            if len(tl_keys2) > 30:
                tl_keys2 = tl_keys2[:30]
            for i in range(len(tl_keys2)):
                for j in range(i + 1, len(tl_keys2)):
                    A1, B1, C1 = tl_keys2[i]
                    A2, B2, C2 = tl_keys2[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            best_new_pos = None
            best_new_score = cur_score
            current_set = set(map(tuple, cur_pts))
            
            for cand in candidates:
                if cand == current_pos or cand in current_set:
                    continue
                
                new_score, new_valid = _delta_score(cur_pts, p_idx, current_pos, cand, lines)
                
                if new_valid and new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
            
            if best_new_pos is not None:
                cur_pts[p_idx] = list(best_new_pos)
                cur_score = best_new_score
                improved = True
                lines = _build_lines(cur_pts)  # Rebuild after move
    
    return cur_pts, cur_score, improved

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_pts = None
    best_score = -1
    
    # Try multiple initial seeds with different strategies
    for seed in range(15):
        if time.time() > deadline:
            break
        
        if seed < 8:
            cur_pts = _generate_initial(seed)
        else:
            cur_pts = _generate_structured_start(seed)
        
        cur_score, cur_valid = _score(cur_pts)
        
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        if cur_valid:
            cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
    
    # If still no good result, use simple grid
    if best_pts is None or best_score < 10:
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, y])
        best_pts = pts
        best_score, _ = _score(best_pts)
    
    return best_pts
