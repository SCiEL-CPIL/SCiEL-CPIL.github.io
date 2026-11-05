import time
import random
from itertools import combinations

def _collinear3(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
            c = dx * y1 - dy * x1
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return True
    return False

def _count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3free_subset(points, deadline=None):
    n = len(points)
    # Precompute collinear masks
    # mask[i] is a bitmask of points collinear with i
    # Actually, we need: for a set S, if i,j in S, then no k in S can be collinear with i,j.
    # Let's use the pair-based mask: mask[i][j] has bits for k collinear with i,j.
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            mask = 0
            for k in range(n):
                if k == i or k == j: continue
                if _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask
            collinear_mask[j][i] = mask

    best = [0]
    
    # Order points by degree (number of collinear pairs involving them) to prune better?
    # Or just standard B&B.
    # Heuristic: Try adding points that are "safe" (don't conflict) first?
    # Or try adding points that are "conflicting" first to fail fast?
    # Usually, for Max Independent Set (which this is similar to, on the conflict graph of pairs),
    # we want to branch on high-degree vertices.
    
    def dfs(current_mask, candidates_mask, current_size):
        if deadline and time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        if candidates_mask == 0:
            return
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates_mask).count('1') <= best[0]:
            return
        
        # Pick a candidate
        # Heuristic: Pick the one with most conflicts in candidates?
        # Let's just pick the first one for simplicity, or random?
        # Random might help avoid bad local minima in search tree?
        # But B&B is deterministic if order is fixed.
        # Let's try picking the point with max degree in the induced subgraph of candidates.
        # Degree of i in candidates: number of j in candidates such that i,j conflict?
        # Conflict: k in current_mask collinear with i,j? No.
        # If we add i, we must remove all j in candidates that form a triple with i and some k in current?
        # No, if we add i, we must ensure no existing pair (j,k) in current is collinear with i.
        # And no new pair (i,j) in new current will be collinear with any k in new current.
        
        # The check `can_add` in the original code is:
        # if (collinear_mask[i][j] & current_mask) != 0: return False
        # This checks if i forms a triple with any pair in current.
        # It does NOT check if i forms a triple with another candidate j that is also added later?
        # Wait, the DFS adds points one by one.
        # If we add i, we update current_mask.
        # Then for next candidate j, we check if j forms a triple with any pair in current_mask (which now includes i).
        # So the logic is correct.
        
        # To prune better, we should pick i such that it eliminates many candidates.
        # If i is added, any j in candidates that is collinear with i and some k in current_mask is removed.
        # Also, if i and j are added, they must not be collinear with any k in current_mask.
        
        # Let's stick to the simple LSB pick for now, but maybe shuffle candidates?
        # No, let's try to pick the candidate with the most "conflicts" with the current set.
        # If i conflicts with many j's in candidates (i.e., adding i would remove j), it's a good branch.
        
        # Simple version: just take the first available.
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1
        
        # Branch 1: Exclude i
        dfs(current_mask, candidates_mask & ~lsb, current_size)
        
        # Branch 2: Include i
        if (collinear_mask[i] if False else 0): # placeholder
            pass
            
        # Check if i can be added
        can_add = True
        # We need to check if i forms a triple with any pair in current_mask.
        # current_mask is a set of indices.
        # We can iterate through pairs in current_mask? That's slow.
        # Better: Precompute for each i, the set of pairs in current that conflict?
        # No, just check:
        temp = current_mask
        while temp:
            lsb_c = temp & -temp
            j = lsb_c.bit_length() - 1
            temp -= lsb_c
            if (collinear_mask[i][j] & current_mask) != 0:
                can_add = False
                break
        
        if can_add:
            new_current = current_mask | lsb
            # Which candidates are now invalid?
            # A candidate k is invalid if it forms a triple with i and some j in new_current?
            # Actually, if we add i, we must remove any k in candidates such that
            # (k, i, j) is collinear for some j in new_current?
            # Or (k, j1, j2) is collinear for some j1, j2 in new_current?
            # The condition for k to be addable later is:
            # For all j in new_current, (collinear_mask[k][j] & new_current) == 0.
            # So if we add i, we can remove k from candidates if:
            # (collinear_mask[k][i] & new_current) != 0  OR
            # There exists j in new_current such that (collinear_mask[k][j] & new_current) != 0?
            # Wait, the check is done when k is considered.
            # To prune candidates_mask now, we can remove k if it is "impossible" to add later.
            # k is impossible if it conflicts with the fixed set new_current.
            # Conflict: (collinear_mask[k][j] & new_current) != 0 for some j in new_current.
            
            new_candidates = candidates_mask & ~lsb
            # Filter new_candidates
            valid_cands = 0
            temp_c = new_candidates
            while temp_c:
                lsb_k = temp_c & -temp_c
                k = lsb_k.bit_length() - 1
                temp_c -= lsb_k
                
                # Check if k is compatible with new_current
                compatible = True
                temp_u = new_current
                while temp_u:
                    lsb_u = temp_u & -temp_u
                    u = lsb_u.bit_length() - 1
                    temp_u -= lsb_u
                    if (collinear_mask[k][u] & new_current) != 0:
                        compatible = False
                        break
                
                if compatible:
                    valid_cands |= lsb_k
            
            dfs(new_current, valid_cands, current_size + 1)

    dfs(0, (1 << n) - 1, 0)
    return best[0]

def _try_line_packing(deadline=None):
    """Systematically build points to create 3-point lines."""
    points = []
    seen = set()
    
    # Define a set of lines to pack
    # We want to maximize triples.
    # Let's try to place points on lines with slopes 0, 1, -1, 2, -2, inf.
    # We will pick a grid of candidate points and select 20.
    
    # Generate candidate points
    candidates = []
    for x in range(-20, 21):
        for y in range(-20, 21):
            candidates.append((x, y))
    
    # Shuffle candidates to break symmetry
    random.shuffle(candidates)
    
    # Greedily select points
    # Score of a point: how many triples it forms with existing points?
    # Or how many "potential" triples (pairs of existing points) it completes?
    
    for cand in candidates:
        if len(points) >= 20:
            break
        if deadline and time.time() > deadline:
            break
            
        x, y = cand
        if (x, y) in seen:
            continue
        if abs(x) > 100 or abs(y) > 100:
            continue
            
        test_points = points + [[x, y]]
        
        # Check no 4 collinear
        if _has_four_collinear(test_points):
            continue
            
        # Count new triples
        new_triples = 0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if _collinear3(points[i], points[j], [x, y]):
                    new_triples += 1
        
        # If it creates at least 1 new triple, or if we are early, add it?
        # If we only add if new_triples > 0, we might get stuck.
        # Strategy: Add if new_triples > 0. If stuck, add random valid point.
        if new_triples > 0:
            points.append([x, y])
            seen.add((x, y))
        else:
            # If we have < 5 points, add anyway to build base
            if len(points) < 5:
                points.append([x, y])
                seen.add((x, y))
            else:
                # Skip if it doesn't help, unless we need to fill up
                # If we run out of candidates, we need to add something
                pass

    # If we didn't get 20, fill with random valid points
    if len(points) < 20:
        # Try to find points that don't create 4-collinear
        attempts = 0
        while len(points) < 20 and attempts < 1000:
            attempts += 1
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) in seen:
                continue
            test_points = points + [[x, y]]
            if _has_four_collinear(test_points):
                continue
            points.append([x, y])
            seen.add((x, y))
            
    return points[:20]

def _local_search_line_aware(points, deadline):
    """Local search that specifically tries to create new collinear triples."""
    current = [p[:] for p in points]
    current_triples = _count_collinear_triples(current)
    best = [p[:] for p in current]
    best_triples = current_triples
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        orig = current[idx][:]
        
        # Try to move it to a position that creates new triples
        moved = False
        
        # 1. Try completing triples: Find pairs (i, j) in current (excluding idx)
        # and place idx on the line through i, j.
        candidates = []
        for i in range(20):
            for j in range(i+1, 20):
                if i == idx or j == idx:
                    continue
                x1, y1 = current[i]
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                # Find integer points on this line within bounds
                # Parametric: (x1 + t*dx, y1 + t*dy)
                # We need integer solutions.
                # Since x1, x2 are integers, dx, dy are integers.
                # Any t integer gives integer point.
                # We need |x|, |y| <= 100.
                # Also must be distinct from all other points.
                
                # Try t values
                # t=0 is x1, t=1 is x2 (occupied)
                # Try t in [-5, 5]
                for t in range(-5, 6):
                    if t == 0 or t == 1:
                        continue
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    cand = (nx, ny)
                    # Check if cand is in current
                    if cand in [tuple(p) for p in current]:
                        continue
                    # Check if cand creates 4-collinear
                    # Replace idx with cand
                    test = [p[:] for p in current]
                    test[idx] = [nx, ny]
                    if _has_four_collinear(test):
                        continue
                    # Check if this creates new triples
                    # It definitely creates triple (i, j, idx).
                    # Does it create others?
                    new_triples = _count_collinear_triples(test)
                    candidates.append((new_triples, [nx, ny]))
        
        # Sort by new_triples descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for _, cand in candidates:
            test = [p[:] for p in current]
            test[idx] = cand
            new_triples = _count_collinear_triples(test)
            if new_triples > current_triples:
                current[idx] = cand
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best = [p[:] for p in current]
                moved = True
                break
        
        if not moved:
            # Try random move
            for _ in range(5):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if any(tuple(cand) == tuple(p) for k, p in enumerate(current) if k != idx):
                    continue
                test = [p[:] for p in current]
                test[idx] = cand
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > current_triples:
                    current[idx] = cand
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best = [p[:] for p in current]
                    moved = True
                    break
        
        if not moved:
            current[idx] = orig
            
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_points = None
    best_score = 20
    
    # Try multiple line-packing constructions
    for attempt in range(10):
        if time.time() > deadline - 15:
            break
        pts = _try_line_packing(deadline - 10)
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        # Do local search to improve
        improved = _local_search_line_aware(pts, deadline - 10)
        
        # Evaluate
        score = _max_3free_subset(improved, deadline - 5)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
            
    if best_points is None:
        # Fallback: random construction
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if tuple(p) in seen:
                continue
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
                continue
            seen.add(tuple(p))
        best_points = points
        
    return best_points
