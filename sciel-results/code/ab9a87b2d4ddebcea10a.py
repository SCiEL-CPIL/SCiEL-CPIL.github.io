import time
import random
import sys

def _elapsed(t0):
    return time.time() - t0

def _is_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return True
    return False

def _build_triples(points):
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            line_points = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_points.append(k)
            m = len(line_points)
            if m >= 3:
                for a in range(m):
                    for b in range(a + 1, m):
                        for c in range(b + 1, m):
                            triple = frozenset([line_points[a], line_points[b], line_points[c]])
                            if triple not in seen:
                                seen.add(triple)
                                triples.append(triple)
    return triples

def _score(points):
    triples = _build_triples(points)
    n = len(points)
    
    # Precompute for each index, the set of other indices it forms a triple with
    # This is for the "remove all conflicts" step in the bitmask backtracker
    # Actually, for the bitmask backtracker, we need to know for each point i,
    # which other points j, k form a triple with i.
    # Let's store for each point i, a bitmask of points that form a triple with i.
    # If we include i, we must remove all j, k such that (i, j, k) is a triple.
    
    # Efficient representation:
    # conflict_mask[i] is a bitmask where bit j is set if i and j are in some triple.
    # Wait, that's not enough. If we include i, we remove j and k.
    # So if we include i, the new remaining mask is remaining & ~conflict_mask[i].
    
    conflict_masks = [0] * n
    for t in triples:
        i, j, k = tuple(t)
        conflict_masks[i] |= (1 << j) | (1 << k)
        conflict_masks[j] |= (1 << i) | (1 << k)
        conflict_masks[k] |= (1 << i) | (1 << j)
    
    best = [0]
    
    # Heuristic ordering: points with more conflicts should be decided first?
    # Or maybe points with fewer conflicts?
    # In maximum independent set, usually high degree first helps pruning.
    # Here, "degree" is the number of points it conflicts with (i.e., popcount(conflict_mask[i])).
    # Let's sort indices by conflict count descending.
    # But the bitmask backtracker works on bit positions 0..n-1.
    # We can permute the points or just use the natural order.
    # Given n=20, natural order is likely fine, but let's try to optimize the selection.
    # The standard bitmask B&B picks the lowest set bit.
    # We can't easily change the bit positions without remapping.
    # Let's stick to natural order but use the conflict masks for pruning.
    
    def backtrack(remaining_mask, current_count):
        if current_count + bin(remaining_mask).count('1') <= best[0]:
            return
        
        if remaining_mask == 0:
            if current_count > best[0]:
                best[0] = current_count
            return
        
        # Pick the lowest set bit
        lsb = remaining_mask & -remaining_mask
        idx = (lsb.bit_length() - 1)
        
        # Option 1: Include idx
        # Remove idx and all its conflicts
        remove_mask = lsb | conflict_masks[idx]
        new_remaining = remaining_mask & ~remove_mask
        backtrack(new_remaining, current_count + 1)
        
        # Option 2: Exclude idx
        new_remaining2 = remaining_mask & ~lsb
        backtrack(new_remaining2, current_count)
        
    backtrack((1 << n) - 1, 0)
    return best[0]

def _two_parabola_config(shift=1):
    """Generate 20 points from two shifted parabolas."""
    points = []
    x_range = list(range(-5, 5))
    
    for x in x_range:
        points.append([x, x * x])
    for x in x_range:
        points.append([x, x * x + shift])
    
    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    if len(unique) != 20:
        return None
    
    for x, y in unique:
        if abs(x) > 100 or abs(y) > 100:
            return None
            
    return unique

def _perturb(points, t0, deadline):
    """Perturb one or two points slightly."""
    new_points = [list(p) for p in points]
    num_perturb = random.choice([1, 2])
    for _ in range(num_perturb):
        idx = random.randint(0, 19)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            new_points[idx] = [nx, ny]
    
    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def construct():
    t0 = time.time()
    deadline = t0 + 115.0 # 35s margin for safety
    
    best_points = None
    best_score = 21
    
    # 1. Try structured initial configurations
    # Two parabolas is a known good starting point
    for shift in [1, 2, 3, -1, -2, 5, 7, 10]:
        if _elapsed(t0) > deadline - 10:
            break
        config = _two_parabola_config(shift)
        if config is None:
            continue
        if _has_four_collinear(config):
            continue
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in config]
            
    # If no structured config worked, use random
    if best_points is None:
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        # Ensure no 4 collinear (unlikely for random, but check)
        while _has_four_collinear(points):
            idx = random.randint(0, 19)
            points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            # Re-check distinctness
            seen = set(tuple(p) for p in points)
            if len(seen) < 20:
                # Re-roll the whole thing if we get duplicates
                points = []
                seen = set()
                while len(points) < 20:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in seen:
                        seen.add((x, y))
                        points.append([x, y])
                continue
        s = _score(points)
        best_score = s
        best_points = [list(p) for p in points]

    # 2. Local Search / Simulated Annealing
    # Objective: Minimize the score (max 3-free subset)
    # Since calculating the exact score is expensive (branch and bound),
    # we can use a proxy metric or just calculate it directly if fast enough.
    # For n=20, the B&B is usually fast.
    
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    temp = 50.0
    cool = 0.999
    
    while _elapsed(t0) < deadline:
        perturbed = _perturb(current_points, t0, deadline)
        if perturbed is None:
            continue
            
        if _has_four_collinear(perturbed):
            continue
            
        # Calculate score
        # To save time, we can first check if the number of triples increased.
        # If triples increase, the score likely decreases or stays same.
        # If triples decrease, score might increase.
        # But calculating triples is O(n^3) ~ 8000 ops, which is fast.
        # Calculating score is B&B, which is exponential but pruned.
        
        # Let's just calculate the score.
        s = _score(perturbed)
        
        delta = s - current_score
        
        # Accept if better (lower score), or with probability exp(-delta/temp)
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_points = perturbed
            current_score = s
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in current_points]
                
        temp *= cool
        if temp < 0.1:
            temp = 0.1

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Found config with score: {_score(pts)}")
    print(pts)
