import random
import time
from math import gcd

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normal vector (perpendicular)
    nx, ny = dy, -dx
    c = -(nx * x1 + ny * y1)
    g2 = gcd(abs(nx), abs(ny))
    if g2 != 0:
        nx //= g2
        ny //= g2
        c //= g2
    return (nx, ny, c)

def _get_lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _is_valid(points):
    lines = _get_lines(points)
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _get_triple_masks(points):
    lines = _get_lines(points)
    n = len(points)
    triple_masks = [0] * n
    for pts in lines.values():
        if len(pts) >= 3:
            # For each point in the line, it forms triples with any other pair
            # But for B&B, we need to know which pairs conflict with a third point.
            # Actually, the standard way is: a set is independent if no triple from the hypergraph is fully contained.
            # The hypergraph edges are the collinear triples.
            # We can precompute for each vertex v, the set of pairs (u, w) such that {u, v, w} is a triple.
            # Or simply, for B&B, we check if adding v creates a triple with any two already selected.
            
            # Let's precompute a bitmask of "conflict pairs" for each vertex?
            # No, let's use the standard approach:
            # For each vertex v, store a bitmask of vertices u such that there exists w with {u, v, w} collinear? No, that's not enough.
            # We need to know if {u, w, v} is a triple.
            
            # Efficient check in B&B:
            # When adding v to S, check if there are u, w in S such that {u, v, w} is a triple.
            # We can precompute for each v, a list of (u, w) pairs? Or a dictionary pair -> mask of thirds?
            
            pass
    
    # Better: Precompute pair_to_thirds
    pair_to_thirds = {}
    for pts in lines.values():
        if len(pts) >= 3:
            pts_list = list(pts)
            for a in range(len(pts_list)):
                for b in range(a + 1, len(pts_list)):
                    u = pts_list[a]
                    w = pts_list[b]
                    pair = (min(u, w), max(u, w))
                    if pair not in pair_to_thirds:
                        pair_to_thirds[pair] = set()
                    # Any other point on this line is a third
                    for c in range(len(pts_list)):
                        if c != a and c != b:
                            pair_to_thirds[pair].add(pts_list[c])
    
    return pair_to_thirds

def _calculate_score(points):
    n = len(points)
    pair_to_thirds = _get_triple_masks(points)
    
    # Degrees for heuristic
    degrees = [0] * n
    for pair, thirds in pair_to_thirds.items():
        u, w = pair
        for v in thirds:
            # This counts how many triples each vertex is part of
            # Actually, just counting pairs is enough for a heuristic
            pass
    
    # Better degree: number of triples containing the vertex
    # Let's recompute degrees properly
    degrees = [0] * n
    lines = _get_lines(points)
    for pts in lines.values():
        if len(pts) >= 3:
            for v in pts:
                degrees[v] += 1

    best_score = 0

    def can_add(v, selected_mask):
        # Check if v can be added to selected_mask
        # We need to ensure no pair (u, w) in selected_mask has v as a third
        # Iterate through set bits in selected_mask
        # Optimization: iterate through pairs in selected_mask?
        # Since n=20, selected_mask has at most 20 bits.
        # Number of pairs is at most 190.
        # For each pair, check if v is in pair_to_thirds[pair].
        
        # Faster: Precompute for each v, the set of pairs that conflict with v?
        # Let's stick to the simple check but optimize the inner loop.
        
        # Alternative: For each u in selected_mask, check if there is w in selected_mask such that (u, w, v) is a triple.
        # This is O(k^2) where k is size of selected_mask.
        
        # Let's try iterating u, w
        bits = []
        m = selected_mask
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
            
        for i in range(len(bits)):
            u = bits[i]
            for j in range(i + 1, len(bits)):
                w = bits[j]
                pair = (min(u, w), max(u, w))
                if pair in pair_to_thirds and v in pair_to_thirds[pair]:
                    return False
        return True

    def search(selected_mask, available_mask, current_size):
        nonlocal best_score
        if available_mask == 0:
            if current_size > best_score:
                best_score = current_size
            return
        
        # Upper bound
        ub = current_size + bin(available_mask).count('1')
        if ub <= best_score:
            return
            
        # Heuristic: Pick vertex with highest degree in available_mask?
        # Or lowest? Usually highest degree is more constraining.
        best_v = -1
        best_deg = -1
        m = available_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if degrees[v] > best_deg:
                best_deg = degrees[v]
                best_v = v
            m ^= lsb
            
        if best_v == -1:
            return
            
        new_available = available_mask & ~(1 << best_v)
        
        # Option 1: Exclude
        search(selected_mask, new_available, current_size)
        
        # Option 2: Include
        if can_add(best_v, selected_mask):
            search(selected_mask | (1 << best_v), new_available, current_size + 1)

    search(0, (1 << n) - 1, 0)
    return best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial construction: Random points, repair to valid
    # A valid point set is one with no 4 collinear.
    # Random points in [-100, 100] are very likely valid.
    
    def random_valid_points():
        while True:
            pts = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(20)]
            # Check distinct
            if len(set(pts)) == 20 and _is_valid(pts):
                return pts

    # Start with a few diverse initial points
    best_score = 20
    best_points = None
    
    # Try to find a good starting point
    # Score 1.4286 implies the current best is 14 (20/14 = 1.428...).
    # Wait, score = seed / value. Seed is likely 20? Or is it 1.0?
    # "1.0 equals the seed program".
    # If seed program score is 1.4286, and seed is 1.0, then value = 1/1.4286 = 0.7? No.
    # "search score is seed/value". If seed=1, score=1/value.
    # If score=1.4286, value = 1/1.4286 = 0.7. This doesn't make sense for an integer score.
    # Maybe the "value" is the score of the configuration (MIS size)?
    # If MIS size is 14, and seed is 20, then 20/14 = 1.428.
    # So we want to minimize the MIS size.
    
    # Let's initialize with a random valid set and calculate its score.
    current_points = random_valid_points()
    current_score = _calculate_score(current_points)
    best_score = current_score
    best_points = current_points[:]
    
    # Local Search
    # Move one point at a time.
    # To be efficient, we should only accept moves that keep the set valid.
    # And we want to minimize the score.
    
    # Optimization: Precompute lines for the current set.
    # When moving a point, only lines involving that point change.
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # Try to find a new position for this point
        # Random search in the box
        found = False
        for _ in range(20): # Try a few random spots
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if (new_x, new_y) == (old_x, old_y):
                continue
            if (new_x, new_y) in current_points:
                continue
            
            # Check if new point creates 4 collinear
            # We can check this quickly by looking at lines through the new point
            # and existing points.
            # If any line through new point and 2 existing points has a 3rd existing point, it's invalid.
            # Actually, we just need to check if the new set is valid.
            # Since only point idx changes, we only need to check lines involving idx.
            
            valid = True
            # Check lines through new point and any other pair
            # This is O(N^2) per check.
            # For each pair (j, k) where j, k != idx:
            # If new_point, j, k are collinear, check if there is a 4th point.
            # This is slow.
            
            # Simpler: Just replace and run _is_valid.
            # _is_valid is O(N^2) for line construction + O(N) for check.
            # With N=20, this is very fast.
            
            current_points[idx] = (new_x, new_y)
            if _is_valid(current_points):
                # Check score
                # We can estimate if it's likely to be better?
                # No, just calculate.
                new_score = _calculate_score(current_points)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_points = current_points[:]
                    found = True
                    break
                else:
                    # Revert
                    current_points[idx] = (old_x, old_y)
            else:
                # Revert
                current_points[idx] = (old_x, old_y)
                
        # If we found a better score, we continue.
        # If not, we might restart or try a different strategy.
        
        # Occasionally, try a "restart" from a different random valid set
        # or a perturbation of the best found so far.
        if random.random() < 0.05:
            # Perturb best_points
            perturbed = best_points[:]
            for _ in range(5):
                i = random.randint(0, 19)
                perturbed[i] = (perturbed[i][0] + random.randint(-10, 10), perturbed[i][1] + random.randint(-10, 10))
                # Clamp
                perturbed[i] = (max(-100, min(100, perturbed[i][0])), max(-100, min(100, perturbed[i][1])))
            
            # Repair perturbed
            # Just try to move points until valid
            for _ in range(50):
                if len(set(perturbed)) == 20 and _is_valid(perturbed):
                    break
                i = random.randint(0, 19)
                perturbed[i] = (random.randint(-100, 100), random.randint(-100, 100))
            
            if len(set(perturbed)) == 20 and _is_valid(perturbed):
                s = _calculate_score(perturbed)
                if s < current_score:
                    current_points = perturbed
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_points = perturbed[:]
                # Even if not better, we can use it as the new current state to explore
                # But greedy says only move if better.
                # Let's accept it if it's not too bad?
                # No, stick to greedy for now to ensure progress.
                # Actually, simulated annealing might be better.
                # Let's try accepting worse moves with a small probability.
                # But we need to track temperature.
                # For simplicity, let's just do greedy + restarts.

    return [list(p) for p in best_points]
