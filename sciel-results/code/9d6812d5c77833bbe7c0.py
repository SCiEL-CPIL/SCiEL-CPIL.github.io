import random
import time
import sys
import math
from itertools import combinations
from math import gcd

def cross(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def _get_line_key(p1, p2):
    """Get normalized line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None
        
    if dx == 0:
        a, b, c = 1, 0, -x1
    elif dy == 0:
        a, b, c = 0, 1, -y1
    else:
        g = gcd(abs(dx), abs(dy))
        dxn = dx // g
        dyn = dy // g
        if dxn < 0:
            dxn = -dxn
            dyn = -dyn
        a, b = dyn, -dxn
        c = -(a * x1 + b * y1)
        g2 = gcd(gcd(abs(a), abs(b)), abs(c))
        if g2 > 0:
            a //= g2
            b //= g2
            c //= g2
    return (a, b, c)

def count_collinear_triples(points):
    """Count the number of collinear triples in the set of points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if invalid."""
    n = len(points)
    # Use line keys to group points
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    for key, idxs in line_map.items():
        if len(idxs) >= 4:
            return True
    return False

def max_independent_set_3_uniform(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses Branch and Bound with bitmasks.
    """
    n = len(points)
    if n <= 2:
        return n

    # Precompute collinear triples as a list of (i, j, k)
    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    if not collinear_triples:
        return n

    # pair_to_third[i][j] = set of k such that {i, j, k} is collinear
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j: continue
            pair_to_third[i][j] = set()
            
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)

    best_size = 0
    full_mask = (1 << n) - 1

    def solve_recur(current_size, current_mask, remaining_mask):
        nonlocal best_size
        
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Pruning: if current_size + popcount(remaining) <= best_size, prune
        if current_size + bin(remaining_mask).count('1') <= best_size:
            return

        # Pick a point from remaining
        # Heuristic: Pick the point with the most conflicts in remaining?
        # For speed, just pick the lowest index.
        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
                
        if p == -1:
            return

        # Option 1: Exclude p
        solve_recur(current_size, current_mask, remaining_mask & ~(1 << p))

        # Option 2: Include p
        # Check if p is compatible with current_mask
        compatible = True
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = (lsb.bit_length() - 1)
            # If r and p are in a triple with some s in current_mask, it's invalid.
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
            
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            
            # Remove forbidden points from new_remaining
            # A point q in new_remaining is forbidden if it forms a triple with p and any r in new_current_mask.
            forbidden_mask = 0
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = (lsb.bit_length() - 1)
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
                
            new_remaining &= ~forbidden_mask
            solve_recur(current_size + 1, new_current_mask, new_remaining)

    solve_recur(0, 0, full_mask)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 20
    best_points = None
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    def greedy_init():
        """
        Greedy construction: Add points one by one to maximize collinear triples.
        """
        points = []
        seen = set()
        
        # Start with a few random points to seed lines
        for _ in range(3):
            p = random_point()
            if p not in seen:
                points.append(p)
                seen.add(p)
        
        while len(points) < 20:
            best_candidate = None
            best_gain = -1
            best_triples = 0
            
            # Try a batch of random candidates
            candidates = [random_point() for _ in range(50)]
            
            for p in candidates:
                if p in seen:
                    continue
                
                # Check no 4 collinear with existing points
                # We check if adding p creates a line with 4 points.
                # This happens if p lies on a line with 3 existing points, 
                # or if p forms a line with 2 existing points that already has 2 other points? 
                # Actually, simpler: check all lines through p and existing points.
                valid = True
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        if are_collinear(p, points[i], points[j]):
                            # Count how many points are on this line
                            line_pts = [p, points[i], points[j]]
                            for k in range(len(points)):
                                if k != i and k != j and are_collinear(p, points[i], points[k]):
                                    line_pts.append(points[k])
                            # If there are already 3 points on the line (excluding p), 
                            # adding p makes 4. 
                            # Wait, if i,j are on line, and k is on line, then i,j,k are collinear.
                            # If we add p, we have p,i,j,k. That's 4.
                            # So if there are 3 existing points collinear with p and any pair, it's invalid?
                            # No, if i,j,k are collinear, and p is on that line, then p,i,j,k are collinear.
                            # So we must check if p lies on any line defined by 3 existing points.
                            pass # Logic below is more direct
                
                # Direct check for 4-collinear violation
                # A violation occurs if there exist 3 points a,b,c in points such that p, a, b, c are collinear.
                # This is equivalent to: for any pair (a,b) in points, if p is collinear with a,b,
                # we must ensure there isn't a third point c in points (c!=a,b) also collinear with a,b.
                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        if are_collinear(p, points[i], points[j]):
                            # Check if there is a third point k on this line
                            for k in range(len(points)):
                                if k != i and k != j and are_collinear(p, points[i], points[k]):
                                    valid = False
                                    break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Calculate gain in triples
                gain = 0
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        if are_collinear(p, points[i], points[j]):
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = p
                    best_triples = gain
            
            if best_candidate is None:
                # If no candidate works, pick a random valid one
                while True:
                    p = random_point()
                    if p in seen:
                        continue
                    # Check validity
                    valid = True
                    for i in range(len(points)):
                        for j in range(i + 1, len(points)):
                            if are_collinear(p, points[i], points[j]):
                                for k in range(len(points)):
                                    if k != i and k != j and are_collinear(p, points[i], points[k]):
                                        valid = False
                                        break
                        if not valid:
                            break
                    if valid:
                        best_candidate = p
                        break
            
            points.append(best_candidate)
            seen.add(best_candidate)
            
        return points

    def swap_sa(points, max_steps=2000):
        """
        Simulated Annealing using SWAPS.
        """
        points = list(points)
        current_tc = count_collinear_triples(points)
        T = 1.0
        cooling_rate = 0.995
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 19)
            j = random.randint(0, 19)
            if i == j:
                continue
            
            # Swap points[i] and points[j]
            points[i], points[j] = points[j], points[i]
            
            if has_four_collinear(points):
                # Revert
                points[i], points[j] = points[j], points[i]
                T *= cooling_rate
                continue
            
            new_tc = count_collinear_triples(points)
            delta = new_tc - current_tc
            
            if delta > 0:
                current_tc = new_tc
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_tc = new_tc
                else:
                    # Revert
                    points[i], points[j] = points[j], points[i]
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                
        return points, current_tc

    # Phase 1: Multiple restarts with Greedy Init + Swap SA
    for restart in range(50):
        if time.time() > deadline:
            break
        
        points = greedy_init()
        
        # Ensure validity (greedy should be valid, but double check)
        if has_four_collinear(points):
            # Repair if needed
            for _ in range(100):
                idx = random.randint(0, 19)
                new_p = random_point()
                if new_p in points:
                    continue
                points[idx] = new_p
                if not has_four_collinear(points):
                    break
        
        if has_four_collinear(points):
            continue
            
        # Run Swap SA
        points, tc = swap_sa(points, max_steps=1500)
        
        if has_four_collinear(points):
            continue
            
        # Compute exact score
        score = max_independent_set_3_uniform(points)
        
        if score < best_score:
            best_score = score
            best_points = list(points)
            
        # If score is very good, do a second pass of Swap SA with lower T
        if best_score <= 15:
            points, tc = swap_sa(points, max_steps=500)
            if not has_four_collinear(points):
                score = max_independent_set_3_uniform(points)
                if score < best_score:
                    best_score = score
                    best_points = list(points)

    if best_points is None:
        # Fallback: Random valid set
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        for _ in range(200):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        best_points = points
    
    return [list(p) for p in best_points]
