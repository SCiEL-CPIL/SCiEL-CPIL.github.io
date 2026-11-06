import time
import math
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def build_line_map(pts):
        """Build a map from line key -> set of point indices on that line."""
        n = len(pts)
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is not None:
                    line_map[key].add(i)
                    line_map[key].add(j)
        return line_map

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        line_map = build_line_map(pts)
        for key, indices in line_map.items():
            if len(indices) >= 4:
                return True
        return False

    def count_triples(pts):
        """Count collinear triples."""
        line_map = build_line_map(pts)
        triples = 0
        for key, indices in line_map.items():
            k = len(indices)
            if k >= 3:
                triples += k * (k - 1) * (k - 2) // 6
        return triples

    def calculate_score(pts):
        """
        Exact branch and bound for largest subset with no 3 collinear.
        Uses bitmask representation and conflict detection.
        """
        n = len(pts)
        if n < 3:
            return n

        # Build triple conflict structure: for each point i, store list of (j,k) 
        # such that i,j,k are collinear. If both j and k are in the subset, 
        # we cannot also include i.
        # More efficiently: for each pair (i,j), find all k such that i,j,k collinear.
        # A subset is valid iff no three are collinear.
        
        # Build: for each point i, a bitmask of "forbidden pairs" - 
        # Actually, let's use: conflicts[i] = set of (j,k) pairs where i,j,k collinear
        # When we add point i, it's invalid if there exist j,k already in subset with (j,k) in conflicts[i]
        
        # Alternative: precompute for each triple (i,j,k) that is collinear.
        # Use a different approach: for each point i, compute the set of points that 
        # form a line with i and some other point. 
        # 
        # Better: For each point i, build a "conflict mask" - but that's complex.
        # Let's use: for each i, store list of (j, k) where j < k and i,j,k collinear.
        
        triples_by_point = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples_by_point[i].append((j, k))
                        triples_by_point[j].append((i, k))
                        triples_by_point[k].append((i, j))
        
        best_score = 0
        
        # Order points by number of conflicts (most constrained first)
        order = sorted(range(n), key=lambda i: len(triples_by_point[i]), reverse=True)
        
        # Precompute: for each position in order, what index it corresponds to
        order_idx = [0] * n
        for pos, idx in enumerate(order):
            order_idx[idx] = pos
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 3:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            # Upper bound
            if current_size + (n - step) <= best_score:
                return
            idx = order[step]
            
            # Try excluding first (usually more productive for pruning)
            bbb(step + 1, current_mask, current_size)
            
            # Try including if valid
            valid = True
            for a, b in triples_by_point[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def greedy_construct_triples():
        """
        Construct 20 points with many collinear triples (exactly 3 on a line).
        Strategy: Start with a set of lines, each with 3 points, sharing points 
        to keep total at 20. Use a structured approach.
        """
        best_pts = None
        best_triples = 0
        
        for attempt in range(20):
            if elapsed() > deadline - 50:
                break
            
            # Create a configuration based on a small grid with controlled structure
            # Use a 5x5 grid region and select 20 points from a larger candidate pool
            
            # Generate candidate points from a structured pattern
            candidates = []
            
            # Pattern 1: Perturbed grid
            base = random.choice([4, 5, 6, 7])
            step = random.choice([10, 12, 15, 18, 20])
            ox = random.randint(-60, 60)
            oy = random.randint(-60, 60)
            
            for i in range(base):
                for j in range(base):
                    x = ox + i * step + random.randint(-2, 2)
                    y = oy + j * step + random.randint(-2, 2)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.append((x, y))
            
            # Remove duplicates
            seen = set()
            unique_candidates = []
            for p in candidates:
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
            
            if len(unique_candidates) < 20:
                continue
            
            # Greedily select 20 points maximizing triple count
            selected = []
            for _ in range(20):
                if elapsed() > deadline - 50:
                    break
                best_cand = None
                best_gain = -1
                
                for cand in unique_candidates:
                    if cand in selected:
                        continue
                    
                    # Check 4-collinear constraint
                    valid = True
                    triple_gain = 0
                    xp, yp = cand
                    
                    # Check against all pairs in selected
                    for i in range(len(selected)):
                        x1, y1 = selected[i]
                        for j in range(i + 1, len(selected)):
                            x2, y2 = selected[j]
                            # Check if cand is collinear with i,j
                            if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                                # Check for 4th point on this line
                                for k in range(len(selected)):
                                    if k != i and k != j:
                                        x3, y3 = selected[k]
                                        if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                                            valid = False
                                            break
                                if not valid:
                                    break
                                else:
                                    triple_gain += 1
                        if not valid:
                            break
                    
                    if valid and triple_gain > best_gain:
                        best_gain = triple_gain
                        best_cand = cand
                
                if best_cand is None:
                    break
                selected.append(best_cand)
            
            if len(selected) == 20 and not has_4_collinear(selected):
                t = count_triples(selected)
                if t > best_triples:
                    best_triples = t
                    best_pts = selected
        
        return best_pts

    def local_search(pts, max_iters=200):
        """Local search to maximize triple count while maintaining validity."""
        if pts is None:
            return None
        
        current = list(pts)
        current_triples = count_triples(current)
        best = list(current)
        best_triples = current_triples
        
        for it in range(max_iters):
            if elapsed() > deadline - 25:
                break
            
            idx = random.randint(0, 19)
            old_p = current[idx]
            
            # Generate neighbor
            r = random.random()
            if r < 0.3:
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            elif r < 0.6:
                new_p = (old_p[0] + random.randint(-15, 15), old_p[1] + random.randint(-15, 15))
            elif r < 0.8:
                new_p = (random.randint(-100, 100), old_p[1] + random.randint(-10, 10))
            else:
                new_p = (old_p[0] + random.randint(-10, 10), random.randint(-100, 100))
            
            if new_p in current:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
            
            # Check validity (no 4 collinear)
            valid = True
            xp, yp = new_p
            for i in range(20):
                if i == idx:
                    continue
                x1, y1 = current[i]
                for j in range(i + 1, 20):
                    if j == idx:
                        continue
                    x2, y2 = current[j]
                    if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                        for k in range(20):
                            if k == idx or k == i or k == j:
                                continue
                            x3, y3 = current[k]
                            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_pts = current[:]
            new_pts[idx] = new_p
            new_triples = count_triples(new_pts)
            
            if new_triples > current_triples:
                current = new_pts
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best = list(current)
            elif new_triples == current_triples and random.random() < 0.15:
                current = new_pts
        
        return best

    def simulated_annealing(pts, max_iters=300):
        """Simulated annealing to maximize triple count."""
        if pts is None:
            return None
        
        current = list(pts)
        current_triples = count_triples(current)
        best = list(current)
        best_triples = current_triples
        
        temp = 10.0
        cooling = 0.99
        
        for it in range(max_iters):
            if elapsed() > deadline - 25:
                break
            
            idx = random.randint(0, 19)
            old_p = current[idx]
            
            r = random.random()
            if r < 0.3:
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            elif r < 0.6:
                new_p = (old_p[0] + random.randint(-15, 15), old_p[1] + random.randint(-15, 15))
            else:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
            
            if new_p in current:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
            
            valid = True
            xp, yp = new_p
            for i in range(20):
                if i == idx:
                    continue
                x1, y1 = current[i]
                for j in range(i + 1, 20):
                    if j == idx:
                        continue
                    x2, y2 = current[j]
                    if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                        for k in range(20):
                            if k == idx or k == i or k == j:
                                continue
                            x3, y3 = current[k]
                            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_pts = current[:]
            new_pts[idx] = new_p
            new_triples = count_triples(new_pts)
            
            delta = new_triples - current_triples
            if delta >= 0 or random.random() < math.exp(delta / temp):
                current = new_pts
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best = list(current)
            
            temp *= cooling
        
        return best

    # Main search
    best_points = None
    best_score = float('inf')
    
    # Phase 1: Multiple construction attempts
    for attempt in range(8):
        if elapsed() > deadline - 70:
            break
        
        pts = greedy_construct_triples()
        if pts is not None:
            # Local search
            improved = local_search(pts, max_iters=80)
            if improved is not None:
                s = calculate_score(improved)
                if s < best_score:
                    best_score = s
                    best_points = improved
    
    # Phase 2: Simulated annealing on best found
    if best_points is not None and elapsed() < deadline - 40:
        improved = simulated_annealing(best_points, max_iters=200)
        if improved is not None:
            s = calculate_score(improved)
            if s < best_score:
                best_score = s
                best_points = improved
    
    # Phase 3: More diverse constructions
    if elapsed() < deadline - 35:
        for attempt in range(6):
            if elapsed() > deadline - 30:
                break
            
            # Different structured patterns
            candidates = []
            
            # Diagonal-based pattern
            pattern_type = random.randint(0, 2)
            
            if pattern_type == 0:
                # Grid with larger spacing
                base = random.randint(5, 8)
                step = random.randint(15, 30)
                ox = random.randint(-70, 70)
                oy = random.randint(-70, 70)
                for i in range(base):
                    for j in range(base):
                        x = ox + i * step + random.randint(-3, 3)
                        y = oy + j * step + random.randint(-3, 3)
                        if -100 <= x <= 100 and -100 <= y <= 100:
                            candidates.append((x, y))
            elif pattern_type == 1:
                # Two interleaved grids
                for offset in [0, random.randint(5, 15)]:
                    base = random.randint(4, 6)
                    step = random.randint(12, 25)
                    ox = random.randint(-60, 60) + offset
                    oy = random.randint(-60, 60)
                    for i in range(base):
                        for j in range(base):
                            x = ox + i * step + random.randint(-2, 2)
                            y = oy + j * step + random.randint(-2, 2)
                            if -100 <= x <= 100 and -100 <= y <= 100:
                                candidates.append((x, y))
            else:
                # Random but clustered
                cx = random.randint(-50, 50)
                cy = random.randint(-50, 50)
                spread = random.randint(20, 60)
                for _ in range(40):
                    x = cx + random.randint(-spread, spread)
                    y = cy + random.randint(-spread, spread)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.append((x, y))
            
            seen = set()
            unique_candidates = []
            for p in candidates:
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
            
            if len(unique_candidates) < 20:
                continue
            
            # Greedy selection
            selected = []
            for _ in range(20):
                if elapsed() > deadline - 30:
                    break
                best_cand = None
                best_gain = -1
                for cand in unique_candidates:
                    if cand in selected:
                        continue
                    valid = True
                    gain = 0
                    xp, yp = cand
                    for i in range(len(selected)):
                        x1, y1 = selected[i]
                        for j in range(i + 1, len(selected)):
                            x2, y2 = selected[j]
                            if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                                for k in range(len(selected)):
                                    if k != i and k != j:
                                        x3, y3 = selected[k]
                                        if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                                            valid = False
                                            break
                                if not valid:
                                    break
                                gain += 1
                        if not valid:
                            break
                    if valid and gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                if best_cand is None:
                    break
                selected.append(best_cand)
            
            if len(selected) == 20 and not has_4_collinear(selected):
                improved = local_search(selected, max_iters=60)
                if improved is not None:
                    s = calculate_score(improved)
                    if s < best_score:
                        best_score = s
                        best_points = improved
    
    # Phase 4: Final intensive search on best
    if best_points is not None and elapsed() < deadline - 15:
        improved = local_search(best_points, max_iters=150)
        if improved is not None:
            s = calculate_score(improved)
            if s < best_score:
                best_score = s
                best_points = improved
        
        if elapsed() < deadline - 10:
            improved = simulated_annealing(best_points, max_iters=100)
            if improved is not None:
                s = calculate_score(improved)
                if s < best_score:
                    best_score = s
                    best_points = improved
    
    # Fallback: random valid configuration
    if best_points is None:
        while True:
            if elapsed() > deadline - 5:
                # Emergency: just return any 20 distinct points
                pts = []
                while len(pts) < 20:
                    p = (random.randint(-100, 100), random.randint(-100, 100))
                    if p not in pts:
                        pts.append(p)
                return [[x, y] for x, y in pts]
            
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break
    
    return [[x, y] for x, y in best_points]
