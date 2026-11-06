import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
    
    def count_4_collinear(points):
        """Return True if any line has >= 4 points"""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                cnt = 2
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        cnt += 1
                        if cnt >= 4:
                            return True
        return False
    
    def count_collinear_triples(points):
        """Count all collinear triples (higher = better for minimizing no-3 subset)"""
        n = len(points)
        triples = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples += 1
        return triples
    
    def max_no_3_collinear(points):
        """Exact branch and bound for largest subset with no 3 collinear"""
        n = len(points)
        
        # Precompute: for each pair (i,j), which other points are collinear
        pair_coll = {}
        for i in range(n):
            for j in range(i + 1, n):
                ks = set()
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        ks.add(k)
                pair_coll[(i,j)] = ks
                pair_coll[(j,i)] = ks
        
        best = [0]
        
        def backtrack(selected, candidates, depth):
            if depth + len(candidates) <= best[0]:
                return
            if not candidates:
                if depth > best[0]:
                    best[0] = depth
                return
            
            p = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Skip p
            backtrack(selected, rest, depth)
            
            # Option 2: Take p if no triple formed
            ok = True
            for s in selected:
                ks = pair_coll.get((p, s), set())
                for s2 in selected:
                    if s2 != s and s2 in ks:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                new_selected = selected | {p}
                # Filter candidates: remove any q that would form a triple with two in new_selected
                new_cands = []
                for q in rest:
                    safe = True
                    for s1 in new_selected:
                        for s2 in new_selected:
                            if s1 != s2:
                                if q in pair_coll.get((s1, s2), set()):
                                    safe = False
                                    break
                        if not safe:
                            break
                    if safe:
                        new_cands.append(q)
                backtrack(new_selected, new_cands, depth + 1)
        
        backtrack(set(), list(range(n)), 0)
        return best[0]
    
    def generate_line_packed_config(seed):
        """Generate a config with points deliberately placed on multiple lines (3 per line)"""
        random.seed(seed)
        pts = []
        
        # Define several lines with different slopes/intercepts
        # We want ~7 lines with 3 points each (21 points, we'll use 20)
        # Lines: y = mx + b with various m, b
        line_defs = [
            (0, 0),    # y = 0
            (0, 1),    # y = 1  
            (0, -1),   # y = -1
            (1, 0),    # y = x
            (-1, 0),   # y = -x
            (1, 1),    # y = x + 1
            (-1, 1),   # y = -x + 1
            (2, 0),    # y = 2x
            (0, 2),    # y = 2
            (0, -2),   # y = -2
        ]
        
        # Shuffle lines and pick 7
        random.shuffle(line_defs)
        chosen_lines = line_defs[:7]
        
        # Assign 3 points to first 6 lines, 2 to last
        counts = [3, 3, 3, 3, 3, 3, 2]
        
        used_points = set()
        
        for (m, b), count in zip(chosen_lines, counts):
            # Pick 'count' distinct x values that give integer y in range
            # and don't create 4-collinear with existing points
            candidates = []
            for x in range(-10, 11):
                y = m * x + b
                if abs(x) > 100 or abs(y) > 100:
                    continue
                if (x, y) in used_points:
                    continue
                candidates.append((x, y))
            
            if len(candidates) < count:
                # Try wider range
                for x in range(-20, 21):
                    y = m * x + b
                    if abs(x) > 100 or abs(y) > 100:
                        continue
                    if (x, y) in used_points:
                        continue
                    if (x, y) not in candidates:
                        candidates.append((x, y))
            
            if len(candidates) < count:
                continue
                
            random.shuffle(candidates)
            for pt in candidates[:count]:
                pts.append(list(pt))
                used_points.add(pt)
        
        # If we don't have 20 points, add random ones
        while len(pts) < 20:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in used_points:
                pts.append([x, y])
                used_points.add((x, y))
        
        if len(pts) > 20:
            pts = pts[:20]
        
        return pts
    
    def generate_perturbed_grid(seed):
        """Generate a perturbed grid config"""
        random.seed(seed)
        pts = []
        for i in range(5):
            for j in range(4):
                x = i * 4 + random.randint(-1, 1)
                y = j * 4 + random.randint(-1, 1)
                pts.append([x, y])
        return pts
    
    def simulated_annealing(pts, max_iters=5000):
        """Optimize using simulated annealing to maximize collinear triples"""
        pts = [list(p) for p in pts]
        current_triples = count_collinear_triples(pts)
        best_triples = current_triples
        best_pts = [list(p) for p in pts]
        
        T = 10.0
        cooling = 0.999
        
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Pick a random point to move
            i = random.randint(0, 19)
            orig = pts[i][:]
            
            # Try a random new position nearby
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            
            nx, ny = orig[0] + dx, orig[1] + dy
            if abs(nx) > 100 or abs(ny) > 100:
                continue
            
            # Check if this creates a duplicate
            new_pts = [list(p) for p in pts]
            new_pts[i] = [nx, ny]
            
            if len(set(map(tuple, new_pts))) < 20:
                continue
            
            # Check no 4 collinear
            if count_4_collinear(new_pts):
                continue
            
            new_triples = count_collinear_triples(new_pts)
            delta = new_triples - current_triples
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                pts = new_pts
                current_triples = new_triples
                
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_pts = [list(p) for p in pts]
            
            T *= cooling
        
        return best_pts, best_triples
    
    # Generate diverse initial configurations
    configs = []
    
    # Type 1: Line-packed configs
    for seed in range(100):
        if time.time() > deadline - 40:
            break
        pts = generate_line_packed_config(seed)
        if len(set(map(tuple, pts))) == 20 and not count_4_collinear(pts):
            configs.append(pts)
    
    # Type 2: Perturbed grids
    for seed in range(50):
        if time.time() > deadline - 40:
            break
        pts = generate_perturbed_grid(seed)
        if len(set(map(tuple, pts))) == 20 and not count_4_collinear(pts):
            configs.append(pts)
    
    # Evaluate each config with simulated annealing
    best_exact_score = 20
    best_config = None
    best_triples = 0
    
    for pts in configs:
        if time.time() > deadline - 30:
            break
        
        # Simulated annealing to improve
        improved_pts, triples = simulated_annealing(pts, max_iters=3000)
        
        if triples > best_triples:
            best_triples = triples
        
        # Compute exact score
        score = max_no_3_collinear(improved_pts)
        
        if score < best_exact_score:
            best_exact_score = score
            best_config = improved_pts
    
    if best_config is None:
        # Fallback to parabola
        best_config = [[x, x * x] for x in range(-10, 10)]
    
    return best_config
