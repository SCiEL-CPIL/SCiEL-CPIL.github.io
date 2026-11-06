import time
import random
import itertools

def construct():
    start_time = time.time()
    timeout = 135  # Safety margin before 150s kill

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        # For each pair, count collinear points
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 2: # i, j, and 2 others -> 4 total
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.add((i, j, k))
        return list(triples)

    def solve_min_vertex_cover(triples, n_points):
        """
        Branch and bound to find the size of the minimum vertex cover
        of the 3-uniform hypergraph defined by 'triples'.
        Returns the size of the MVC.
        """
        if not triples:
            return 0
        
        # Pre-compute point frequencies for heuristic ordering
        point_freq = [0] * n_points
        for t in triples:
            for p in t:
                point_freq[p] += 1
        
        best_cover = n_points
        # Initial upper bound: Greedy set cover
        def greedy_cover(triples_list):
            if not triples_list:
                return 0
            remaining = set(triples_list)
            cover = 0
            while remaining:
                # Find point appearing in most remaining triples
                counts = [0] * n_points
                for t in remaining:
                    for p in t:
                        counts[p] += 1
                best_p = max(range(n_points), key=lambda i: counts[i])
                cover += 1
                # Remove all triples containing best_p
                remaining = {t for t in remaining if best_p not in t}
            return cover

        best_cover = min(best_cover, greedy_cover(triples))
        
        # Convert triples to set for fast removal
        triples_set = set(triples)
        
        def backtrack(remaining_set, current_cover):
            nonlocal best_cover
            
            if not remaining_set:
                if current_cover < best_cover:
                    best_cover = current_cover
                return
                
            if current_cover >= best_cover:
                return
                
            # Lower bound: 
            # If we have 'k' disjoint edges, we need at least k more vertices.
            # Finding max matching in 3-uniform hypergraph is hard, but we can approximate.
            # Simple bound: len(remaining_set) / 3 is a lower bound? No, one vertex covers 3 edges.
            # So we need at least ceil(len(remaining_set) / max_degree_in_remaining)?
            # Let's use a simpler bound: if current_cover + 1 >= best_cover, we might prune, but we need to be careful.
            
            # Heuristic: Pick an edge with the point having highest frequency in remaining_set?
            # Or just pick the first edge. To improve, sort edges by "tightness".
            
            # Let's pick an edge arbitrarily but consistently for determinism in B&B
            # Actually, picking an edge that contains a point with high degree is better.
            
            # Calculate degrees in remaining_set
            deg = [0] * n_points
            for t in remaining_set:
                for p in t:
                    deg[p] += 1
            
            # Pick edge with max sum of degrees? Or just first edge containing max degree point.
            best_edge = None
            max_deg_sum = -1
            for t in remaining_set:
                s = deg[t[0]] + deg[t[1]] + deg[t[2]]
                if s > max_deg_sum:
                    max_deg_sum = s
                    best_edge = t
            
            if best_edge is None:
                return

            u, v, w = best_edge
            
            # Branch 1: Pick u
            new_set_1 = set()
            for t in remaining_set:
                if u not in t:
                    new_set_1.add(t)
            backtrack(new_set_1, current_cover + 1)
            
            if current_cover + 1 >= best_cover:
                return

            # Branch 2: Pick v
            new_set_2 = set()
            for t in remaining_set:
                if v not in t:
                    new_set_2.add(t)
            backtrack(new_set_2, current_cover + 1)
            
            if current_cover + 1 >= best_cover:
                return

            # Branch 3: Pick w
            new_set_3 = set()
            for t in remaining_set:
                if w not in t:
                    new_set_3.add(t)
            backtrack(new_set_3, current_cover + 1)

        backtrack(triples_set, 0)
        return best_cover

    def evaluate(points):
        """
        Returns (mvc_size, triple_count).
        Higher MVC size is better (means smaller independent set).
        """
        if has_4_collinear(points):
            return -1, 0 # Invalid
        
        triples = get_collinear_triples(points)
        mvc_size = solve_min_vertex_cover(triples, 20)
        return mvc_size, len(triples)

    def generate_grid_perturbed(seed):
        random.seed(seed)
        pts = []
        # 5x4 grid base
        for i in range(5):
            for j in range(4):
                x = -10 + i * 5 + random.randint(-1, 1)
                y = -10 + j * 5 + random.randint(-1, 1)
                pts.append([x, y])
        
        # Perturb to break 4-collinearities
        # We check and adjust locally
        for _ in range(100):
            if not has_4_collinear(pts):
                break
            # Find a violating 4-set and move one point
            # Simple: move a random point
            idx = random.randint(0, 19)
            pts[idx][0] += random.randint(-3, 3)
            pts[idx][1] += random.randint(-3, 3)
            # Bounds check
            if not (-100 <= pts[idx][0] <= 100 and -100 <= pts[idx][1] <= 100):
                pts[idx][0] = -100 + random.randint(0, 200)
                pts[idx][1] = -100 + random.randint(0, 200)
            
            # Ensure distinct
            seen = set()
            new_pts = []
            for p in pts:
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    new_pts.append(p)
            while len(new_pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    new_pts.append([x, y])
                    seen.add((x, y))
            pts = new_pts[:20]
            
        return pts

    def local_search(points, time_limit):
        current_pts = [p[:] for p in points]
        current_mvc, current_triples = evaluate(current_pts)
        
        if current_mvc == -1:
             return current_pts, 0, 0

        best_mvc = current_mvc
        best_triples = current_triples
        best_pts = [p[:] for p in current_pts]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > time_limit:
                    break
                old_x, old_y = current_pts[i]
                
                # Try a few moves
                for _ in range(10):
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        continue
                    nx = old_x + dx
                    ny = old_y + dy
                    if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                        continue
                    
                    # Check collision
                    collision = False
                    for j, p in enumerate(current_pts):
                        if j != i and p[0] == nx and p[1] == ny:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    current_pts[i] = [nx, ny]
                    new_mvc, new_triples = evaluate(current_pts)
                    
                    if new_mvc == -1:
                        current_pts[i] = [old_x, old_y]
                        continue
                    
                    # Accept if MVC increases, or if MVC same but triples increase
                    if new_mvc > current_mvc or (new_mvc == current_mvc and new_triples > current_triples):
                        current_mvc = new_mvc
                        current_triples = new_triples
                        improved = True
                        if current_mvc > best_mvc or (current_mvc == best_mvc and current_triples > best_triples):
                            best_mvc = current_mvc
                            best_triples = current_triples
                            best_pts = [p[:] for p in current_pts]
                        break # Move to next point
                    else:
                        current_pts[i] = [old_x, old_y]
                        
        return best_pts, best_mvc, best_triples

    best_score = 0 # MVC size
    best_points = None
    best_triples_count = 0

    # Generate candidates
    candidates = []
    for seed in range(20):
        pts = generate_grid_perturbed(seed)
        if not has_4_collinear(pts):
            candidates.append(pts)
            
    # Also try some random dense ones
    for seed in range(10):
        random.seed(100 + seed)
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        if not has_4_collinear(pts):
            candidates.append(pts)

    # Evaluate and Local Search
    deadline = start_time + timeout - 10
    
    for cand in candidates:
        if time.time() > deadline:
            break
            
        # Initial evaluation
        mvc, tric = evaluate(cand)
        if mvc == -1:
            continue
            
        # Local search
        ls_pts, ls_mvc, ls_tric = local_search(cand, min(time.time() + 5, deadline))
        
        if ls_mvc > best_score or (ls_mvc == best_score and ls_tric > best_triples_count):
            best_score = ls_mvc
            best_triples_count = ls_tric
            best_points = ls_pts
            
    # Fallback
    if best_points is None:
        # Parabola
        pts = [[x, x*x] for x in range(-5, 6)]
        pts += [[x, x*x + 25] for x in range(-5, 6)]
        best_points = pts
        # Ensure valid
        if has_4_collinear(best_points):
             # Just random
             random.seed(1)
             pts = []
             used = set()
             while len(pts) < 20:
                 x = random.randint(-100, 100)
                 y = random.randint(-100, 100)
                 if (x,y) not in used:
                     pts.append([x,y])
                     used.add((x,y))
             best_points = pts

    return best_points
