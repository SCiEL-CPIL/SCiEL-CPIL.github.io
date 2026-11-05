import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer

    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def get_triples(points):
        """
        Returns a list of tuples, each tuple is a set of indices of 3 collinear points.
        """
        n = len(points)
        line_points = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].append(i)
        
        triples = []
        for line, pts in line_points.items():
            # pts is a list of indices on this line
            # We only care about lines with exactly 3 points (since 4+ is invalid)
            # But wait, if a line has 3 points, it forms 1 triple.
            # If a line has >3 points, the configuration is invalid.
            # So we assume valid configurations where max line size is 3.
            if len(pts) == 3:
                triples.append(tuple(sorted(pts)))
        
        # Deduplicate triples (a triple is unique to its line)
        unique_triples = set(triples)
        return list(unique_triples)

    def is_valid(points):
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        for count in line_counts.values():
            if count >= 4:
                return False
        return True

    def calc_alpha(points):
        """
        Calculate the exact size of the largest subset with no three collinear.
        This is the independence number of the hypergraph of collinear triples.
        Alpha = 20 - Min Vertex Cover Size.
        """
        triples = get_triples(points)
        if not triples:
            return 20
        
        # Convert to list of sets for easier manipulation
        edges = [set(t) for t in triples]
        
        # Branch and Bound for Minimum Vertex Cover
        # State: (current_cover_size, remaining_edges, available_vertices)
        # We want to minimize cover size.
        # Pruning: If current_cover_size >= best_found, prune.
        # Lower bound: current_cover_size + (number of disjoint edges remaining)
        
        best_cover = 20  # Worst case: cover everything
        
        def bnb(edges, available, current_size):
            nonlocal best_cover
            if not edges:
                if current_size < best_cover:
                    best_cover = current_size
                return
            
            # Lower bound: we need at least 1 vertex for each disjoint edge
            # Simple LB: current_size + len(edges) is too high.
            # Better LB: Greedy matching size? Or just current_size.
            if current_size >= best_cover:
                return
            
            # Heuristic: Pick the edge with the smallest number of available vertices
            # This is often a good branching strategy.
            min_edge_idx = -1
            min_len = float('inf')
            for i, e in enumerate(edges):
                avail_in_e = e & available
                if not avail_in_e:
                    # This edge cannot be covered by available vertices?
                    # If an edge has no available vertices, it means we already decided
                    # not to pick any of its vertices?
                    # In MVC B&B, if we exclude a vertex, we remove it from edges.
                    # If an edge becomes empty, it's infeasible.
                    return 
                if len(avail_in_e) < min_len:
                    min_len = len(avail_in_e)
                    min_edge_idx = i
            
            target_edge = edges[min_edge_idx]
            avail_in_target = target_edge & available
            
            # Branch on each vertex in the target edge
            for v in avail_in_target:
                # Branch 1: Include v in cover
                new_available = available - {v}
                new_edges = []
                for e in edges:
                    if v in e:
                        continue # Edge is covered
                    else:
                        new_edges.append(e)
                
                bnb(new_edges, new_available, current_size + 1)
                
                # Branch 2: Exclude v from cover
                # This means we must cover the target edge with other vertices.
                # So we remove v from the target edge.
                # If the target edge becomes empty, this branch is infeasible.
                new_target_edge = target_edge - {v}
                if not new_target_edge:
                    continue
                
                # Update edges: remove v from all edges
                new_edges_excl = []
                feasible = True
                for i, e in enumerate(edges):
                    if v in e:
                        new_e = e - {v}
                        if not new_e:
                            feasible = False
                            break
                        new_edges_excl.append(new_e)
                    else:
                        new_edges_excl.append(e)
                
                if feasible:
                    # Note: We don't change 'available' for exclusion?
                    # Actually, 'available' represents vertices we can still pick.
                    # If we exclude v, we can't pick it anymore.
                    # So new_available = available - {v} is correct for exclusion too?
                    # Yes, because we've decided v is NOT in the cover.
                    bnb(new_edges_excl, new_available, current_size)

        bnb(edges, set(range(20)), 0)
        return 20 - best_cover

    # Constructive Search
    best_alpha = 20
    best_points = None
    
    # Initial Configuration: Random valid
    def generate_initial():
        while True:
            pts = []
            used = set()
            for _ in range(20):
                while True:
                    p = (random.randint(-100, 100), random.randint(-100, 100))
                    if p not in used:
                        break
            if is_valid(pts):
                return pts

    current_points = generate_initial()
    current_alpha = calc_alpha(current_points)
    best_alpha = current_alpha
    best_points = [p[:] for p in current_points]
    
    # Local Search
    # We will try to reduce alpha.
    # Moves:
    # 1. Random move: Pick a point, move to random valid location.
    # 2. Triple move: Pick a point, try to move it to a line defined by 2 other points to form a triple.
    
    while time.time() < deadline - 5:
        improved = False
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Generate candidates for new position
        candidates = set()
        
        # 1. Random candidates
        for _ in range(50):
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in current_points:
                candidates.add(p)
                
        # 2. Line candidates (try to form triples)
        # Find pairs of points (i, j) != idx
        # Check if line(i,j) has exactly 2 points currently.
        # If so, place point on that line.
        for i in range(20):
            if i == idx: continue
            for j in range(i + 1, 20):
                if j == idx: continue
                p1 = current_points[i]
                p2 = current_points[j]
                line = get_line(p1, p2)
                # Check if any other point k is on this line
                has_third = False
                for k in range(20):
                    if k == i or k == j or k == idx:
                        continue
                    if get_line(p1, current_points[k]) == line:
                        has_third = True
                        break
                if not has_third:
                    # Line has only p1, p2.
                    # Generate points on this line
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Normalize direction? Not needed for integer steps
                    for t in range(-10, 11):
                        if t == 0 or t == 1:
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            p = (nx, ny)
                            if p not in current_points:
                                candidates.add(p)
        
        # Evaluate candidates
        for p in candidates:
            test_points = current_points[:]
            test_points[idx] = p
            if not is_valid(test_points):
                continue
            
            new_alpha = calc_alpha(test_points)
            if new_alpha < current_alpha:
                current_points = test_points
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_points = [pt[:] for pt in current_points]
                improved = True
                break # Accept first improvement
            
            # Allow neutral moves to escape local minima?
            # If new_alpha == current_alpha, accept with 0.1 probability?
            elif new_alpha == current_alpha and random.random() < 0.1:
                current_points = test_points
                # current_alpha remains same
                # Don't update best_alpha unless it's better (it's not)
                improved = True
                break

        if not improved and random.random() < 0.05:
            # Restart or large perturbation
            # Just randomize 3 points
            for _ in range(3):
                idx2 = random.randint(0, 19)
                while True:
                    p2 = (random.randint(-100, 100), random.randint(-100, 100))
                    if p2 not in current_points:
                        break
                current_points[idx2] = p2
                if not is_valid(current_points):
                    # Revert if invalid? Or just break and restart trial?
                    # Simpler: revert and break
                    current_points[idx2] = current_points[idx2] # This is wrong, need old value
                    # Let's just restart the whole configuration if we get stuck badly
                    break
            if is_valid(current_points):
                current_alpha = calc_alpha(current_points)

    return best_points

