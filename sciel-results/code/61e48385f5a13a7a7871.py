import itertools
import random
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Helper functions
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    def get_collinear_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return True
        return False

    def max_no_3_collinear(points):
        """
        Find the largest subset of points with no three collinear.
        Returns the size of that subset.
        """
        n = len(points)
        # Precompute which triples are collinear
        collinear_triples = set()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        collinear_triples.add((i, j, k))
        
        # For each point, precompute which other points form a collinear triple with it
        # point_conflicts[i] = set of (j, k) pairs where (i, j, k) is collinear
        # Actually, we need: if we include i, which pairs (j,k) can we NOT include?
        # Better: for each point i, store the set of points that are "in conflict" with i
        # in the sense that some third point forms a collinear triple.
        
        # Let's use a different approach: branch and bound
        # State: which points are included, which are excluded
        # We want to maximize the number of included points with no three collinear
        
        best_size = 0
        
        # Sort points by some heuristic - maybe by degree (how many collinear triples they're in)
        point_degree = [0] * n
        for (i, j, k) in collinear_triples:
            point_degree[i] += 1
            point_degree[j] += 1
            point_degree[k] += 1
        
        # Order points: high degree first (more constrained)
        order = sorted(range(n), key=lambda x: -point_degree[x])
        
        # For efficiency, precompute for each point which other points it conflicts with
        # Two points i, j conflict if there exists k such that (i,j,k) is collinear
        # But actually, the constraint is: no three collinear. So if we pick i and j,
        # we can't pick any k that forms a collinear triple with both.
        
        # Let's use a bitmask approach for n=20
        # But 2^20 = 1M which is feasible
        
        # Actually, let's do branch and bound with good pruning
        
        # included_mask: bitmask of included points
        # excluded_mask: bitmask of excluded points
        # available: points not yet decided
        
        # Precompute: for each pair (i,j), the set of k's that form a collinear triple
        pair_conflicts = {}
        for i in range(n):
            for j in range(i + 1, n):
                conflicts = set()
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (min(i,j,k), min(max(i,j),k), max(i,j,k)) in collinear_triples or \
                       (min(i,j,k), min(max(i,j),k), max(i,j,k)) in collinear_triples:
                        # Check if i,j,k are collinear
                        if are_collinear(points[i], points[j], points[k]):
                            conflicts.add(k)
                pair_conflicts[(i,j)] = conflicts
        
        # Also need: for each triple, we know it's collinear
        # Let's recompute more carefully
        pair_conflicts = {}
        for i in range(n):
            for j in range(i + 1, n):
                conflicts = set()
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            conflicts.add(k)
                pair_conflicts[(i,j)] = conflicts
        
        # Branch and bound
        # We process points in order
        # State: index in order, set of included points, count of included
        # Pruning: if current_count + remaining_points <= best_size, prune
        
        def backtrack(idx, included, count):
            nonlocal best_size
            
            # Prune
            if count + (n - idx) <= best_size:
                return
            
            if idx == n:
                best_size = max(best_size, count)
                return
            
            p = order[idx]
            
            # Option 1: Exclude p
            backtrack(idx + 1, included, count)
            
            # Option 2: Include p
            # Check if p conflicts with any pair in included
            # p conflicts with (a,b) if a,b,p are collinear
            # i.e., if there exist a,b in included such that are_collinear(points[a], points[b], points[p])
            can_include = True
            for a in included:
                for b in included:
                    if a < b:
                        if p in pair_conflicts[(a,b)]:
                            can_include = False
                            break
                if not can_include:
                    break
            
            if can_include:
                new_included = included | {p}
                backtrack(idx + 1, new_included, count + 1)
        
        backtrack(0, set(), 0)
        return best_size

    def calculate_score(points):
        if has_4_collinear(points):
            return None # Invalid
        return max_no_3_collinear(points)

    def generate_initial_config():
        # Try to create a configuration with many collinear triples
        # Use a structured approach
        pts = []
        # Create several lines with 3 points each, sharing some points
        # Line 1: y=0, x=-10,0,10
        pts.extend([[-10, 0], [0, 0], [10, 0]])
        # Line 2: y=1, x=-9,1,11
        pts.extend([[-9, 1], [1, 1], [11, 1]])
        # Line 3: y=-1, x=-8,2,12
        pts.extend([[-8, -1], [2, -1], [12, -1]])
        # Line 4: x=0, y=-5,5,15
        pts.extend([[0, -5], [0, 5], [0, 15]])
        # Line 5: x=1, y=-4,6,16
        pts.extend([[1, -4], [1, 6], [1, 16]])
        # That's 15 points, need 5 more
        # Add points that create additional triples
        pts.extend([[-1, 2], [-2, -2], [3, 3], [4, -3], [5, 4]])
        
        # Check for duplicates and 4-collinear
        seen = set()
        unique_pts = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_pts.append(p)
        
        # Fill remaining with random points
        while len(unique_pts) < 20:
            x = random.randint(-15, 15)
            y = random.randint(-15, 15)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        
        return unique_pts[:20]

    def random_perturb(points, radius=3):
        new_pts = []
        for p in points:
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            nx = p[0] + dx
            ny = p[1] + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                new_pts.append([nx, ny])
            else:
                new_pts.append([p[0], p[1]])
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in new_pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_pts.append(p)
        # If we lost points, add random ones
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        return unique_pts[:20]

    # Main Search Loop
    best_score = 20
    best_points = None
    
    # Try multiple initial configurations
    initial_configs = []
    for _ in range(10):
        initial_configs.append(generate_initial_config())
    
    # Pick the best initial config
    for config in initial_configs:
        if len(set((p[0], p[1]) for p in config)) == 20 and not has_4_collinear(config):
            score = calculate_score(config)
            if score is not None and score < best_score:
                best_score = score
                best_points = config
                break
    
    if best_points is None:
        # Fallback to random valid
        for _ in range(1000):
            pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
            if len(set((p[0], p[1]) for p in pts)) == 20:
                if not has_4_collinear(pts):
                    best_points = pts
                    best_score = calculate_score(pts)
                    break
    
    current_points = best_points
    current_score = best_score
    
    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each point
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            original = current_points[i]
            # Try nearby positions
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx = original[0] + dx
                    ny = original[1] + dy
                    if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                        continue
                    
                    new_points = current_points.copy()
                    new_points[i] = [nx, ny]
                    
                    # Check distinct
                    if len(set((p[0], p[1]) for p in new_points)) < 20:
                        continue
                    
                    # Check 4-collinear
                    if has_4_collinear(new_points):
                        continue
                    
                    # Calculate score
                    score = calculate_score(new_points)
                    if score is not None:
                        if score < current_score:
                            current_points = new_points
                            current_score = score
                            if current_score < best_score:
                                best_score = current_score
                                best_points = current_points
                            improved = True
                            break
                        elif score == current_score:
                            # Accept with some probability to escape local minima
                            if random.random() < 0.05:
                                current_points = new_points
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        # Try swapping two points
        if not improved and time.time() - start_time < time_limit:
            for i in range(20):
                for j in range(i + 1, 20):
                    if time.time() - start_time > time_limit:
                        break
                    # Try moving point i to a random position near point j
                    for _ in range(3):
                        nx = random.randint(current_points[j][0] - 5, current_points[j][0] + 5)
                        ny = random.randint(current_points[j][1] - 5, current_points[j][1] + 5)
                        if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                            continue
                        
                        new_points = current_points.copy()
                        new_points[i] = [nx, ny]
                        
                        if len(set((p[0], p[1]) for p in new_points)) < 20:
                            continue
                        if has_4_collinear(new_points):
                            continue
                        
                        score = calculate_score(new_points)
                        if score is not None:
                            if score < current_score:
                                current_points = new_points
                                current_score = score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_points = current_points
                                improved = True
                                break
                            elif score == current_score and random.random() < 0.05:
                                current_points = new_points
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Random restart with larger perturbation
            new_points = current_points.copy()
            for _ in range(4):
                idx = random.randint(0, 19)
                dx = random.randint(-8, 8)
                dy = random.randint(-8, 8)
                nx = new_points[idx][0] + dx
                ny = new_points[idx][1] + dy
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_points[idx] = [nx, ny]
            
            if len(set((p[0], p[1]) for p in new_points)) == 20 and not has_4_collinear(new_points):
                score = calculate_score(new_points)
                if score is not None:
                    if score < current_score:
                        current_points = new_points
                        current_score = score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = current_points
                    elif score == current_score:
                        if random.random() < 0.1:
                            current_points = new_points

    return best_points if best_points else current_points

# EVOLVE-BLOCK-END
