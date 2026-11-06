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
        # Check all lines defined by pairs
        # Optimization: Use a set of lines? Or just count for each pair.
        # For N=20, pairs = 190. For each pair, check other 18 points. 190*18 = 3420 ops. Fast.
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

    def min_vertex_cover_3uniform(triples, n_points):
        """
        Returns the size of the minimum vertex cover of the 3-uniform hypergraph.
        The score is n_points - min_vertex_cover.
        """
        if not triples:
            return 0
        
        # Convert triples to a set for faster lookup
        triples_set = set(triples)
        
        best_cover = n_points # Upper bound
        
        # Greedy heuristic for initial upper bound
        # Pick point in most triples, remove it, repeat
        def greedy_cover(triples_list):
            if not triples_list:
                return 0
            counts = {i: 0 for i in range(n_points)}
            for t in triples_list:
                for p in t:
                    counts[p] += 1
            cover = 0
            remaining = set(triples_list)
            while remaining:
                # Find point in remaining triples with max count
                local_counts = {i: 0 for i in range(n_points)}
                for t in remaining:
                    for p in t:
                        local_counts[p] += 1
                best_point = max(local_counts, key=local_counts.get)
                cover += 1
                # Remove all triples containing best_point
                remaining = {t for t in remaining if best_point not in t}
            return cover
        
        initial_cover = greedy_cover(triples_set)
        best_cover = min(best_cover, initial_cover)
        
        def backtrack(remaining_triples, current_cover):
            nonlocal best_cover
            if not remaining_triples:
                if current_cover < best_cover:
                    best_cover = current_cover
                return
            
            if current_cover >= best_cover:
                return
                
            # Heuristic: Pick a triple with points that appear in many other triples?
            # Or just pick the first one.
            # To improve pruning, pick a triple where one of the points is "critical" (in many triples)
            # Let's just pick the first triple for simplicity, but sort triples by frequency of points?
            # For N=20, simple B&B is often fast enough.
            
            # Optimization: If any point is in all remaining triples, we must pick it.
            # Check for "forced" points
            # This might be expensive. Let's stick to branching on a triple.
            
            t = next(iter(remaining_triples))
            
            # Branch 1: Pick t[0]
            new_remaining_1 = {rt for rt in remaining_triples if t[0] not in rt}
            backtrack(new_remaining_1, current_cover + 1)
            
            # Branch 2: Pick t[1]
            new_remaining_2 = {rt for rt in remaining_triples if t[1] not in rt}
            backtrack(new_remaining_2, current_cover + 1)
            
            # Branch 3: Pick t[2]
            new_remaining_3 = {rt for rt in remaining_triples if t[2] not in rt}
            backtrack(new_remaining_3, current_cover + 1)

        backtrack(triples_set, 0)
        return best_cover

    def calculate_score(points):
        if has_4_collinear(points):
            return None # Invalid
        triples = get_collinear_triples(points)
        cover_size = min_vertex_cover_3uniform(triples, 20)
        return 20 - cover_size

    def generate_initial_config():
        # Start with a 4x5 grid
        pts = []
        for i in range(4):
            for j in range(5):
                x = -10 + i * 5
                y = -10 + j * 5
                pts.append([x, y])
        return pts

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
    
    # Initial configuration
    current_points = generate_initial_config()
    
    # Try to fix initial config
    # If invalid, perturb
    for attempt in range(50):
        if not has_4_collinear(current_points):
            break
        current_points = random_perturb(current_points, radius=2)
    
    if has_4_collinear(current_points):
        # Fallback to random valid
        for _ in range(1000):
            pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
            # Check distinct
            if len(set((p[0], p[1]) for p in pts)) == 20:
                if not has_4_collinear(pts):
                    current_points = pts
                    break

    current_score = calculate_score(current_points)
    if current_score is not None and current_score < best_score:
        best_score = current_score
        best_points = current_points

    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each point
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            original = current_points[i]
            # Try nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
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
                            # Tie-breaker: prefer more triples (might help future moves)
                            # Count triples
                            old_triples = len(get_collinear_triples(current_points))
                            new_triples = len(get_collinear_triples(new_points))
                            if new_triples > old_triples:
                                current_points = new_points
                                # Score is same, but structure might be better
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Random restart or larger perturbation
            # Perturb a random subset
            new_points = current_points.copy()
            for _ in range(3):
                idx = random.randint(0, 19)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
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
                        # Accept with some probability (Simulated Annealing style)
                        if random.random() < 0.1:
                            current_points = new_points

    return best_points if best_points else current_points

# EVOLVE-BLOCK-END
