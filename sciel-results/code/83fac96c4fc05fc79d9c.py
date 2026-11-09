import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill (contract says 120s, but let's be safe)

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges(points):
        n = len(points)
        edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Directions on triangular lattice (distance 1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Precompute neighbors for a point to speed up checks
    def get_neighbors(a, b):
        return [(a + da, b + db) for da, db in dirs]

    def is_valid_placement(points, idx, new_point):
        """Check if new_point can replace points[idx] without violating distance >= 1"""
        a, b = new_point
        if abs(a) > 64 or abs(b) > 64:
            return False
        for j in range(len(points)):
            if j == idx:
                continue
            if dist_sq(new_point, points[j]) < 1:
                return False
        return True

    def incremental_edge_delta(points, idx, old_point, new_point):
        """Calculate change in edge count if points[idx] moves from old_point to new_point"""
        delta = 0
        for j in range(len(points)):
            if j == idx:
                continue
            d_old = dist_sq(old_point, points[j])
            d_new = dist_sq(new_point, points[j])
            # Remove old edges
            if d_old == 1:
                delta -= 1
            # Add new edges
            if d_new == 1:
                delta += 1
        return delta

    def local_search(points, max_iters, deadline, seed):
        """Perform local search with simulated annealing and steepest descent."""
        random.seed(seed)
        points = [list(p) for p in points]
        n = len(points)
        current_edges = count_edges(points)
        best_points = [list(p) for p in points]
        best_edges = current_edges
        
        T_start = 5.0
        T_end = 0.01
        T = T_start
        cooling_rate = 0.995
        
        # Extended move set for SA: 1-step, 2-step, 3-step in lattice dirs
        move_dirs = []
        for da, db in dirs:
            for k in [1, 2, 3]:
                move_dirs.append((k*da, k*db))
        # Add some random jumps
        for _ in range(10):
            move_dirs.append((random.randint(-5, 5), random.randint(-5, 5)))

        no_improve = 0
        max_no_improve = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                # Restart from best
                points = [list(p) for p in best_points]
                current_edges = best_edges
                T = T_start
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)
                continue

            # Pick a random point
            i = random.randint(0, n - 1)
            old_point = (points[i][0], points[i][1])
            
            best_candidate = None
            best_candidate_edges = current_edges
            
            # Try moves
            for da, db in move_dirs:
                new_point = (old_point[0] + da, old_point[1] + db)
                if not is_valid_placement(points, i, new_point):
                    continue
                
                delta_e = incremental_edge_delta(points, i, old_point, new_point)
                new_edges = current_edges + delta_e
                
                if new_edges > best_candidate_edges:
                    best_candidate_edges = new_edges
                    best_candidate = new_point
                elif delta_e < 0:
                    # Accept worse moves with probability exp(delta_e / T)
                    if random.random() < math.exp(delta_e / max(T, 1e-10)):
                        if new_edges > best_candidate_edges:
                            best_candidate_edges = new_edges
                            best_candidate = new_point
            
            if best_candidate is not None:
                points[i][0] = best_candidate[0]
                points[i][1] = best_candidate[1]
                current_edges = best_candidate_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = [list(p) for p in points]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T *= cooling_rate

        return best_points, best_edges

    def steepest_descent(points, max_iters, deadline):
        """Polish with steepest descent (best move only)."""
        points = [list(p) for p in points]
        n = len(points)
        current_edges = count_edges(points)
        best_points = [list(p) for p in points]
        best_edges = current_edges
        
        # Move set: 1-step neighbors
        move_dirs = dirs
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            best_move_delta = 0
            best_move_i = -1
            best_move_new = None
            
            for i in range(n):
                old_point = (points[i][0], points[i][1])
                for da, db in move_dirs:
                    new_point = (old_point[0] + da, old_point[1] + db)
                    if not is_valid_placement(points, i, new_point):
                        continue
                    delta_e = incremental_edge_delta(points, i, old_point, new_point)
                    if delta_e > best_move_delta:
                        best_move_delta = delta_e
                        best_move_i = i
                        best_move_new = new_point
            
            if best_move_i != -1 and best_move_delta > 0:
                points[best_move_i][0] = best_move_new[0]
                points[best_move_i][1] = best_move_new[1]
                current_edges += best_move_delta
                improved = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = [list(p) for p in points]
            else:
                if not improved:
                    break

        return best_points, best_edges

    # --- Generate Initial Candidates ---
    
    candidates = []
    
    # 1. Hexagonal Rings (Current approach, refined)
    # Build rings 0 to 4
    visited = set()
    rings = {0: [(0, 0)]}
    visited.add((0, 0))
    current_layer = [(0, 0)]
    for r in range(1, 5):
        next_layer = []
        for p in current_layer:
            a, b = p
            for da, db in dirs:
                np = (a + da, b + db)
                if np not in visited:
                    visited.add(np)
                    next_layer.append(np)
        rings[r] = next_layer
        current_layer = next_layer
    
    # Select 32 points from rings
    # Rings 0, 1, 2 have 1+6+12 = 19 points.
    # Ring 3 has 18 points. We need 13 from Ring 3.
    base_points = []
    for r in range(3):
        base_points.extend(rings[r])
    base_set = set(base_points)
    
    ring3 = rings[3]
    
    def angle(p):
        a, b = p
        x = a + 0.5 * b
        y = math.sqrt(3) / 2 * b
        return math.atan2(y, x)
    
    ring3_sorted = sorted(ring3, key=angle)
    n_ring = len(ring3_sorted)
    
    neighbor_counts = {}
    for p in ring3:
        a, b = p
        cnt = 0
        for da, db in dirs:
            if (a + da, b + db) in base_set:
                cnt += 1
        neighbor_counts[p] = cnt
    
    # Find best contiguous block of 13 in the cycle
    k = 13
    best_score = -1
    best_start = 0
    for i in range(n_ring):
        s = 0
        for j in range(k):
            idx = (i + j) % n_ring
            s += neighbor_counts[ring3_sorted[idx]]
        if s > best_score:
            best_score = s
            best_start = i
    
    selected_ring3 = []
    for j in range(k):
        idx = (best_start + j) % n_ring
        selected_ring3.append(ring3_sorted[idx])
    
    final_ring_points = base_points + selected_ring3
    candidates.append(final_ring_points)
    
    # 2. Spiral Growth
    # Start at (0,0), add neighbors in a spiral order
    # Order of dirs for spiral: E, NE, N, NW, W, SW
    spiral_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def generate_spiral(n_points):
        points = [(0, 0)]
        if n_points == 1:
            return points
        visited = {(0, 0)}
        # We need to add points in a way that keeps the cluster compact.
        # A simple greedy approach: at each step, add the unvisited neighbor 
        # of the current cluster that maximizes the number of neighbors in the cluster.
        # If tie, prefer the one with smaller "radius" or specific angle.
        
        for _ in range(n_points - 1):
            best_p = None
            best_score = -1
            # Collect all boundary candidates
            candidates = []
            for p in points:
                for da, db in spiral_dirs:
                    np = (p[0] + da, p[1] + db)
                    if np not in visited:
                        candidates.append(np)
            
            # Score each candidate
            for c in candidates:
                score = 0
                for p in points:
                    if dist_sq(c, p) == 1:
                        score += 1
                # Tie-breaker: prefer points closer to origin?
                # dist_sq(c, (0,0))
                d_origin = dist_sq(c, (0, 0))
                # Combine score and distance
                # We want max score, min distance
                # Use tuple comparison: (score, -d_origin)
                if score > best_score or (score == best_score and d_origin < dist_sq(best_p, (0,0)) if best_p else False):
                    if best_p is None or score > best_score or (score == best_score and d_origin < dist_sq(best_p, (0,0))):
                        best_p = c
                        best_score = score
            
            if best_p is not None:
                points.append(best_p)
                visited.add(best_p)
            else:
                # Fallback if no candidates (should not happen for small n)
                points.append((len(points), 0))
        
        return points

    spiral_points = generate_spiral(32)
    candidates.append(spiral_points)
    
    # 3. Random Cluster
    # Generate a random set of 32 points in a 10x10 box, ensure validity
    def generate_random_cluster(n_points, seed):
        random.seed(seed)
        points = []
        visited = set()
        # Start with a random center
        cx = random.randint(-5, 5)
        cy = random.randint(-5, 5)
        points.append((cx, cy))
        visited.add((cx, cy))
        
        for _ in range(n_points - 1):
            found = False
            for _ in range(100): # 100 attempts to find a valid neighbor
                # Pick a random existing point
                p = random.choice(points)
                # Pick a random direction
                da, db = random.choice(dirs)
                np = (p[0] + da, p[1] + db)
                if np in visited:
                    continue
                # Check distance to all others
                valid = True
                for q in points:
                    if dist_sq(np, q) < 1:
                        valid = False
                        break
                if valid:
                    points.append(np)
                    visited.add(np)
                    found = True
                    break
            if not found:
                # If stuck, try adding a point further away
                for _ in range(100):
                    np = (random.randint(-10, 10), random.randint(-10, 10))
                    if np in visited:
                        continue
                    valid = True
                    for q in points:
                        if dist_sq(np, q) < 1:
                            valid = False
                            break
                    if valid:
                        points.append(np)
                        visited.add(np)
                        found = True
                        break
            if not found:
                # Force add (might violate constraints, but local search will fix)
                points.append((random.randint(-10, 10), random.randint(-10, 10)))
        return points

    candidates.append(generate_random_cluster(32, 123))
    candidates.append(generate_random_cluster(32, 456))

    # --- Optimize Candidates ---
    
    best_overall_points = None
    best_overall_edges = 0
    
    # Time allocation
    time_per_candidate = (DEADLINE - (time.time() - start_time) - 5.0) / len(candidates)
    time_per_candidate = max(time_per_candidate, 10.0) # Min 10s per candidate
    
    for i, cand in enumerate(candidates):
        if time.time() > DEADLINE - 5:
            break
            
        # 1. Local Search (SA)
        sa_points, sa_edges = local_search(cand, 5000, time.time() + time_per_candidate * 0.6, seed=1000 + i)
        
        # 2. Steepest Descent Polish
        sd_points, sd_edges = steepest_descent(sa_points, 1000, time.time() + time_per_candidate * 0.4)
        
        if sd_edges > best_overall_edges:
            best_overall_edges = sd_edges
            best_overall_points = sd_points
            
        # If we found the optimum, we can stop early
        if best_overall_edges == 76:
            break

    # Final Polish on the best result with remaining time
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5 and best_overall_points:
        # Run one more SA cycle on the best result
        final_points, final_edges = local_search(best_overall_points, 10000, time.time() + remaining_time - 2, seed=9999)
        if final_edges > best_overall_edges:
            best_overall_points = final_points
            best_overall_edges = final_edges
            
        # Final Steepest Descent
        final_points, final_edges = steepest_descent(best_overall_points, 2000, time.time() + remaining_time - 1)
        if final_edges > best_overall_edges:
            best_overall_points = final_points
            best_overall_edges = final_edges

    if best_overall_points is None:
        # Fallback
        best_overall_points = [[a, b] for a in range(16) for b in range(2)]

    return best_overall_points
