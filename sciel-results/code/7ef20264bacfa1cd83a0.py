import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate candidate pool: Radius 4 hexagon (61 points) - sufficient for 32-point cluster
    max_radius = 4
    all_points = []
    for a in range(-max_radius, max_radius + 1):
        for b in range(-max_radius, max_radius + 1):
            if hex_dist([a, b]) <= max_radius:
                all_points.append([a, b])
    
    n_candidates = len(all_points)
    
    # Map point to index for fast lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points)}
    
    # Precompute neighbors (distance squared == 1) for each candidate
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        for j in range(i+1, n_candidates):
            if dist_sq(all_points[i], all_points[j]) == 1:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Convert neighbors to lists for faster iteration
    neighbors_list = [list(s) for s in neighbors]
    
    # Adjacency matrix for O(1) lookup
    adj = [[False] * n_candidates for _ in range(n_candidates)]
    for i in range(n_candidates):
        for j in neighbors[i]:
            adj[i][j] = True
            adj[j][i] = True

    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    candidates_sorted = sorted(range(n_candidates), key=lambda i: (hex_dist(all_points[i]), angle(all_points[i])))
    
    def compute_edges(selected_set):
        """Compute edge count for a selected set."""
        count = 0
        for i in selected_set:
            for j in neighbors_list[i]:
                if j in selected_set and j > i:
                    count += 1
        return count

    def local_search(selected_set, max_time=10.0):
        """Perform local search with proper incremental updates."""
        current_edges = compute_edges(selected_set)
        best_set = set(selected_set)
        best_edges = current_edges
        
        # Compute deg for all nodes
        deg = [0] * n_candidates
        for i in range(n_candidates):
            for n in neighbors_list[i]:
                if n in selected_set:
                    deg[i] += 1
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            for i in list(selected_set):
                if time.time() > max_time:
                    break
                
                for j in range(n_candidates):
                    if j in selected_set:
                        continue
                    
                    is_adj = adj[i][j]
                    delta = (deg[j] - (1 if is_adj else 0)) - deg[i]
                    
                    if delta > 0:
                        # Perform swap: remove i, add j
                        selected_set.discard(i)
                        selected_set.add(j)
                        
                        # Update deg: remove i from consideration
                        for k in neighbors_list[i]:
                            deg[k] -= 1
                        
                        # Update deg: add j to consideration
                        for k in neighbors_list[j]:
                            if k in selected_set:
                                deg[k] += 1
                        
                        # Update deg[i] and deg[j] explicitly
                        if is_adj:
                            deg[i] += 1
                            deg[j] -= 1
                        
                        current_edges += delta
                        
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_set = set(selected_set)
                        
                        improved = True
                        break
                if improved:
                    break
        
        return best_set, best_edges

    # Strategy: Multiple restarts with different initial configurations
    best_result = None
    best_edge_count = 0
    
    # Seed the random number generator for reproducibility
    random.seed(42)
    
    # Initial configuration 1: Greedy center-out
    initial_selected_1 = set(candidates_sorted[:32])
    result_set, edge_count = local_search(initial_selected_1, max_time=20.0)
    if edge_count > best_edge_count:
        best_edge_count = edge_count
        best_result = result_set
    
    # If we already hit the optimum, return early
    if best_edge_count >= 76:
        result = [all_points[i] for i in best_result]
        assert len(result) == 32
        assert len(set(tuple(p) for p in result)) == 32
        for p in result:
            assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        return result
    
    # Initial configuration 2: Slightly different ordering
    random.shuffle(candidates_sorted)
    initial_selected_2 = set(candidates_sorted[:32])
    result_set, edge_count = local_search(initial_selected_2, max_time=20.0)
    if edge_count > best_edge_count:
        best_edge_count = edge_count
        best_result = result_set
    
    if best_edge_count >= 76:
        result = [all_points[i] for i in best_result]
        assert len(result) == 32
        assert len(set(tuple(p) for p in result)) == 32
        for p in result:
            assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        return result
    
    # More restarts if time permits
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial configuration biased toward center
        weights = [1.0 / (hex_dist(all_points[i]) + 1) for i in range(n_candidates)]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        selected = set()
        while len(selected) < 32:
            idx = random.choices(range(n_candidates), weights=weights, k=1)[0]
            if idx not in selected:
                selected.add(idx)
        
        result_set, edge_count = local_search(selected, max_time=min(20.0, deadline - time.time() - 5.0))
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_result = result_set
        
        if best_edge_count >= 76:
            break
    
    # Final result
    result = [all_points[i] for i in best_result]
    
    # Verify
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        
    return result
