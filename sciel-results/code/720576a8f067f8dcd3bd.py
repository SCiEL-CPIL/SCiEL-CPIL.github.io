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

    # Generate candidate pool: Radius 4 hexagon (61 points)
    all_points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist([a, b]) <= 4:
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

    # Initial configuration: Greedy center-out
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    candidates_sorted = sorted(range(n_candidates), key=lambda i: (hex_dist(all_points[i]), angle(all_points[i])))
    
    # Select first 32 points
    initial_selected = set(candidates_sorted[:32])
    selected_set = set(initial_selected)
    
    # Compute initial edge count
    current_edges = count_edges([all_points[i] for i in selected_set])
    
    # Maintain degrees: deg[k] = number of neighbors of k in selected_set
    deg = [0] * n_candidates
    for k in range(n_candidates):
        for n in neighbors[k]:
            if n in selected_set:
                deg[k] += 1
    
    best_edges = current_edges
    best_selected = set(selected_set)
    
    # Hill climbing with proper delta updates
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        for i in sorted(selected_set):
            if time.time() > deadline - 30.0:
                break
            for j in range(n_candidates):
                if j in selected_set:
                    continue
                
                # Calculate delta: gain from adding j, loss from removing i
                # deg[j] = number of neighbors of j in current selected_set
                # is_adj = 1 if i and j are adjacent (would be double-counted)
                is_adj = 1 if j in neighbors[i] else 0
                
                # When we remove i and add j:
                # - We lose deg[i] edges from i to other selected nodes
                # - We gain deg[j] edges from j to other selected nodes
                # - But if i and j are adjacent, we'd count that edge in both, so subtract 1
                delta = (deg[j] - is_adj) - deg[i]
                
                if delta > 0:
                    # Perform swap: remove i, add j
                    selected_set.remove(i)
                    selected_set.add(j)
                    
                    # Update degrees for all affected nodes
                    # Affected: i, j, and neighbors of i and j
                    affected = set([i, j])
                    affected.update(neighbors[i])
                    affected.update(neighbors[j])
                    
                    for k in affected:
                        count = 0
                        for n in neighbors[k]:
                            if n in selected_set:
                                count += 1
                        deg[k] = count
                    
                    current_edges += delta
                    improved = True
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_selected = set(selected_set)
                    break
            if improved:
                break
    
    # Simulated annealing phase to escape local optima
    if time.time() < deadline - 10.0:
        sa_start = time.time()
        sa_deadline = deadline - 10.0
        
        # Start from best solution found
        selected_set = set(best_selected)
        current_edges = best_edges
        
        # Recompute degrees
        deg = [0] * n_candidates
        for k in range(n_candidates):
            for n in neighbors[k]:
                if n in selected_set:
                    deg[k] += 1
        
        # SA parameters
        temp = 2.0
        cool_rate = 0.9995
        min_temp = 0.01
        
        while temp > min_temp and time.time() < sa_deadline:
            # Pick random i in selected, random j not in selected
            selected_list = list(selected_set)
            unselected_list = [x for x in range(n_candidates) if x not in selected_set]
            
            if not unselected_list:
                break
            
            i = selected_list[random.randint(0, len(selected_list)-1)]
            j = unselected_list[random.randint(0, len(unselected_list)-1)]
            
            is_adj = 1 if j in neighbors[i] else 0
            delta = (deg[j] - is_adj) - deg[i]
            
            # Accept if improvement or with probability exp(-delta/temp) for worsening
            if delta >= 0 or random.random() < math.exp(-delta / temp):
                # Perform swap
                selected_set.remove(i)
                selected_set.add(j)
                
                # Update degrees
                affected = set([i, j])
                affected.update(neighbors[i])
                affected.update(neighbors[j])
                
                for k in affected:
                    count = 0
                    for n in neighbors[k]:
                        if n in selected_set:
                            count += 1
                    deg[k] = count
                
                current_edges += delta
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_selected = set(selected_set)
            
            temp *= cool_rate
    
    # Final result
    result = [all_points[i] for i in best_selected]
    
    # Verify
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        
    return result
