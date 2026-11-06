import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    random.seed(42)  # Deterministic for reproducibility

    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + db * db + da * db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def edge_delta(old_pt, new_pt, point_set, exclude_idx, points):
        """Calculate change in edge count when moving point from old_pt to new_pt"""
        delta = 0
        for j, pt in enumerate(points):
            if j == exclude_idx:
                continue
            if dist_sq(old_pt, pt) == 1:
                delta -= 1
            if dist_sq(new_pt, pt) == 1:
                delta += 1
        return delta

    # Generate all candidate points within hex distance 4
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist(a, b) <= 4:
                candidates.append((a, b))
    
    # Precompute adjacency: for each candidate, which other candidates are at distance 1
    cand_index = {p: i for i, p in enumerate(candidates)}
    n_cand = len(candidates)
    adj = [[] for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if dist_sq(candidates[i], candidates[j]) == 1:
                adj[i].append(j)
                adj[j].append(i)
    
    # Initial configuration: hexagonal spiral (rings 0,1,2 full + 13 from ring 3)
    # Ring 0
    init_pts = [(0, 0)]
    
    # Ring 1
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    init_pts.extend(ring1)
    
    # Ring 2
    ring2 = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) == 2:
                ring2.append((a, b))
    init_pts.extend(ring2)
    
    # Ring 3: select 13 points that maximize connectivity
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    # Sort by angle for consistent ordering
    def angle_key(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    ring3.sort(key=angle_key)
    init_pts.extend(ring3[:13])
    
    assert len(init_pts) == 32, f"Expected 32 points, got {len(init_pts)}"
    assert len(set(init_pts)) == 32, "Points not distinct"
    for a, b in init_pts:
        assert abs(a) <= 64 and abs(b) <= 64
    
    # Convert to bitmask representation for fast operations
    init_mask = 0
    for p in init_pts:
        if p in cand_index:
            init_mask |= (1 << cand_index[p])
    
    def mask_to_points(mask):
        pts = []
        for i in range(n_cand):
            if mask & (1 << i):
                pts.append(candidates[i])
        return pts
    
    def count_edges_from_mask(mask):
        count = 0
        for i in range(n_cand):
            if mask & (1 << i):
                for j in adj[i]:
                    if j > i and (mask & (1 << j)):
                        count += 1
        return count
    
    def edge_delta_mask(mask, old_idx, new_idx):
        """Calculate edge count change when swapping old_idx to new_idx in mask"""
        delta = 0
        for j in adj[old_idx]:
            if j != new_idx and (mask & (1 << j)):
                delta -= 1
        for j in adj[new_idx]:
            if j != old_idx and (mask & (1 << j)):
                delta += 1
        return delta
    
    current_mask = init_mask
    current_edges = count_edges_from_mask(current_mask)
    best_mask = current_mask
    best_edges = current_edges
    
    # If already optimal, return immediately
    if current_edges >= 76:
        return [list(p) for p in mask_to_points(current_mask)]
    
    # Simulated annealing
    T_start = 2.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # Build list of valid moves: (old_idx, new_idx) where new_idx is empty
    def get_possible_moves(mask):
        moves = []
        for i in range(n_cand):
            if mask & (1 << i):
                # Try moving to adjacent empty positions
                for j in adj[i]:
                    if not (mask & (1 << j)):
                        moves.append((i, j))
                # Also try positions at hex distance 2 (for larger jumps)
                for k in range(n_cand):
                    if k != i and not (mask & (1 << k)):
                        if hex_dist(candidates[i][0], candidates[i][1], 
                                   candidates[k][0], candidates[k][1]) == 2:
                            moves.append((i, k))
        return moves
    
    T = T_start
    no_improve_count = 0
    
    while T > T_end and time.time() < deadline:
        # Generate a random move
        if current_mask:
            occupied = [i for i in range(n_cand) if current_mask & (1 << i)]
            if occupied:
                old_idx = random.choice(occupied)
                # Choose a random empty position near old_idx
                old_pt = candidates[old_idx]
                # Find all empty positions within hex distance 3
                possible_new = []
                for k in range(n_cand):
                    if not (current_mask & (1 << k)) and k != old_idx:
                        if hex_dist(old_pt[0], old_pt[1], candidates[k][0], candidates[k][1]) <= 3:
                            possible_new.append(k)
                
                if possible_new:
                    new_idx = random.choice(possible_new)
                    delta = edge_delta_mask(current_mask, old_idx, new_idx)
                    
                    # Accept if better, or with probability exp(-delta/T) if worse
                    if delta > 0 or random.random() < math.exp(-delta / T):
                        new_mask = current_mask
                        new_mask &= ~(1 << old_idx)
                        new_mask |= (1 << new_idx)
                        current_mask = new_mask
                        current_edges += delta
                        
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_mask = current_mask
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling_rate
    
    # Final local search polish on best solution
    best_points = mask_to_points(best_mask)
    best_point_set = set(best_points)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_points)):
            if time.time() > deadline:
                break
            ax, ay = best_points[i]
            # Try all empty positions within hex distance 2
            for a2 in range(ax - 2, ax + 3):
                for b2 in range(ay - 2, ay + 3):
                    if a2 == ax and b2 == ay:
                        continue
                    if (a2, b2) in best_point_set:
                        continue
                    if abs(a2) > 64 or abs(b2) > 64:
                        continue
                    
                    new_pt = (a2, b2)
                    delta = 0
                    for j in range(len(best_points)):
                        if j == i:
                            continue
                        if dist_sq((ax, ay), best_points[j]) == 1:
                            delta -= 1
                        if dist_sq(new_pt, best_points[j]) == 1:
                            delta += 1
                    
                    if delta > 0:
                        best_point_set.remove((ax, ay))
                        best_point_set.add(new_pt)
                        best_points[i] = new_pt
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    result = [list(p) for p in best_points]
    
    # Final verification
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for a, b in result:
        assert abs(a) <= 64 and abs(b) <= 64
    
    return result
