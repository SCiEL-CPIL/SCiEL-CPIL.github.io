import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Target area: we'll try to maximize multiplicity of a specific area
    # For 30 points, C(30,3) = 4060 triples. If all had the same area, multiplicity would be 4060.
    # But we exclude zero-area.
    
    # Strategy:
    # 1. Start with a configuration known to have high multiplicity for a specific area.
    #    A lattice grid often produces many triangles of the same area.
    #    Let's try to find a configuration where many triangles have area A.
    #    
    #    Alternative: Local search that specifically tries to increase the count of the most frequent area.
    #    
    #    Let's use a smarter local search:
    #    - Maintain the full count of areas.
    #    - When moving a point, calculate the delta in the count of the *current best area*.
    #    - If the move increases the count of the current best area (or creates a new best area), accept.
    #    - Also accept moves that keep the count of the current best area the same (to explore plateau).
    #    - Use simulated annealing or just greedy with random restarts.
    
    # Let's try a different approach:
    # Construct points on a line? No, distinct points.
    # Construct points on a circle? 
    # 
    # Actually, the "many equal area triangles" problem is related to finding large sets of points with many congruent triangles or equal area triangles.
    # 
    # Let's stick to the local search but optimize the inner loop heavily using NumPy if possible, or just tight Python loops.
    # The current program is likely bottlenecked by the O(N^2) update in get_new_score.
    # N=30, so O(900) per move. 4060 triples total.
    # 
    # Let's try to precompute the contribution of each point to the area counts?
    # Area(i, j, k) depends on all three.
    # If we move point i, all triples containing i change.
    # Number of triples containing i is C(29, 2) = 406.
    # So we need to update 406 counts.
    # This is fast.
    
    # Let's implement a fast local search.
    
    def compute_all_areas(points):
        counts = Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if d > 0:
                        counts[d] += 1
        return counts

    def make_grid(w, h, ox, oy):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x + ox, y + oy))
        return pts

    # Initial candidates
    # 6x5 grid
    # 5x6 grid
    # 10x3 grid
    # 3x10 grid
    # Also try some random configurations
    
    inits = [
        make_grid(6, 5, -2, -2),
        make_grid(5, 6, -2, -2),
        make_grid(10, 3, -4, -1),
        make_grid(3, 10, -1, -4),
    ]
    
    # Add a few random inits
    all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for _ in range(5):
        random.shuffle(all_pts)
        # Pick 30 distinct points
        # Ensure they are distinct
        pts = []
        seen = set()
        for p in all_pts:
            if p not in seen:
                pts.append(p)
                seen.add(p)
                if len(pts) == 30:
                    break
        inits.append(pts)

    best_score = 0
    best_points = None
    
    for init in inits:
        if len(set(init)) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in init):
            continue
            
        points = init[:]
        counts = compute_all_areas(points)
        curr_score = max(counts.values()) if counts else 0
        
        if curr_score > best_score:
            best_score = curr_score
            best_points = points[:]
            
    if best_points is None:
        return make_grid(6, 5, -2, -2)
        
    # Local search
    # We will run multiple rounds of local search from the best point found so far.
    # Or just one long search.
    
    points = best_points[:]
    counts = compute_all_areas(points)
    curr_score = max(counts.values())
    
    # Precompute list of all points for random selection
    # We need to be able to pick a new point that is not in the current set.
    # We can just pick a random point from the grid and check if it's in the set.
    # To make it faster, we can maintain a set of current points.
    
    current_set = set(points)
    
    def try_move(idx, new_pt):
        nonlocal counts, curr_score, current_set
        
        old_pt = points[idx]
        
        # Remove old point contributions
        # Triples containing idx: (idx, j, k) for j < k, j != idx, k != idx
        # We need to iterate over all pairs (j, k) from the other 29 points.
        
        others = [i for i in range(30) if i != idx]
        
        # Subtract old
        for a_idx in range(len(others)):
            j = others[a_idx]
            x2, y2 = points[j]
            for b_idx in range(a_idx + 1, len(others)):
                k = others[b_idx]
                x3, y3 = points[k]
                
                x1, y1 = old_pt
                d_old = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if d_old > 0:
                    counts[d_old] -= 1
                    if counts[d_old] == 0:
                        del counts[d_old]
                        
        # Add new
        for a_idx in range(len(others)):
            j = others[a_idx]
            x2, y2 = points[j]
            for b_idx in range(a_idx + 1, len(others)):
                k = others[b_idx]
                x3, y3 = points[k]
                
                x1, y1 = new_pt
                d_new = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if d_new > 0:
                    counts[d_new] = counts.get(d_new, 0) + 1
                    
        # Update state
        points[idx] = new_pt
        current_set.remove(old_pt)
        current_set.add(new_pt)
        
        new_score = max(counts.values()) if counts else 0
        return new_score

    # Search loop
    # We will use a simple hill climbing with random restarts or simulated annealing.
    # Given the time limit, let's just do greedy hill climbing with some randomization.
    
    steps = 0
    while time.time() - start_time < deadline:
        idx = random.randint(0, 29)
        old_pt = points[idx]
        
        # Generate a candidate move
        # Instead of picking a random point from the whole grid, 
        # let's try to pick a point that is "close" to the old point or in a promising region.
        # Or just random.
        
        new_pt = None
        # Try a few random candidates
        for _ in range(20):
            cand = random.choice(all_pts)
            if cand in current_set:
                continue
            new_pt = cand
            break
            
        if new_pt is None:
            continue
            
        new_score = try_move(idx, new_pt)
        
        if new_score >= curr_score:
            curr_score = new_score
            if curr_score > best_score:
                best_score = curr_score
                best_points = points[:]
                # Optional: restart search from this new best?
                # Or continue.
        else:
            # Revert
            # We need to revert the move.
            # try_move(idx, old_pt) will revert it.
            # But we must ensure old_pt is not in current_set (it was removed).
            # try_move will remove new_pt and add old_pt.
            try_move(idx, old_pt)
            
        steps += 1
        
        # If we hit a plateau for a long time, we might want to perturb.
        # But let's keep it simple.
        
    return best_points
