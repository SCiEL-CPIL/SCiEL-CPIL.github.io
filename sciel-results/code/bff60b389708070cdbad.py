import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # 1. Generate initial points
    # A grid is a good start. 6x5 grid centered.
    def make_grid(w, h, ox, oy):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x + ox, y + oy))
        return pts
        
    # Try a few initial grids
    inits = [
        make_grid(6, 5, -2, -2),
        make_grid(5, 6, -2, -2),
        make_grid(10, 3, -4, -1),
        make_grid(3, 10, -1, -4),
    ]
    
    best_score = -1
    best_points = None
    
    # Precompute all valid points for random selection
    all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    for init in inits:
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in init):
            continue
        if len(set(init)) != 30:
            continue
            
        points = init[:]
        # Initialize counts
        counts = {}
        for i in range(30):
            x1, y1 = points[i]
            for j in range(i+1, 30):
                x2, y2 = points[j]
                for k in range(j+1, 30):
                    x3, y3 = points[k]
                    d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if d > 0:
                        counts[d] = counts.get(d, 0) + 1
        
        curr_score = max(counts.values()) if counts else 0
        
        if curr_score > best_score:
            best_score = curr_score
            best_points = points[:]
            
        # Local Search from this start
        # We will run local search for a fixed number of steps or time
        # Let's run a time-sliced search
        
        # To avoid re-initializing counts for every restart in the main loop, 
        # we can just run search from the best found so far.
        
    # Run search from best_points
    if best_points is None:
        return make_grid(6, 5, -2, -2)
        
    points = best_points[:]
    counts = {}
    for i in range(30):
        x1, y1 = points[i]
        for j in range(i+1, 30):
            x2, y2 = points[j]
            for k in range(j+1, 30):
                x3, y3 = points[k]
                d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if d > 0:
                    counts[d] = counts.get(d, 0) + 1
    
    curr_score = max(counts.values())
    
    # Helper to update counts and return new score
    def get_new_score(points, idx, new_pt):
        # We need to revert the old point's contributions and add new point's
        # Old point: points[idx]
        # New point: new_pt
        
        # We must be careful: points[idx] is still the old one in the list passed?
        # Let's assume points[idx] is OLD. We calculate delta.
        
        old_pt = points[idx]
        
        # List of other points indices
        others = [i for i in range(30) if i != idx]
        
        # We need to subtract old contributions and add new ones
        # To avoid modifying the global counts dict in a way that's hard to revert if we reject,
        # we can just calculate the "delta" in a local dict or just recompute the max?
        # No, recomputing max from scratch is O(M).
        # But we need to update the counts dict to keep it consistent for the NEXT move.
        
        # Strategy:
        # 1. Create a copy of counts? No, too slow.
        # 2. Modify counts, get score, if reject, revert counts.
        
        # Let's modify counts in place.
        
        # Subtract old
        for j_idx in range(len(others)):
            j = others[j_idx]
            for k_idx in range(j_idx + 1, len(others)):
                k = others[k_idx]
                
                x1, y1 = old_pt
                x2, y2 = points[j]
                x3, y3 = points[k]
                d_old = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if d_old > 0:
                    counts[d_old] -= 1
                    if counts[d_old] == 0:
                        del counts[d_old]
                        
        # Add new
        for j_idx in range(len(others)):
            j = others[j_idx]
            for k_idx in range(j_idx + 1, len(others)):
                k = others[k_idx]
                
                x1, y1 = new_pt
                x2, y2 = points[j]
                x3, y3 = points[k]
                d_new = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if d_new > 0:
                    counts[d_new] = counts.get(d_new, 0) + 1
        
        # Update points
        points[idx] = new_pt
        
        # Calculate new score
        if counts:
            return max(counts.values())
        return 0

    # Search Loop
    while time.time() - start_time < deadline:
        idx = random.randint(0, 29)
        old_pt = points[idx]
        
        # Pick a random new point
        # Ensure it's not the same as any other point
        # Simple rejection
        new_pt = None
        for _ in range(10):
            cand = random.choice(all_pts)
            if cand == old_pt:
                continue
            if cand in points: # O(N) check
                continue
            new_pt = cand
            break
            
        if new_pt is None:
            continue
            
        # Try the move
        # We need to revert points[idx] to old_pt before calling get_new_score?
        # get_new_score expects points[idx] to be old_pt.
        # It updates points[idx] to new_pt internally.
        
        new_score = get_new_score(points, idx, new_pt)
        
        if new_score > curr_score:
            curr_score = new_score
            if curr_score > best_score:
                best_score = curr_score
                best_points = points[:]
        elif new_score == curr_score:
            if random.random() < 0.5:
                # Accept, keep new state
                curr_score = new_score
            else:
                # Revert
                # We need to revert counts and points
                # This is the expensive part of rejection.
                # We can just call get_new_score again with old_pt?
                # get_new_score(points, idx, old_pt) where points[idx] is currently new_pt.
                # It will subtract new_pt contributions and add old_pt contributions.
                # And set points[idx] = old_pt.
                get_new_score(points, idx, old_pt)
        else:
            # Revert
            get_new_score(points, idx, old_pt)
            
    return best_points
