import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
    
    n = 30
    triples = list(itertools.combinations(range(n), 3))
    triple_count = len(triples)
    
    # Precompute for each point i, the list of (j, k) pairs
    pairs_for_point = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(n):
                if i == k or j == k: continue
                # Avoid duplicates in pairs_for_point? 
                # If we iterate j,k such that j<k, we get unique pairs.
                if j < k:
                    pairs_for_point[i].append((j, k))

    # Initial configuration: 6x5 grid
    points = [(x, y) for x in range(6) for y in range(5)]
    
    def init_counts(pts):
        counts = Counter()
        for i in range(n):
            for j, k in pairs_for_point[i]:
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    counts[det] += 1
        return counts

    global_counts = init_counts(points)
    current_max_freq = max(global_counts.values()) if global_counts else 0
    best_points = points[:]
    best_max_freq = current_max_freq
    
    # Local Search with restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        if restart == 0:
            pts = [(x, y) for x in range(6) for y in range(5)]
        else:
            pts = best_points[:]
            # Perturb
            for _ in range(3):
                idx = random.randint(0, n-1)
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if (nx, ny) not in pts:
                    pts[idx] = (nx, ny)
                    
        global_counts = init_counts(pts)
        current_max_freq = max(global_counts.values()) if global_counts else 0
        
        if current_max_freq > best_max_freq:
            best_max_freq = current_max_freq
            best_points = pts[:]
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Try moves for point i
                # We can restrict search space to nearby points or random
                # Random is better for escape
                for _ in range(20):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pos = (nx, ny)
                    
                    if new_pos in pts:
                        continue
                    
                    old_pos = pts[i]
                    
                    # Calculate old areas for triples containing i
                    old_areas = []
                    for j, k in pairs_for_point[i]:
                        x1, y1 = old_pos
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if det > 0:
                            old_areas.append(det)
                            global_counts[det] -= 1
                            if global_counts[det] == 0:
                                del global_counts[det]
                    
                    pts[i] = new_pos
                    
                    # Calculate new areas
                    new_areas = []
                    for j, k in pairs_for_point[i]:
                        x1, y1 = new_pos
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if det > 0:
                            new_areas.append(det)
                            global_counts[det] += 1
                            
                    new_max_freq = max(global_counts.values()) if global_counts else 0
                    if new_max_freq > current_max_freq:
                        current_max_freq = new_max_freq
                        improved = True
                        if current_max_freq > best_max_freq:
                            best_max_freq = current_max_freq
                            best_points = pts[:]
                        break
                    else:
                        # Revert
                        pts[i] = old_pos
                        # Re-add old areas
                        for det in old_areas:
                            global_counts[det] += 1
                        # Remove new areas
                        for det in new_areas:
                            global_counts[det] -= 1
                            if global_counts[det] == 0:
                                del global_counts[det]
                        
        if time.time() > deadline:
            break

    return best_points
