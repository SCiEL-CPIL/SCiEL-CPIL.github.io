import time
import random
import math
from collections import Counter

def construct():
    deadline = time.time() + 45.0  # safety margin for 90s limit
    random.seed(42)
    
    N = 30
    
    # Precompute all triplets (i, j, k) with i < j < k
    triplets = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triplets.append((i, j, k))
    
    # Precompute which triplets involve each index for fast delta computation
    triplets_by_idx = [[] for _ in range(N)]
    for idx_t, (i, j, k) in enumerate(triplets):
        triplets_by_idx[i].append(idx_t)
        triplets_by_idx[j].append(idx_t)
        triplets_by_idx[k].append(idx_t)
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def full_count(points):
        areas = []
        for i, j, k in triplets:
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count
    
    def count_target(points, target):
        """Count how many triples have doubled area == target."""
        count = 0
        for i, j, k in triplets:
            if doubled_area(points[i], points[j], points[k]) == target:
                count += 1
        return count
    
    def smooth_objective(points, target, beta=0.5):
        """
        Max-SAT-inspired smooth surrogate:
        For each triple, compute a soft reward based on how close its area is to target.
        Uses log-sum-exp style smooth max to encourage concentration on target area.
        """
        total = 0.0
        for i, j, k in triplets:
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                # Soft clause: reward = exp(-beta * |a - target| / target) if target > 0
                # This creates a smooth peak at the target area
                if target > 0:
                    diff = abs(a - target)
                    reward = math.exp(-beta * diff / max(target, 1))
                else:
                    reward = 0.0
                total += reward
        return total
    
    # Generate initial configurations
    def generate_grid_config():
        return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    
    initial_configs = [
        generate_grid_config(),
        [(x+1, y) for x in range(-2, 4) for y in range(-2, 3)],
        [(x, y) for x in range(-3, 4) for y in range(-3, 4) if abs(x) + abs(y) <= 4][:30],
    ]
    
    best_initial = None
    best_initial_count = 0
    
    for config in initial_configs:
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]
        
        counter, ba, bc = full_count(filtered)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered
    
    coords = list(best_initial)
    counter, target, current_count = full_count(coords)
    best_overall = current_count
    best_points = list(coords)
    best_target = target
    
    # Max-SAT-inspired local search with smooth surrogate
    # Phase 1: Use smooth objective to guide search toward high-density area regions
    # Phase 2: Polish with exact target count
    
    T = 5.0
    cooling_rate = 0.995
    beta = 0.3  # Smoothness parameter for soft clauses
    
    no_improve = 0
    max_no_improve = 80
    restarts = 0
    max_restarts = 15
    phase = 1  # 1 = smooth surrogate, 2 = exact polish
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            
            # Re-evaluate current state
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            
            # Switch to the most common area
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 30
            T = 5.0
            
            # Transition to polish phase after some restarts
            if phase == 1 and restarts >= 5:
                phase = 2
                T = 2.0
                beta = 0.0  # Disable smooth surrogate, use exact count
            
            # Perturb current solution
            for _ in range(3):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    break
            
            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        
        # Generate candidates
        candidates = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.append(new_pt)
        for _ in range(8):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.append(new_pt)
        
        if not candidates:
            no_improve += 1
            T *= cooling_rate
            continue
        
        best_delta = -999
        best_new_pt = None
        
        # Precompute old contribution for this idx
        if phase == 1 and beta > 0:
            # Smooth surrogate evaluation
            old_score = 0.0
            for t_idx in triplets_by_idx[idx]:
                i, j, k = triplets[t_idx]
                a = doubled_area(coords[i], coords[j], coords[k])
                if a > 0:
                    diff = abs(a - target)
                    old_score += math.exp(-beta * diff / max(target, 1))
            
            for new_pt in candidates:
                new_score = 0.0
                for t_idx in triplets_by_idx[idx]:
                    i, j, k = triplets[t_idx]
                    if i == idx:
                        p1, p2, p3 = new_pt, coords[j], coords[k]
                    elif j == idx:
                        p1, p2, p3 = coords[i], new_pt, coords[k]
                    else:
                        p1, p2, p3 = coords[i], coords[j], new_pt
                    a = doubled_area(p1, p2, p3)
                    if a > 0:
                        diff = abs(a - target)
                        new_score += math.exp(-beta * diff / max(target, 1))
                
                delta = new_score - old_score
                if delta > best_delta:
                    best_delta = delta
                    best_new_pt = new_pt
        else:
            # Exact count evaluation
            old_matches = 0
            for t_idx in triplets_by_idx[idx]:
                i, j, k = triplets[t_idx]
                if doubled_area(coords[i], coords[j], coords[k]) == target:
                    old_matches += 1
            
            for new_pt in candidates:
                new_matches = 0
                for t_idx in triplets_by_idx[idx]:
                    i, j, k = triplets[t_idx]
                    if i == idx:
                        p1, p2, p3 = new_pt, coords[j], coords[k]
                    elif j == idx:
                        p1, p2, p3 = coords[i], new_pt, coords[k]
                    else:
                        p1, p2, p3 = coords[i], coords[j], new_pt
                    if doubled_area(p1, p2, p3) == target:
                        new_matches += 1
                
                delta = new_matches - old_matches
                if delta > best_delta:
                    best_delta = delta
                    best_new_pt = new_pt
        
        if best_new_pt is not None:
            if best_delta > 0:
                accept = True
            else:
                if T > 0.1:
                    accept_prob = math.exp(best_delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False
            
            if accept:
                coords[idx] = best_new_pt
                no_improve = 0
                # Update best if improved
                c, ba, bc = full_count(coords)
                if bc > best_overall:
                    best_overall = bc
                    best_points = list(coords)
                    best_target = ba
                    target = ba
            else:
                no_improve += 1
        
        T *= cooling_rate
    
    # Final high-precision polish pass
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        for _ in range(50):
            if time.time() >= polish_deadline:
                break
            idx = random.randint(0, N-1)
            old_pt = coords[idx]
            best_delta = 0
            best_new_pt = None
            
            old_matches = 0
            for t_idx in triplets_by_idx[idx]:
                i, j, k = triplets[t_idx]
                if doubled_area(coords[i], coords[j], coords[k]) == target:
                    old_matches += 1
            
            # Try all nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if new_pt == old_pt or new_pt in coords:
                        continue
                    if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                        continue
                    
                    new_matches = 0
                    for t_idx in triplets_by_idx[idx]:
                        i, j, k = triplets[t_idx]
                        if i == idx:
                            p1, p2, p3 = new_pt, coords[j], coords[k]
                        elif j == idx:
                            p1, p2, p3 = coords[i], new_pt, coords[k]
                        else:
                            p1, p2, p3 = coords[i], coords[j], new_pt
                        if doubled_area(p1, p2, p3) == target:
                            new_matches += 1
                    
                    delta = new_matches - old_matches
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pt = new_pt
            
            if best_new_pt is not None and best_delta > 0:
                coords[idx] = best_new_pt
                c, ba, bc = full_count(coords)
                if bc > best_overall:
                    best_overall = bc
                    best_points = list(coords)
                    best_target = ba
                    target = ba
    
    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
