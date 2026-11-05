import time
import random
import math
from collections import Counter

def construct():
    deadline = time.time() + 82.0  # safety margin for 90s limit
    random.seed(42)
    
    N = 30
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def full_count(points):
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count
    
    def get_pairs_excluding(idx):
        pairs = []
        for j in range(N):
            if j == idx:
                continue
            for k in range(j+1, N):
                if k == idx:
                    continue
                pairs.append((j, k))
        return pairs
    
    all_pairs = [get_pairs_excluding(i) for i in range(N)]
    
    def count_target(points, target):
        """Count how many triples have doubled area == target."""
        count = 0
        for i in range(N):
            for j, k in all_pairs[i]:
                if doubled_area(points[i], points[j], points[k]) == target:
                    count += 1
        return count

    def generate_arithmetic_progression_config():
        """Generate points along an arithmetic progression in 2D."""
        points = []
        # Use a 2D arithmetic progression: p_i = start + i * step
        # But we need 30 distinct points. Let's use a grid-like structure.
        # Try a 6x5 grid shifted to center
        points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
        return points
    
    def generate_hexagonal_config():
        """Generate a hexagonal lattice configuration."""
        points = []
        # Hexagonal lattice points within a radius
        for q in range(-3, 4):
            for r in range(-3, 4):
                # Convert axial to cartesian for hex lattice
                x = q + r/2
                y = r * math.sqrt(3)/2
                # Only include integer coordinates
                if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
                    xi, yi = int(round(x)), int(round(y))
                    if -30 <= xi <= 30 and -30 <= yi <= 30:
                        points.append((xi, yi))
        # Deduplicate and take first 30
        seen = set()
        unique = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        while len(unique) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:30]
    
    def generate_double_grid(shift_x, shift_y):
        """Generate 30 points as union of two 15-point grids."""
        grid_A = [(x, y) for x in range(-2, 3) for y in range(-1, 2)]
        grid_B = [(x + shift_x, y + shift_y) for x, y in grid_A]
        
        points = grid_A + grid_B
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                unique_points.append(p)
        
        while len(unique_points) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        
        return unique_points[:30]

    # Try multiple structured initial configurations
    initial_configs = [
        generate_arithmetic_progression_config(),
        generate_hexagonal_config(),
        generate_double_grid(3, 0),
        generate_double_grid(0, 3),
        generate_double_grid(2, 2),
        generate_double_grid(4, 1),
        generate_double_grid(5, 0),
        generate_double_grid(0, 5),
        # Diamond configuration
        [(x, y) for x in range(-3, 4) for y in range(-3, 4) if abs(x) + abs(y) <= 4][:30],
    ]
    
    # Filter to valid distinct points in [-30,30]^2
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
    
    # Simulated Annealing with target-focused optimization
    T = 15.0
    cooling_rate = 0.995
    
    no_improve = 0
    max_no_improve = 300
    restarts = 0
    max_restarts = 15
    
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
            
            # Switch to the most common area in current configuration
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 100
            T = 15.0  # Reset temperature
            
            # Random restart with perturbation
            for _ in range(5):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    current_count = count_target(coords, target)
                    break
            
            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        pairs = all_pairs[idx]
        
        # Try multiple candidate positions - focus on nearby points first
        candidates = []
        # Nearby candidates (more likely to maintain structure)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.append(new_pt)
        # Add some random candidates
        for _ in range(10):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.append(new_pt)
        
        best_delta = -999
        best_new_pt = None
        
        # Precompute old matches for this idx
        old_matches = 0
        for j, k in pairs:
            p_j = coords[j]
            p_k = coords[k]
            if doubled_area(old_pt, p_j, p_k) == target:
                old_matches += 1
        
        for new_pt in candidates:
            new_matches = 0
            for j, k in pairs:
                p_j = coords[j]
                p_k = coords[k]
                if doubled_area(new_pt, p_j, p_k) == target:
                    new_matches += 1
            
            delta = new_matches - old_matches
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt
        
        if best_new_pt is not None:
            # Simulated Annealing acceptance
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
                current_count += best_delta
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
            else:
                no_improve += 1
        
        T *= cooling_rate
    
    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
