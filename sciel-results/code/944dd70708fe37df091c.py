import time
import random
from collections import defaultdict
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def line_hash(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:
            return ('v', x1)
        if y1 == y2:
            return ('h', y1)
        
        dy = y2 - y1
        dx = x2 - x1
        
        g = math.gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        elif dx_n == 0 and dy_n < 0:
            dy_n = -dy_n
            
        a = dy_n
        b = -dx_n
        c = dx_n * y1 - dy_n * x1
        
        g = math.gcd(abs(a), abs(b), abs(c))
        if g > 0:
            a //= g
            b //= g
            c //= g
            
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
            
        return (a, b, c)

    def evaluate(points):
        n = len(points)
        if n != 16:
            return -1
        if len(set(map(tuple, points))) != 16:
            return -1
            
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                key = line_hash(points[i], points[j])
                line_counts[key] += 1
                
        score = 0
        for key, count in line_counts.items():
            disc = 1 + 8 * count
            sqrt_disc = int(math.isqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                return -1
            k = (1 + sqrt_disc) // 2
            if k >= 5:
                return -1
            if k == 4:
                score += 1
        return score

    def get_line_info(points):
        """Get detailed line information for a configuration."""
        line_points = defaultdict(list)
        for i in range(16):
            for j in range(i + 1, 16):
                key = line_hash(points[i], points[j])
                line_points[key].append(i)
        
        # Convert to proper point lists
        result = {}
        for key, indices in line_points.items():
            # Deduplicate indices
            unique_indices = list(set(indices))
            if len(unique_indices) >= 2:
                result[key] = unique_indices
        
        return result

    def simulated_annealing(initial_points, deadline):
        """Simulated annealing optimization."""
        current_points = [p[:] for p in initial_points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_points = [p[:] for p in current_points]
        best_score = current_score
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        cooling_rate = 0.9995
        
        T = T_start
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline:
            if no_improve > max_no_improve:
                break
            
            # Pick a point to move
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            
            # Generate new position - mix of local and global moves
            if random.random() < 0.15:
                # Large jump
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
            elif random.random() < 0.5:
                # Medium move
                step = random.randint(3, 10)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            else:
                # Small move
                step = random.randint(1, 3)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            
            # Keep within bounds
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
                
            new_pt = [new_x, new_y]
            
            # Check distinctness
            distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p == new_pt:
                    distinct = False
                    break
            if not distinct:
                continue
                
            # Try the move
            current_points[idx] = new_pt
            new_score = evaluate(current_points)
            
            # Accept if better or with probability based on temperature
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Simulated annealing acceptance
                delta = new_score - current_score
                if delta < 0 and random.random() < math.exp(delta / T):
                    current_score = new_score
                    no_improve += 1
                else:
                    # Revert
                    current_points[idx] = old_pt
                    no_improve += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
        
        return best_points, best_score

    # Initial candidates
    candidates = []
    
    # 1. 4x4 Grid
    grid = [[i, j] for i in range(4) for j in range(4)]
    candidates.append(grid)
    
    # 2. 4x4 Grid with perturbation
    perturbed_grid = [[i + random.randint(-2, 2), j + random.randint(-2, 2)] for i in range(4) for j in range(4)]
    # Ensure distinctness
    unique_pts = []
    for p in perturbed_grid:
        if p not in unique_pts:
            unique_pts.append(p)
    while len(unique_pts) < 16:
        x = random.randint(-5, 5)
        y = random.randint(-5, 5)
        if [x, y] not in unique_pts:
            unique_pts.append([x, y])
    candidates.append(unique_pts[:16])
    
    # 3. Random configurations with structured bias
    random.seed(42)
    for _ in range(15):
        pts = []
        # Start with some structure
        base_x = random.randint(-10, 10)
        base_y = random.randint(-10, 10)
        for i in range(16):
            # Mix of local and global placement
            if random.random() < 0.3:
                x = base_x + random.randint(-5, 5)
                y = base_y + random.randint(-5, 5)
            else:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
            if [x, y] not in pts:
                pts.append([x, y])
        candidates.append(pts)
    
    # 4. Configuration based on finite field geometry (simplified)
    # Use points from a 4x4 grid but with some offsets to break 5-collinear
    ff_points = []
    for i in range(4):
        for j in range(4):
            # Add small offsets based on position
            offset = (i * j) % 3
            ff_points.append([i * 3 + offset, j * 3 + (offset + 1) % 2])
    candidates.append(ff_points)
    
    best_score = -1
    best_points = grid
    
    # Run simulated annealing on each candidate
    for cand in candidates:
        if time.time() > deadline - 10:  # Keep 10s buffer
            break
            
        # Ensure candidate is valid and distinct
        if len(set(map(tuple, cand))) != 16:
            continue
        if any(not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000) for p in cand):
            continue
            
        res_pts, res_score = simulated_annealing(cand, deadline - 10)
        
        if res_score > best_score:
            best_score = res_score
            best_points = [p[:] for p in res_pts]
            
        # If we found a high score, we might be done
        if best_score >= 15:
            break
    
    # If we still have time, do a final intensive search
    if time.time() < deadline - 5 and best_score < 12:
        # Use the best configuration as starting point for more intensive search
        intensive_deadline = deadline - 2
        res_pts, res_score = simulated_annealing(best_points, intensive_deadline)
        if res_score > best_score:
            best_score = res_score
            best_points = [p[:] for p in res_pts]
    
    return best_points
