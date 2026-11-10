import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 45  # safety margin before 60s kill
    
    def compute_doubled_areas(points):
        """Compute all non-zero doubled areas (absolute determinants) for all triples."""
        n = len(points)
        areas = []
        for i in range(n - 2):
            x1, y1 = points[i]
            for j in range(i + 1, n - 1):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x2*y1 - y2*x3 + y2*x1
                    # Simplified: det = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                    d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if d > 0:
                        areas.append(d)
        return areas
    
    def score_config(points):
        """Return the maximum multiplicity of any single positive doubled area."""
        areas = compute_doubled_areas(points)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())
    
    # Generate candidate configurations
    candidates = []
    
    # 1. Grid-like configurations
    for width in range(3, 10):
        for height in range(3, 10):
            if width * height >= 30:
                # Try different subsets of the grid
                grid = [(x, y) for x in range(width) for y in range(height)]
                if len(grid) >= 30:
                    # Take first 30 or random sample
                    candidates.append(grid[:30])
                    # Also try centered versions
                    ox = -width // 2
                    oy = -height // 2
                    candidates.append([(x+ox, y+oy) for x, y in grid[:30]])
    
    # 2. Concentric polygon-like configurations
    for scale in [1, 2, 3]:
        # Square rings
        cand = []
        for ring in range(1, min(5, 31//8 + 1)):
            s = ring * scale
            for x in range(-s, s+1):
                for y in range(-s, s+1):
                    if abs(x) == s or abs(y) == s:
                        cand.append((x, y))
            if len(cand) >= 30:
                candidates.append(cand[:30])
                break
        # Add more if needed
        while len(cand) < 30:
            for x in range(-30, 31):
                for y in range(-30, 31):
                    if (x, y) not in cand:
                        cand.append((x, y))
                    if len(cand) >= 30:
                        break
                if len(cand) >= 30:
                    break
        candidates.append(cand[:30])
    
    # 3. Arithmetic progression patterns
    for base in range(-20, 21, 5):
        for step in [2, 3, 4, 5]:
            cand = []
            for i in range(30):
                x = base + (i % 10) * step
                y = (i // 10) * step
                if abs(x) <= 30 and abs(y) <= 30:
                    cand.append((x, y))
            if len(cand) >= 30:
                candidates.append(cand[:30])
    
    # 4. Diagonal line patterns
    for offset in range(-20, 21, 5):
        cand = []
        for i in range(30):
            x = offset + i
            y = (i * 2) % 15 - 7
            if abs(x) <= 30 and abs(y) <= 30:
                cand.append((x, y))
        if len(cand) >= 30:
            candidates.append(cand[:30])
    
    # 5. Random configurations
    for _ in range(50):
        random_points = set()
        while len(random_points) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            random_points.add((x, y))
        candidates.append(list(random_points))
    
    # 6. Structured: multiple parallel lines with offsets
    for num_lines in [3, 4, 5, 6]:
        for spacing in [3, 4, 5, 6, 8]:
            cand = []
            for line_idx in range(num_lines):
                y_base = line_idx * spacing - (num_lines - 1) * spacing // 2
                for i in range(30 // num_lines + 1):
                    x = i * 2 - 10
                    if abs(x) <= 30 and abs(y_base) <= 30:
                        cand.append((x, y_base))
            if len(cand) >= 30:
                candidates.append(cand[:30])
    
    # Evaluate all candidates
    best_score = 0
    best_points = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        if any(abs(x) > 30 or abs(y) > 30 for x, y in cand):
            continue
        s = score_config(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
    
    if best_points is None:
        # Fallback to seed
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = score_config(best_points)
    
    # Local search to improve
    current_points = list(best_points)
    current_score = best_score
    global_best_points = list(best_points)
    global_best_score = best_score
    
    # Precompute all valid positions
    valid_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    random.shuffle(valid_positions)
    
    # Local search iterations
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 29)
        old_point = current_points[idx]
        
        # Try a random new position
        new_point = valid_positions[random.randint(0, len(valid_positions)-1)]
        
        if new_point == old_point or new_point in set(current_points):
            continue
        
        current_points[idx] = new_point
        new_score = score_config(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > global_best_score:
                global_best_score = new_score
                global_best_points = list(current_points)
        else:
            current_points[idx] = old_point
    
    return global_best_points
