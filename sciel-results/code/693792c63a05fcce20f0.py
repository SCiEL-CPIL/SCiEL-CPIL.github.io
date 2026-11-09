import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def evaluate(points):
        """Return the max multiplicity of a single positive doubled area."""
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    # Doubled area = |det(B-A, C-A)|
                    d = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        return max(Counter(areas).values())
    
    # Initial configuration: try a structured grid
    # Use a 6x5 grid centered in the range
    def make_grid(w, h, ox, oy):
        pts = []
        for i in range(w):
            for j in range(h):
                x = ox + i * 2
                y = oy + j * 2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts[:30]
    
    # Try several initial configurations
    candidates = []
    
    # Grid 6x5
    pts = make_grid(6, 5, -10, -8)
    if len(pts) == 30:
        candidates.append(pts)
    
    # Grid 5x6
    pts = make_grid(5, 6, -8, -10)
    if len(pts) == 30:
        candidates.append(pts)
    
    # Random initial
    random.seed(42)
    pts = random.sample([(x,y) for x in range(-30, 31) for y in range(-30, 31)], 30)
    candidates.append(pts)
    
    # Pick best initial
    best_pts = None
    best_score = -1
    for pts in candidates:
        s = evaluate(pts)
        if s > best_score:
            best_score = s
            best_pts = pts[:]
    
    # Local search with simulated annealing
    current = best_pts[:]
    current_score = best_score
    best = best_pts[:]
    best_found = best_score
    
    # All valid positions
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    temperature = 2.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_pos = current[idx]
        
        # Try moving to a random new position
        new_pos = random.choice(all_positions)
        if new_pos == old_pos:
            continue
        
        # Check if new_pos is already occupied
        if any(current[i] == new_pos for i in range(30) if i != idx):
            continue
        
        current[idx] = new_pos
        new_score = evaluate(current)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_found:
                best_found = new_score
                best = current[:]
        else:
            # Accept with probability exp(-(delta)/T)
            delta = current_score - new_score
            if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                current_score = new_score
            else:
                current[idx] = old_pos
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best
