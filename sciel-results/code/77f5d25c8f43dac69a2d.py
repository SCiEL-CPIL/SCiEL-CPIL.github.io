import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safe margin for 60s limit
    
    def full_evaluate(points):
        """Calculate full area counts and max multiplicity."""
        n = len(points)
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    d = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0, Counter()
        c = Counter(areas)
        return max(c.values()), c

    # Initial configuration: grid
    def make_grid(w, h, ox, oy):
        pts = []
        for i in range(w):
            for j in range(h):
                x = ox + i * 2
                y = oy + j * 2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts[:30]
    
    candidates = []
    pts = make_grid(6, 5, -10, -8)
    if len(pts) == 30:
        candidates.append(pts)
    
    pts = make_grid(5, 6, -8, -10)
    if len(pts) == 30:
        candidates.append(pts)
    
    random.seed(42)
    pts = random.sample([(x,y) for x in range(-30, 31) for y in range(-30, 31)], 30)
    candidates.append(pts)
    
    best_pts = None
    best_score = -1
    best_counter = None
    for pts in candidates:
        s, c = full_evaluate(pts)
        if s > best_score:
            best_score = s
            best_pts = pts[:]
            best_counter = c
    
    current = best_pts[:]
    current_score = best_score
    current_counter = best_counter
    best = best_pts[:]
    best_found = best_score
    occupied = set(current)
    
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    temperature = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        idx = random.randint(0, 29)
        old_pos = current[idx]
        
        new_pos = random.choice(all_positions)
        if new_pos == old_pos or new_pos in occupied:
            continue
        
        others = [i for i in range(30) if i != idx]
        
        # 1. Remove old contributions
        for a_idx in range(len(others)):
            j = others[a_idx]
            for b_idx in range(a_idx + 1, len(others)):
                k = others[b_idx]
                ax, ay = old_pos
                bx, by = current[j]
                cx, cy = current[k]
                d = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                if d > 0:
                    current_counter[d] -= 1
                    if current_counter[d] == 0:
                        del current_counter[d]
        
        # 2. Update state
        current[idx] = new_pos
        occupied.remove(old_pos)
        occupied.add(new_pos)
        
        # 3. Add new contributions
        for a_idx in range(len(others)):
            j = others[a_idx]
            for b_idx in range(a_idx + 1, len(others)):
                k = others[b_idx]
                ax, ay = new_pos
                bx, by = current[j]
                cx, cy = current[k]
                d = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                if d > 0:
                    current_counter[d] = current_counter.get(d, 0) + 1
        
        new_score = max(current_counter.values()) if current_counter else 0
        prev_score = current_score
        
        # 4. Accept or Reject
        if new_score > prev_score:
            current_score = new_score
            if new_score > best_found:
                best_found = new_score
                best = current[:]
        else:
            delta = prev_score - new_score
            # Accept with probability exp(-delta/T)
            if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                current_score = new_score
            else:
                # Revert state
                current[idx] = old_pos
                occupied.remove(new_pos)
                occupied.add(old_pos)
                
                # Revert counter: Remove new, Add old
                for a_idx in range(len(others)):
                    j = others[a_idx]
                    for b_idx in range(a_idx + 1, len(others)):
                        k = others[b_idx]
                        ax, ay = new_pos
                        bx, by = current[j]
                        cx, cy = current[k]
                        d = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                        if d > 0:
                            current_counter[d] -= 1
                            if current_counter[d] == 0:
                                del current_counter[d]
                
                for a_idx in range(len(others)):
                    j = others[a_idx]
                    for b_idx in range(a_idx + 1, len(others)):
                        k = others[b_idx]
                        ax, ay = old_pos
                        bx, by = current[j]
                        cx, cy = current[k]
                        d = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                        if d > 0:
                            current_counter[d] = current_counter.get(d, 0) + 1
                            
                current_score = prev_score
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
            
    return best
