import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0
    
    # 1. Scoring Function
    def score(pts):
        counts = Counter()
        n = len(pts)
        for i in range(n - 2):
            xi, yi = pts[i]
            for j in range(i + 1, n - 1):
                xj, yj = pts[j]
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        counts[d] += 1
        return max(counts.values()) if counts else 0

    # 2. Candidate Generation
    def make_grid(w, h, dx=1, dy=1, ox=0, oy=0):
        pts = []
        for i in range(w):
            for j in range(h):
                x = ox + i * dx
                y = oy + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts if len(pts) == 30 else None

    candidates = []
    
    # Grids
    factors = [1, 2, 3, 5, 6, 10, 15, 30]
    for w in factors:
        h = 30 // w
        for dx in [1, 2]:
            for dy in [1, 2]:
                # Center the grid
                ox = -((w - 1) * dx) // 2
                oy = -((h - 1) * dy) // 2
                g = make_grid(w, h, dx, dy, ox, oy)
                if g:
                    candidates.append(g)
                    
    # Random
    random.seed(42)
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))
        
    # Hexagonal-ish
    hex_pts = []
    # Generate a hex grid pattern and pick 30
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i
            y = j + (i % 2) # simple skew
            if -30 <= x <= 30 and -30 <= y <= 30:
                hex_pts.append((x, y))
    if len(hex_pts) >= 30:
        random.shuffle(hex_pts)
        candidates.append(hex_pts[:30])

    # 3. Evaluate Initials
    scored = [(score(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_pts = scored[0]
    
    # 4. Local Search
    def local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < end_time:
            if no_improve_count > max_no_improve:
                # Restart from best or random perturbation
                # Simple restart: randomize 5 points
                for _ in range(5):
                    idx = random.randint(0, 29)
                    old = current_pts[idx]
                    current_set.remove(old)
                    new = (random.randint(-30, 30), random.randint(-30, 30))
                    while new in current_set:
                        new = (random.randint(-30, 30), random.randint(-30, 30))
                    current_pts[idx] = new
                    current_set.add(new)
                current_score = score(current_pts)
                no_improve_count = 0
                continue

            move_type = random.random()
            if move_type < 0.7:
                # Single move
                i = random.randint(0, 29)
                old_p = current_pts[i]
                if random.random() < 0.8:
                    nx = old_p[0] + random.randint(-2, 2)
                    ny = old_p[1] + random.randint(-2, 2)
                else:
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                new_p = (nx, ny)
                if -30 <= nx <= 30 and -30 <= ny <= 30 and new_p not in current_set:
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    new_score = score(current_pts)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        current_pts[i] = old_p
                        current_set.remove(new_p)
                        current_set.add(old_p)
                        no_improve_count += 1
            else:
                # Swap
                i = random.randint(0, 29)
                j = random.randint(0, 29)
                if i == j: continue
                pi = current_pts[i]
                pj = current_pts[j]
                
                # Generate new positions
                if random.random() < 0.8:
                    nxi = pi[0] + random.randint(-2, 2)
                    nyi = pi[1] + random.randint(-2, 2)
                    nxj = pj[0] + random.randint(-2, 2)
                    nyj = pj[1] + random.randint(-2, 2)
                else:
                    nxi, nyi = random.randint(-30, 30), random.randint(-30, 30)
                    nxj, nyj = random.randint(-30, 30), random.randint(-30, 30)
                
                np_i = (nxi, nyi)
                np_j = (nxj, nyj)
                
                if (-30 <= nxi <= 30 and -30 <= nyi <= 30 and 
                    -30 <= nxj <= 30 and -30 <= nyj <= 30 and
                    np_i not in current_set and np_j not in current_set and
                    np_i != np_j):
                    
                    current_pts[i] = np_i
                    current_pts[j] = np_j
                    current_set.remove(pi)
                    current_set.remove(pj)
                    current_set.add(np_i)
                    current_set.add(np_j)
                    
                    new_score = score(current_pts)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        current_pts[i] = pi
                        current_pts[j] = pj
                        current_set.remove(np_i)
                        current_set.remove(np_j)
                        current_set.add(pi)
                        current_set.add(pj)
                        no_improve_count += 1
                        
        return current_score, current_pts

    # Run searches
    # Allocate time to top 3 initial candidates
    search_times = [20, 15, 10]
    for idx in range(min(3, len(scored))):
        if time.time() > total_deadline - 5: break
        rem = total_deadline - time.time() - 2
        if rem <= 0: break
        t = min(search_times[idx], rem)
        s, p = local_search(scored[idx][1], t)
        if s > best_score:
            best_score = s
            best_pts = p

    return best_pts
