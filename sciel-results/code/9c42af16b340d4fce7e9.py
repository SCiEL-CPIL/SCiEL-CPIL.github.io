import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def score(pts):
        # Vectorized scoring using NumPy
        # pts: list of (x, y) or np.array of shape (n, 2)
        if isinstance(pts, list):
            pts = np.array(pts)
        n = len(pts)
        if n < 3:
            return 0
        
        # Compute doubled areas for all triples (i, j, k) with i < j < k
        # Area = 0.5 * |det([p_i, p_j, p_k])|
        # det = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
        
        # To be fast, we can loop over i and vectorize j, k
        # Or fully vectorize if n is small enough (30 is small)
        
        # Full vectorization approach:
        # Create indices
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        # Filter i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        pi = pts[i_flat]
        pj = pts[j_flat]
        pk = pts[k_flat]
        
        # Determinant calculation
        # det = pi_x * (pj_y - pk_y) + pj_x * (pk_y - pi_y) + pk_x * (pi_y - pj_y)
        d = pi[:, 0] * (pj[:, 1] - pk[:, 1]) + pj[:, 0] * (pk[:, 1] - pi[:, 1]) + pk[:, 0] * (pi[:, 1] - pj[:, 1])
        d = np.abs(d)
        
        # Exclude zero area
        d = d[d > 0]
        
        if len(d) == 0:
            return 0
            
        # Find max multiplicity
        # Using np.unique with return_counts is fast
        vals, counts = np.unique(d, return_counts=True)
        return int(np.max(counts))

    def make_grid(w, h, dx, dy, ox, oy):
        pts = []
        for i in range(w):
            for j in range(h):
                x = ox + i * dx
                y = oy + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts if len(pts) == 30 else None

    def make_lattice(s, ox, oy):
        # Generate points on a lattice scaled by s
        # We need 30 points.
        # Try a hexagonal or square lattice pattern centered at (ox, oy)
        pts = set()
        # Generate candidates in a box large enough to hold 30 points
        # s=1: 30 points in ~6x5 grid
        # s=2: 30 points in ~3x2 grid (sparse)
        # Let's just generate a grid of size sufficient and pick 30
        # Or generate specific patterns
        
        # Pattern 1: Square grid
        # Pattern 2: Hexagonal grid (shifted rows)
        
        # Simple approach: generate a grid of width W, height H such that W*H >= 30
        # And scale by s
        # Then center it
        
        # For s=1, we can use a 6x5 grid
        # For s=2, we can use a 6x5 grid with spacing 2 (fits in 10x8)
        # For s=3, we can use a 6x5 grid with spacing 3 (fits in 15x12)
        
        # Let's try to generate a 6x5 grid for any s, scaled
        w, h = 6, 5
        grid_pts = []
        for i in range(w):
            for j in range(h):
                x = ox + i * s
                y = oy + j * s
                if -30 <= x <= 30 and -30 <= y <= 30:
                    grid_pts.append((x, y))
        
        if len(grid_pts) == 30:
            return grid_pts
            
        # Try 5x6
        w, h = 5, 6
        grid_pts = []
        for i in range(w):
            for j in range(h):
                x = ox + i * s
                y = oy + j * s
                if -30 <= x <= 30 and -30 <= y <= 30:
                    grid_pts.append((x, y))
        if len(grid_pts) == 30:
            return grid_pts
            
        return None

    # Generate initial candidates
    candidates = []
    
    # 1. Structured Grids
    for s in [1, 2, 3, 4]:
        # Center the grid
        # For 6x5 grid with step s: width = 5*s, height = 4*s
        # Center ox = -2.5*s, oy = -2*s -> use integer offsets
        ox = -((6 - 1) * s) // 2
        oy = -((5 - 1) * s) // 2
        g = make_grid(6, 5, s, s, ox, oy)
        if g:
            candidates.append(g)
            
        ox = -((5 - 1) * s) // 2
        oy = -((6 - 1) * s) // 2
        g = make_grid(5, 6, s, s, ox, oy)
        if g:
            candidates.append(g)
            
    # 2. Lattice-based (Hexagonal approximation)
    # Create a hex-like pattern
    for s in [1, 2, 3]:
        pts = []
        # Generate rows with alternating shifts
        # Row y = 0, s, 2s...
        # x offset = 0, s/2, 0...
        # Since we need integer coords, if s is even, shift is integer.
        # If s is odd, we can't do half-integers easily without scaling coords.
        # Let's stick to integer shifts.
        
        # Simple hex grid:
        # (i, j) -> (i + j//2, j * (s * 3 // 2)) ? No, that's not integer for all.
        # Let's use a simple skew: (i, j) -> (i + j//2, j*s)
        # This creates a parallelogram lattice.
        
        max_range = 30
        count = 0
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * s + (j // 2) * s
                y = j * s
                if -max_range <= x <= max_range and -max_range <= y <= max_range:
                    pts.append((x, y))
                    if len(pts) >= 30:
                        break
            if len(pts) >= 30:
                break
        
        if len(pts) == 30:
            # Center it
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            # Check bounds
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in centered):
                candidates.append(centered)

    # 3. Random
    random.seed(42)
    for _ in range(20):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))
        
    # Evaluate initial candidates
    scored = []
    for c in candidates:
        s = score(c)
        scored.append((s, c))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_pts = scored[0]
    
    # Simulated Annealing
    def simulated_annealing(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        
        start_temp = 1.0
        end_temp = 0.01
        start_time = time.time()
        end_time = start_time + duration
        
        # We need to estimate time per iteration to set cooling schedule?
        # Or just use a fixed number of steps if we can estimate speed.
        # Let's do time-based with a fixed step size reduction.
        
        t = start_temp
        cool_rate = 0.999  # Adjust based on expected iterations
        
        while time.time() < end_time:
            # Generate move
            # 1. Move one point
            # 2. Swap two points
            
            if random.random() < 0.7:
                i = random.randint(0, 29)
                old_p = current_pts[i]
                # New position: random nearby or random anywhere
                if random.random() < 0.8:
                    nx = old_p[0] + random.randint(-3, 3)
                    ny = old_p[1] + random.randint(-3, 3)
                else:
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                
                new_p = (nx, ny)
                if -30 <= nx <= 30 and -30 <= ny <= 30 and new_p not in current_set:
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    new_score = score(current_pts)
                    
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / t):
                        current_score = new_score
                    else:
                        current_pts[i] = old_p
                        current_set.remove(new_p)
                        current_set.add(old_p)
            else:
                # Swap
                i = random.randint(0, 29)
                j = random.randint(0, 29)
                if i == j:
                    continue
                pi = current_pts[i]
                pj = current_pts[j]
                
                if random.random() < 0.8:
                    nxi = pi[0] + random.randint(-3, 3)
                    nyi = pi[1] + random.randint(-3, 3)
                    nxj = pj[0] + random.randint(-3, 3)
                    nyj = pj[1] + random.randint(-3, 3)
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
                    
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / t):
                        current_score = new_score
                    else:
                        current_pts[i] = pi
                        current_pts[j] = pj
                        current_set.remove(np_i)
                        current_set.remove(np_j)
                        current_set.add(pi)
                        current_set.add(pj)
                        
            t *= cool_rate
            
            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_pts = [p for p in current_pts]
                
        return best_score, best_pts

    # Run SA on top candidates
    # Allocate time to top 3
    search_times = [20, 15, 10]
    for idx in range(min(3, len(scored))):
        if time.time() > total_deadline - 5:
            break
        rem = total_deadline - time.time() - 2
        if rem <= 0:
            break
        t = min(search_times[idx], rem)
        s, p = simulated_annealing(scored[idx][1], t)
        if s > best_score:
            best_score = s
            best_pts = p

    return best_pts

