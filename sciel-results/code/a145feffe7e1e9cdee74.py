"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            a, b = dy_n, -dx_n
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 1:
                a, b, c = a // g2, b // g2, c // g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for key, pts_on_line in lines.items():
        cnt = len(pts_on_line)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def construct():
    deadline = time.time() + 145
    
    # Multiple initial configurations to try
    configs = [
        [[x, y] for x in range(4) for y in range(4)],  # 4x4 grid
        [[x*2, y*2] for x in range(4) for y in range(4)],  # scaled grid
        [[x*3, y] for x in range(4) for y in range(4)],  # stretched
        [[x, y*3] for x in range(4) for y in range(4)],  # stretched other way
        [[x, y] for x in range(5) for y in range(4) if x+y < 8] + [[8,0],[9,0],[10,0],[11,0]],  # custom
    ]
    
    best_score = -1
    best_points = None
    
    for cfg in configs:
        if len(cfg) != 16:
            continue
        s, h5 = _score_points(cfg)
        if not h5 and s > best_score:
            best_score = s
            best_points = [p[:] for p in cfg]
    
    if best_points is None:
        best_points = [p[:] for p in configs[0]]
        best_score, _ = _score_points(best_points)
    
    rng = random.Random(12345)
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current = [p[:] for p in best_points]
        elif rng.random() < 0.7:
            current = [p[:] for p in best_points]
        else:
            # Perturb from a random config
            base_cfg = configs[rng.randint(0, len(configs)-1)]
            if len(base_cfg) == 16:
                current = [p[:] for p in base_cfg]
            else:
                current = [p[:] for p in best_points]
            for _ in range(rng.randint(2, 10)):
                idx = rng.randint(0, 15)
                new_x = current[idx][0] + rng.randint(-6, 6)
                new_y = current[idx][1] + rng.randint(-6, 6)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                new_pt = [new_x, new_y]
                if any(new_pt == p for p in current):
                    continue
                current[idx] = new_pt
        
        s, h5 = _score_points(current)
        if h5:
            continue
        current_score = s
        
        T = 1.0
        no_improve = 0
        max_no_improve = 800
        while time.time() < deadline - 8 and no_improve < max_no_improve:
            idx = rng.randint(0, 15)
            old = current[idx][:]
            
            # Vary jump size based on stagnation
            jump = 3 if no_improve < 100 else (5 if no_improve < 300 else 8)
            new_x = old[0] + rng.randint(-jump, jump)
            new_y = old[1] + rng.randint(-jump, jump)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            new_pt = [new_x, new_y]
            if any(new_pt == p for p in current):
                continue
            
            current[idx] = new_pt
            s_new, h5_new = _score_points(current)
            
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
            
            # Accept if better or with simulated annealing probability
            if s_new > current_score:
                current_score = s_new
                no_improve = 0
                if s_new > best_score:
                    best_score = s_new
                    best_points = [p[:] for p in current]
            elif s_new == current_score and rng.random() < 0.3:
                no_improve += 1
            elif T > 0.01 and rng.random() < math.exp((s_new - current_score) / T):
                current_score = s_new
                no_improve += 1
            else:
                current[idx] = old
                no_improve += 1
            
            T = max(0.01, T * 0.999)
    
    return best_points
# EVOLVE-BLOCK-END
