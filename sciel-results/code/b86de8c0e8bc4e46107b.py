"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter

def _canonical_line(p1, p2):
    """Return a canonical tuple (A, B, C) for the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Normalize by GCD
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Ensure a unique sign representation
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _score(points):
    """
    Calculate the score: number of lines with exactly 4 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    
    score = 0
    for count in line_counts.values():
        # C(k,2) pairs.
        # k=4 -> 6 pairs
        # k=5 -> 10 pairs
        if count >= 10:
            return -1
        if count == 6:
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Strategy: Fix 3 non-collinear points at (0,0), (1,0), (0,1).
    # This exploits the affine invariance (partially) by removing 6 DOF.
    # The 4x4 grid [0,3]x[0,3] contains these 3 points.
    
    fixed_points = [[0, 0], [1, 0], [0, 1]]
    
    # Initial configuration: 4x4 grid
    # Points: (0,0)..(3,3)
    # Fixed: (0,0), (1,0), (0,1)
    # Free: The other 13 points of the grid
    initial_free = []
    for x in range(4):
        for y in range(4):
            if [x, y] not in fixed_points:
                initial_free.append([x, y])
                
    # Verify initial free count is 13
    assert len(initial_free) == 13, f"Expected 13 free points, got {len(initial_free)}"
    
    best_points = fixed_points + [p[:] for p in initial_free]
    best_score = _score(best_points)
    
    current_free = [p[:] for p in initial_free]
    current_points = fixed_points + [p[:] for p in current_free]
    current_score = best_score
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < 0.01:
            T = 5.0
            # Restart from best
            current_free = [p[:] for p in best_points[3:]]
            current_points = fixed_points + [p[:] for p in current_free]
            current_score = best_score
            # Small perturbation
            for _ in range(2):
                idx = random.randint(0, 12)
                current_free[idx][0] += random.randint(-2, 2)
                current_free[idx][1] += random.randint(-2, 2)
                # Bounds check
                current_free[idx][0] = max(-1000, min(1000, current_free[idx][0]))
                current_free[idx][1] = max(-1000, min(1000, current_free[idx][1]))
                # Distinctness check against fixed and other free
                new_p = current_free[idx]
                valid = True
                for fp in fixed_points:
                    if new_p == fp:
                        valid = False
                        break
                if valid:
                    for j, fp in enumerate(current_free):
                        if j != idx and new_p == fp:
                            valid = False
                            break
                if not valid:
                    # Revert if invalid perturbation
                    current_free[idx] = [p[:] for p in best_points[3:]][idx]
            current_points = fixed_points + [p[:] for p in current_free]
            current_score = _score(current_points)
            if current_score == -1:
                current_score = -100

        # Choose move type
        if random.random() < 0.7:
            # 1-point move on a free point
            idx = random.randint(0, 12)
            old_x, old_y = current_free[idx]
            
            # Random jump, biased towards small moves to maintain structure
            # but allow large jumps to escape local minima
            if random.random() < 0.5:
                new_x = random.randint(-5, 5)
                new_y = random.randint(-5, 5)
            else:
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            # Check distinctness
            valid_move = True
            for fp in fixed_points:
                if fp[0] == new_x and fp[1] == new_y:
                    valid_move = False
                    break
            if valid_move:
                for j, p in enumerate(current_free):
                    if j != idx and p[0] == new_x and p[1] == new_y:
                        valid_move = False
                        break
            
            if not valid_move:
                continue
                
            current_free[idx][0] = new_x
            current_free[idx][1] = new_y
            current_points = fixed_points + [p[:] for p in current_free]
            new_score = _score(current_points)
            
            if new_score == -1:
                current_free[idx][0] = old_x
                current_free[idx][1] = old_y
                continue
                
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_free[idx][0] = old_x
                    current_free[idx][1] = old_y
        else:
            # 2-point swap
            i1, i2 = random.sample(range(13), 2)
            p1 = current_free[i1][:]
            p2 = current_free[i2][:]
            
            current_free[i1] = p2
            current_free[i2] = p1
            current_points = fixed_points + [p[:] for p in current_free]
            new_score = _score(current_points)
            
            if new_score == -1:
                current_free[i1] = p1
                current_free[i2] = p2
                continue

            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_free[i1] = p1
                    current_free[i2] = p2
        
        T *= cooling
        
        if time.time() > deadline:
            break
    
    # Final polish
    # Try small moves on best_points
    current_free = [p[:] for p in best_points[3:]]
    current_points = fixed_points + [p[:] for p in current_free]
    current_score = best_score
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        idx = random.randint(0, 12)
        old_x, old_y = current_free[idx]
        
        found_improvement = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                    continue
                
                valid_move = True
                for fp in fixed_points:
                    if fp[0] == new_x and fp[1] == new_y:
                        valid_move = False
                        break
                if valid_move:
                    for j, p in enumerate(current_free):
                        if j != idx and p[0] == new_x and p[1] == new_y:
                            valid_move = False
                            break
                if not valid_move:
                    continue
                
                current_free[idx][0] = new_x
                current_free[idx][1] = new_y
                current_points = fixed_points + [p[:] for p in current_free]
                new_score = _score(current_points)
                
                if new_score != -1 and new_score > current_score:
                    current_score = new_score
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                    no_improve = 0
                    found_improvement = True
                    break
                else:
                    current_free[idx][0] = old_x
                    current_free[idx][1] = old_y
            
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve += 1
            
    return best_points

# EVOLVE-BLOCK-END
