"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    seen = set()
    ordinary = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            cnt = 0
            for (px, py) in pts:
                if a * px + b * py == c:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary

def _local_search(pts, max_time=10.0):
    """Local search with simulated annealing."""
    best_pts = [tuple(p) for p in pts]
    best_count = _count_ordinary(best_pts)
    current_pts = list(best_pts)
    current_count = best_count
    
    rng = random.Random(42)
    deadline = time.time() + max_time
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Generate a move: pick a point and move it to a random nearby position
        idx = rng.randint(0, len(current_pts) - 1)
        x0, y0 = current_pts[idx]
        
        # Try a few random nearby positions
        moved = False
        for _ in range(10):
            dx = rng.randint(-10, 10)
            dy = rng.randint(-10, 10)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x0 + dx, y0 + dy
            
            # Check bounds (loose check, we can rescale later if needed, but task says <= 1e9)
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
                
            # Check collision
            collision = False
            for i, (px, py) in enumerate(current_pts):
                if i != idx and px == nx and py == ny:
                    collision = True
                    break
            if collision:
                continue
                
            new_pts = list(current_pts)
            new_pts[idx] = (nx, ny)
            new_count = _count_ordinary(new_pts)
            
            delta = new_count - current_count
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_pts = new_pts
                current_count = new_count
                moved = True
                if current_count < best_count:
                    best_count = current_count
                    best_pts = list(current_pts)
                break
        
        if not moved:
            # Try a swap
            i = rng.randint(0, len(current_pts) - 1)
            j = rng.randint(0, len(current_pts) - 1)
            if i != j:
                new_pts = list(current_pts)
                new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
                new_count = _count_ordinary(new_pts)
                delta = new_count - current_count
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_pts = new_pts
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_pts = list(current_pts)
        
        T *= cooling

    return best_pts, best_count

def construct():
    # Multi-start local search
    best_pts = None
    best_count = float('inf')
    
    # Start 1: 4x3 grid
    pts1 = [(x, y) for x in range(4) for y in range(3)]
    if time.time() < _DEADLINE - 5:
        p, c = _local_search(pts1, max_time=min(20.0, _DEADLINE - time.time() - 5))
        if c < best_count:
            best_count = c
            best_pts = p
    
    # Start 2: 3x3 grid + 3 random points
    pts2 = [(x, y) for x in range(3) for y in range(3)]
    rng = random.Random(123)
    for _ in range(3):
        pts2.append((rng.randint(-5, 5), rng.randint(-5, 5)))
    # Ensure distinct
    seen = set()
    unique_pts2 = []
    for p in pts2:
        if p not in seen:
            unique_pts2.append(p)
            seen.add(p)
    while len(unique_pts2) < 12:
        p = (rng.randint(-10, 10), rng.randint(-10, 10))
        if p not in seen:
            unique_pts2.append(p)
            seen.add(p)
    
    if time.time() < _DEADLINE - 5:
        p, c = _local_search(unique_pts2, max_time=min(20.0, _DEADLINE - time.time() - 5))
        if c < best_count:
            best_count = c
            best_pts = p
            
    # Start 3: Random
    if time.time() < _DEADLINE - 5:
        pts3 = []
        seen = set()
        while len(pts3) < 12:
            p = (rng.randint(-20, 20), rng.randint(-20, 20))
            if p not in seen:
                pts3.append(p)
                seen.add(p)
        p, c = _local_search(pts3, max_time=min(20.0, _DEADLINE - time.time() - 5))
        if c < best_count:
            best_count = c
            best_pts = p

    # Final polish with remaining time
    if time.time() < _DEADLINE - 2 and best_pts is not None:
        p, c = _local_search(best_pts, max_time=_DEADLINE - time.time() - 2)
        if c < best_count:
            best_count = c
            best_pts = p

    if best_pts is None:
        return [[x, y] for x, y in pts1]
        
    return [[x, y] for x, y in best_pts]
# EVOLVE-BLOCK-END
