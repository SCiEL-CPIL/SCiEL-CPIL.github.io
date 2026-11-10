"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import comb

_DEADLINE = 85.0  # safety margin before 90s kill

def _doubled_areas(points):
    """Compute absolute doubled areas for all triples of 30 points.
    Returns array of shape (4060,) with absolute determinant values."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    # For each triple (i,j,k), doubled area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # Vectorize: compute all pairs first
    # Use broadcasting for efficiency
    i_idx = np.repeat(np.arange(n), comb(n,2))
    j_idx = np.concatenate([np.arange(i, n) for i in range(n-1)])
    # Actually let's use a different approach for triples
    # For 30 points, C(30,3) = 4060 triples
    # Compute all pair differences first
    # pts[i] - pts[j] for i < j
    diffs = np.zeros((n*(n-1)//2, 2), dtype=np.int64)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            diffs[idx] = pts[j] - pts[i]
            idx += 1
    
    # For triple (i,j,k) with i<j<k: area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # Let's compute directly
    areas = np.zeros(comb(n,3), dtype=np.int64)
    idx = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[idx] = abs(v1[0]*v2[1] - v1[1]*v2[0])
                idx += 1
    return areas

def _score(points):
    """Return the maximum multiplicity of any positive doubled area."""
    areas = _doubled_areas(points)
    positive = areas[areas > 0]
    if len(positive) == 0:
        return 0
    counts = np.bincount(positive)
    return int(counts.max())

def _generate_diverse_initial():
    """Generate a diverse initial configuration."""
    strategy = random.choice(['grid', 'diagonal', 'parabolic', 'random', 'ring', 'spiral'])
    
    if strategy == 'grid':
        # 6x5 grid
        pts = [(x, y) for x in range(6) for y in range(5)]
    elif strategy == 'diagonal':
        # Points along several diagonal lines
        pts = []
        for d in range(-10, 11, 2):
            for t in range(15):
                x = t + d
                y = t
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
                    if len(pts) == 30:
                        break
            if len(pts) >= 30:
                break
        # Fill if needed
        while len(pts) < 30:
            pts.append((random.randint(-30,30), random.randint(-30,30)))
        pts = pts[:30]
    elif strategy == 'parabolic':
        pts = []
        for a in range(-2, 3):
            for t in range(6):
                x = t * 2 - 5
                y = a * x * x + random.randint(-3, 3)
                if -30 <= y <= 30:
                    pts.append((x, y))
                    if len(pts) == 30:
                        break
            if len(pts) >= 30:
                break
        while len(pts) < 30:
            pts.append((random.randint(-30,30), random.randint(-30,30)))
        pts = pts[:30]
    elif strategy == 'ring':
        # Points on concentric circles
        pts = []
        radii = [3, 6, 9, 12, 15]
        for r in radii:
            for angle in range(0, 360, 360 // 6):
                x = round(r * np.cos(np.radians(angle)))
                y = round(r * np.sin(np.radians(angle)))
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((int(x), int(y)))
                    if len(pts) == 30:
                        break
            if len(pts) >= 30:
                break
        while len(pts) < 30:
            pts.append((random.randint(-30,30), random.randint(-30,30)))
        pts = pts[:30]
    elif strategy == 'spiral':
        pts = []
        for i in range(30):
            angle = i * 0.5
            r = 1 + i * 0.8
            x = round(r * np.cos(angle))
            y = round(r * np.sin(angle))
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((int(x), int(y)))
            else:
                pts.append((random.randint(-30,30), random.randint(-30,30)))
    else:
        pts = [(random.randint(-30,30), random.randint(-30,30)) for _ in range(30)]
    
    # Ensure distinctness
    unique = list(dict.fromkeys(pts))
    while len(unique) < 30:
        unique.append((random.randint(-30,30), random.randint(-30,30)))
    return unique[:30]

def _mutate(points, strength=1):
    """Move one or two points by a small random amount."""
    pts = list(points)
    n_moves = 1 if random.random() < 0.7 else 2
    for _ in range(n_moves):
        idx = random.randrange(30)
        dx = random.randint(-strength, strength)
        dy = random.randint(-strength, strength)
        if dx == 0 and dy == 0:
            dx = random.choice([-1, 1])
        new_pt = (pts[idx][0] + dx, pts[idx][1] + dy)
        # Clamp to bounds
        new_pt = (max(-30, min(30, new_pt[0])), max(-30, min(30, new_pt[1])))
        # Check distinctness
        if new_pt not in pts or new_pt == pts[idx]:
            pts[idx] = new_pt
    return pts

def construct():
    start_time = time.time()
    best_points = _generate_diverse_initial()
    best_score = _score(best_points)
    
    # Simulated annealing
    temperature = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < _DEADLINE:
        if temperature < min_temp:
            # Restart with new diverse initial
            temperature = 5.0
            new_init = _generate_diverse_initial()
            new_score = _score(new_init)
            if new_score > best_score:
                best_points = new_init
                best_score = new_score
            points = new_init
        else:
            # Mutate
            strength = max(1, int(temperature * 2))
            new_points = _mutate(points, strength)
            
            # Quick check for distinctness
            if len(set(new_points)) != 30:
                temperature *= cooling
                continue
            
            new_score = _score(new_points)
            
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
            
            if new_score >= _score(points) or random.random() < np.exp(-( _score(points) - new_score) / temperature):
                points = new_points
            # else: reject, keep current points
            
            temperature *= cooling
    
    return best_points
# EVOLVE-BLOCK-END
