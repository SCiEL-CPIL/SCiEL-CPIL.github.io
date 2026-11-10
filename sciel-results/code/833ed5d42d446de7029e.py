"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area among all triples."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute doubled areas for all triples
    # For points i, j, k: doubled area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # = |(x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i)|
    best_mult = 0
    # Use a dictionary to count areas
    area_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                da = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if da > 0:
                    area_counts[da] = area_counts.get(da, 0) + 1
                    if area_counts[da] > best_mult:
                        best_mult = area_counts[da]
    return best_mult

def _score_fast(points):
    """Faster scoring using NumPy vectorization."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all pairwise vectors
    # For each pair (i,j), compute vector v = pts[j] - pts[i]
    # For each triple (i,j,k), doubled area = |cross(v_ij, v_ik)|
    best_mult = 0
    area_counts = {}
    for i in range(n):
        vi = pts[i]
        for j in range(i+1, n):
            vj = pts[j] - vi
            for k in range(j+1, n):
                vk = pts[k] - vi
                da = abs(vj[0]*vk[1] - vj[1]*vk[0])
                if da > 0:
                    area_counts[da] = area_counts.get(da, 0) + 1
                    if area_counts[da] > best_mult:
                        best_mult = area_counts[da]
    return best_mult

def _generate_rect_grid(w, h, ox=0, oy=0):
    """Generate a rectangular grid of w*h points."""
    return [(ox + x, oy + y) for x in range(w) for y in range(h)]

def _generate_lattice(a, b, c, d, n1, n2, ox=0, oy=0):
    """Generate a lattice: points = ox + n1*(a,c) + n2*(b,d) for n1 in [0,n1), n2 in [0,n2)."""
    pts = []
    for i in range(n1):
        for j in range(n2):
            x = ox + i*a + j*b
            y = oy + i*c + j*d
            pts.append((x, y))
    return pts

def _perturb_points(points, num_moves, rng, max_shift=3):
    """Apply random small perturbations to some points."""
    pts = list(points)
    for _ in range(num_moves):
        idx = rng.randint(0, len(pts)-1)
        x, y = pts[idx]
        dx = rng.randint(-max_shift, max_shift)
        dy = rng.randint(-max_shift, max_shift)
        nx, ny = x + dx, y + dy
        # Check bounds and uniqueness
        if -30 <= nx <= 30 and -30 <= ny <= 30:
            new_pt = (nx, ny)
            if new_pt not in pts or new_pt == pts[idx]:
                pts[idx] = new_pt
    return pts

def _swap_points(points, rng):
    """Swap two points with random nearby positions."""
    pts = list(points)
    i, j = rng.sample(range(len(pts)), 2)
    x1, y1 = pts[i]
    x2, y2 = pts[j]
    # Try to move point i to a random nearby position
    for _ in range(20):
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        nx, ny = x1 + dx, y1 + dy
        if -30 <= nx <= 30 and -30 <= ny <= 30:
            new_pt = (nx, ny)
            if new_pt not in pts:
                pts[i] = new_pt
                break
    # Try to move point j
    for _ in range(20):
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        nx, ny = x2 + dx, y2 + dy
        if -30 <= nx <= 30 and -30 <= ny <= 30:
            new_pt = (nx, ny)
            if new_pt not in pts:
                pts[j] = new_pt
                break
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    rng = random.Random(42)
    
    # Generate candidate initial configurations
    candidates = []
    
    # Rectangular grids
    for w in range(3, 11):
        for h in range(3, 11):
            if w * h == 30:
                for ox in range(-10, 11, 2):
                    for oy in range(-10, 11, 2):
                        pts = _generate_rect_grid(w, h, ox, oy)
                        # Check bounds
                        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                            candidates.append(pts)
    
    # Lattice configurations
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(-5, 6):
                for d in range(1, 8):
                    # Check if it's a valid lattice (non-degenerate)
                    det = a*d - b*c
                    if det == 0:
                        continue
                    for n1 in range(2, 8):
                        for n2 in range(2, 8):
                            if n1 * n2 == 30:
                                for ox in range(-5, 6, 2):
                                    for oy in range(-5, 6, 2):
                                        pts = _generate_lattice(a, b, c, d, n1, n2, ox, oy)
                                        # Check bounds and uniqueness
                                        if len(set(pts)) == 30:
                                            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                                                candidates.append(pts)
    
    # Also try the seed configuration
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    
    # Evaluate all candidates
    best_pts = None
    best_score = 0
    for pts in candidates:
        s = _score_fast(pts)
        if s > best_score:
            best_score = s
            best_pts = list(pts)
        if time.time() > deadline:
            break
    
    if best_pts is None:
        return [(x, y) for x in range(6) for y in range(5)]
    
    # Simulated annealing
    current_pts = list(best_pts)
    current_score = best_score
    temp = 10.0
    cooling = 0.95
    
    while time.time() < deadline - 5:
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.5:
            neighbor = _perturb_points(current_pts, rng.randint(1, 3), rng, max_shift=2)
        else:
            neighbor = _swap_points(current_pts, rng)
        
        # Check validity
        if len(set(neighbor)) != 30:
            continue
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in neighbor):
            continue
        
        # Evaluate
        neighbor_score = _score_fast(neighbor)
        delta = neighbor_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 0.1)):
            current_pts = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_pts = list(current_pts)
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    return best_pts
# EVOLVE-BLOCK-END

