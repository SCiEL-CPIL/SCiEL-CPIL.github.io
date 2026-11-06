import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def get_determinants(points):
        """
        Calculate all non-zero absolute determinants for a set of points.
        Returns a Counter of determinants.
        """
        n = len(points)
        pts = np.array(points)
        counts = Counter()
        # Generate all triples indices
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Vector from i to j
                    v1 = pts[j] - pts[i]
                    # Vector from i to k
                    v2 = pts[k] - pts[i]
                    det = abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if det > 0:
                        counts[det] += 1
        return counts

    def evaluate(points):
        if not points:
            return 0
        counts = get_determinants(points)
        if not counts:
            return 0
        return max(counts.values())

    def incremental_move(points, idx, new_pos):
        """
        Returns the change in the max multiplicity if we move points[idx] to new_pos.
        Actually, it's better to just re-evaluate the specific affected determinants 
        and update a global counter, but for simplicity in the search loop, 
        we can compute the new score from scratch if N is small enough, 
        OR implement a class. 
        Given N=30, full re-evaluation is ~4000 ops. 
        4000 * 1000 moves = 4M ops. In Python, 4M ops is ~2-4 seconds.
        So full re-evaluation is feasible for a few thousand steps.
        Let's stick to full re-evaluation for simplicity and robustness, 
        but optimize the inner loop with NumPy if possible.
        """
        pass

    # Optimized evaluation using NumPy broadcasting for speed
    def fast_evaluate(points):
        n = len(points)
        pts = np.array(points, dtype=np.int32)
        # Create indices for all triples
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        # Filter i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        # Get points
        p_i = pts[i_flat]
        p_j = pts[j_flat]
        p_k = pts[k_flat]
        
        # Vectors
        v1 = p_j - p_i
        v2 = p_k - p_i
        
        # Determinants
        dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Filter non-zero
        dets = dets[dets > 0]
        
        if len(dets) == 0:
            return 0
            
        # Count occurrences
        # np.unique returns sorted unique values and counts
        unique_vals, counts = np.unique(dets, return_counts=True)
        return int(np.max(counts))

    # Candidate generators
    def gen_lines(num_lines, pts_per_line, spacing=1, x_start=0):
        pts = []
        for l in range(num_lines):
            y = l * spacing
            for x in range(pts_per_line):
                pts.append((x + x_start, y))
        return pts

    def gen_grid(w, h, x_start=0, y_start=0):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x + x_start, y + y_start))
        return pts

    best_points = None
    best_score = 0

    # 1. Try structured configurations
    configs = []
    
    # Lines
    for nl in range(2, 8):
        for p1 in range(1, 31):
            for p2 in range(1, 31):
                if nl == 2:
                    if p1 + p2 == 30:
                        pts = gen_lines(2, p1, 1, 0) + gen_lines(1, p2, 1, 0) # Wait, gen_lines logic needs adjustment
                        # Let's just build manually for clarity in candidates
                        pass
                # Simplified: Just generate a few specific ones
    # Specific candidates
    candidates = [
        gen_lines(2, 15, 1, 0), # Two lines
        gen_lines(3, 10, 1, 0), # Three lines
        gen_lines(4, 7, 1, 0) + [(7, 0), (7, 1), (7, 2), (7, 3)], # 4 lines approx
        gen_grid(5, 6),
        gen_grid(6, 5),
        gen_grid(10, 3),
        gen_grid(3, 10),
    ]
    
    # Fix gen_lines to allow different counts per line
    def gen_custom_lines(counts, spacing=1):
        pts = []
        for i, c in enumerate(counts):
            for x in range(c):
                pts.append((x, i * spacing))
        return pts

    candidates = [
        gen_custom_lines([15, 15]),
        gen_custom_lines([10, 10, 10]),
        gen_custom_lines([8, 8, 8, 6]),
        gen_custom_lines([6, 6, 6, 6, 6]),
        gen_grid(5, 6),
        gen_grid(6, 5),
        gen_grid(10, 3),
        gen_grid(3, 10),
        gen_grid(15, 2),
    ]

    for cand in candidates:
        if len(set(cand)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in cand):
            continue
        score = fast_evaluate(cand)
        if score > best_score:
            best_score = score
            best_points = cand[:]
        if time.time() > deadline:
            return best_points

    # 2. Local Search
    # We will use fast_evaluate, which is vectorized.
    # It takes ~1-2ms per call.
    # Budget: 85s.
    # Max steps: ~40,000 steps.
    
    random.seed(42)
    current_points = best_points[:]
    current_score = best_score
    
    # Hill Climbing
    steps = 0
    while time.time() < deadline:
        steps += 1
        improved = False
        
        # Try moving each point
        for i in range(30):
            if time.time() > deadline:
                break
            orig_x, orig_y = current_points[i]
            
            # Try small moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    
                    new_points = current_points[:]
                    new_points[i] = (new_x, new_y)
                    
                    # Check distinctness
                    if len(set(new_points)) != 30:
                        continue
                    
                    new_score = fast_evaluate(new_points)
                    
                    if new_score > current_score:
                        current_score = new_score
                        current_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try larger random moves or restart
            if steps % 100 == 0:
                # Random restart with perturbation
                restart_pts = current_points[:]
                for _ in range(5):
                    idx = random.randint(0, 29)
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    nx, ny = restart_pts[idx][0] + dx, restart_pts[idx][1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        restart_pts[idx] = (nx, ny)
                
                if len(set(restart_pts)) == 30:
                    rs = fast_evaluate(restart_pts)
                    if rs > current_score:
                        current_score = rs
                        current_points = restart_pts
                    else:
                        # Try a completely new random configuration occasionally
                        if steps % 500 == 0:
                             rand_pts = [(random.randint(-15, 15), random.randint(-15, 15)) for _ in range(30)]
                             while len(set(rand_pts)) < 30:
                                 idx = random.randint(0, 29)
                                 rand_pts[idx] = (random.randint(-15, 15), random.randint(-15, 15))
                             rs = fast_evaluate(rand_pts)
                             if rs > current_score:
                                 current_score = rs
                                 current_points = rand_pts

        if current_score > best_score:
            best_score = current_score
            best_points = current_points[:]

    return best_points
