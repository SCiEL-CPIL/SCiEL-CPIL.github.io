# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import math

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    def is_valid(pts):
        if pts.shape != (n, 2):
            return False
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return False
        if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
            return False
        # Check pairwise distances and collinearity
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1,y1 = pts[i]
                    x2,y2 = pts[j]
                    x3,y3 = pts[k]
                    det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                    if abs(det) < 1e-9:
                        return False
        return True
    
    def count_convex_7goms(pts):
        """Count convex 7-gons using the specified method."""
        order = np.lexsort((pts[:,1], pts[:,0]))
        sorted_pts = pts[order]
        
        C = 0
        for i in range(n):
            pi = sorted_pts[i]
            later = sorted_pts[i+1:]
            m = len(later)
            if m < 6:
                break
            
            angles = np.arctan2(later[:,1] - pi[1], later[:,0] - pi[0])
            ang_order = np.argsort(angles)
            sorted_later = later[ang_order]
            
            # Brute force over all 6-combinations
            for combo in itertools.combinations(range(m), 6):
                s = sorted_later[list(combo)]
                pts7 = np.vstack([pi.reshape(1,2), s])
                convex = True
                for t in range(7):
                    p1 = pts7[t]
                    p2 = pts7[(t+1) % 7]
                    p3 = pts7[(t+2) % 7]
                    cross = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
                    if cross <= 1e-9:
                        convex = False
                        break
                if convex:
                    C += 1
        return C
    
    # Initial: random with fixed seed
    np.random.seed(0)
    best_pts = np.random.rand(n, 2)
    while not is_valid(best_pts):
        best_pts = np.random.rand(n, 2)
    best_count = count_convex_7goms(best_pts)
    
    # Local search with limited iterations
    step_sizes = [0.05, 0.02, 0.01]
    directions = [(1,0), (-1,0), (0,1), (0,-1), 
                  (1/math.sqrt(2), 1/math.sqrt(2)), (1/math.sqrt(2), -1/math.sqrt(2)),
                  (-1/math.sqrt(2), 1/math.sqrt(2)), (-1/math.sqrt(2), -1/math.sqrt(2))]
    
    improved = True
    while improved and time.time() < deadline - 180:
        improved = False
        for i in range(n):
            if time.time() > deadline - 180:
                break
            for step in step_sizes:
                if time.time() > deadline - 180:
                    break
                for dx, dy in directions:
                    new_pts = best_pts.copy()
                    new_pts[i, 0] += dx * step
                    new_pts[i, 1] += dy * step
                    new_pts = np.clip(new_pts, 0.001, 0.999)
                    if not is_valid(new_pts):
                        continue
                    new_count = count_convex_7goms(new_pts)
                    if new_count < best_count:
                        best_pts = new_pts
                        best_count = new_count
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
