# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54 points and m=27 lines on integer lattice.
    Uses vectorized scoring and incremental updates for fast local search.
    """
    start_time = time.time()
    deadline = start_time + 590  # 590s wall-clock limit (safety margin)
    
    def normalize_line(A, B, C):
        g = math.gcd(abs(int(A)), math.gcd(abs(int(B)), abs(int(C))))
        if g == 0:
            return (0, 0, 0)
        A, B, C = int(A) // g, int(B) // g, int(C) // g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    def count_incidents(pts, lns):
        """Vectorized: (54,2) points, (27,3) lines -> int count of A*x+B*y+C==0"""
        # pts: (54, 2), lns: (27, 3)
        # vals[i,j] = lns[i,0]*pts[j,0] + lns[i,1]*pts[j,1] + lns[i,2]
        vals = lns[:, 0:1].T * pts[:, 0:1] + lns[:, 1:2].T * pts[:, 1:2] + lns[:, 2:3].T
        return int(np.sum(vals == 0))
    
    def has_duplicate_points(pts):
        seen = set()
        for p in pts:
            key = (int(p[0]), int(p[1]))
            if key in seen:
                return True
            seen.add(key)
        return False
    
    def has_duplicate_lines(lns):
        seen = set()
        for i in range(len(lns)):
            n = normalize_line(lns[i, 0], lns[i, 1], lns[i, 2])
            if n in seen:
                return True
            seen.add(n)
        return False
    
    # Baseline: k=3 grid construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    assert not has_duplicate_points(points)
    assert not has_duplicate_lines(lines)
    
    best_score = count_incidents(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    np.random.seed(42)
    
    # Build candidate sets
    candidate_xs = list(range(1, 12))
    candidate_ys = list(range(1, 35))
    candidate_points = set()
    for x in candidate_xs:
        for y in candidate_ys:
            candidate_points.add((x, y))
    
    # Candidate lines with small coefficients
    candidate_lines_set = set()
    for A in range(-6, 7):
        for B in range(-6, 7):
            for C in range(-30, 31):
                if A == 0 and B == 0:
                    continue
                n = normalize_line(A, B, C)
                if n != (0, 0, 0):
                    candidate_lines_set.add(n)
    candidate_lines_list = list(candidate_lines_set)
    
    # Local search with incremental scoring
    improvement_found = True
    iteration = 0
    
    while improvement_found:
        improvement_found = False
        iteration += 1
        
        if time.time() > deadline:
            break
        
        # Try replacing each point
        for pi in range(54):
            if time.time() > deadline:
                break
            
            orig_pt = tuple(int(x) for x in best_points[pi])
            
            # Score contribution of original point: how many lines pass through it
            orig_pt_inc = 0
            for li in range(27):
                A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                if A * orig_pt[0] + B * orig_pt[1] + C == 0:
                    orig_pt_inc += 1
            
            for new_pt in candidate_points:
                if new_pt == orig_pt:
                    continue
                # Check duplicate
                if any(tuple(int(x) for x in best_points[j]) == new_pt for j in range(54) if j != pi):
                    continue
                
                # Score contribution of new point
                new_pt_inc = 0
                for li in range(27):
                    A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                    if A * new_pt[0] + B * new_pt[1] + C == 0:
                        new_pt_inc += 1
                
                delta = new_pt_inc - orig_pt_inc
                if delta > 0:
                    new_points = best_points.copy()
                    new_points[pi] = np.array(new_pt, dtype=np.int64)
                    best_score += delta
                    best_points = new_points
                    improvement_found = True
                    break
        
        # Try replacing each line
        for li in range(27):
            if time.time() > deadline:
                break
            
            orig_line = tuple(int(x) for x in best_lines[li])
            
            # Score contribution of original line: how many points lie on it
            orig_line_inc = 0
            for pj in range(54):
                A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                if A * int(best_points[pj, 0]) + B * int(best_points[pj, 1]) + C == 0:
                    orig_line_inc += 1
            
            for new_line in candidate_lines_list:
                if new_line == orig_line:
                    continue
                # Check duplicate
                is_dup = False
                for j in range(27):
                    if j != li and normalize_line(*tuple(int(x) for x in best_lines[j])) == new_line:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Score contribution of new line
                new_line_inc = 0
                A, B, C = new_line
                for pj in range(54):
                    if A * int(best_points[pj, 0]) + B * int(best_points[pj, 1]) + C == 0:
                        new_line_inc += 1
                
                delta = new_line_inc - orig_line_inc
                if delta > 0:
                    new_lines = best_lines.copy()
                    new_lines[li] = np.array(new_line, dtype=np.int64)
                    best_score += delta
                    best_lines = new_lines
                    improvement_found = True
                    break
    
    # Final verification
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    assert not has_duplicate_points(best_points)
    assert not has_duplicate_lines(best_lines)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
