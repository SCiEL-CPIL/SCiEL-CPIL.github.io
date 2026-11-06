"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_valid(points):
    n = len(points)
    # Check integer distances
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_square(_sq_dist(points[i], points[j])):
                return False
    
    # Check no 3 collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    
    # Check no 4 concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    
                    # Circle equation: x^2 + y^2 + Ax + By + C = 0
                    # 2x1 A + 2y1 B + C = -(x1^2 + y1^2)
                    a11, a12, a13 = 2 * x1, 2 * y1, 1
                    a21, a22, a23 = 2 * x2, 2 * y2, 1
                    a31, a32, a33 = 2 * x3, 2 * y3, 1
                    b1 = -(x1 * x1 + y1 * y1)
                    b2 = -(x2 * x2 + y2 * y2)
                    b3 = -(x3 * x3 + y3 * y3)
                    
                    det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
                    if det == 0:
                        continue
                    
                    dA = b1 * (a22 * a33 - a23 * a32) - a12 * (b2 * a33 - a23 * b3) + a13 * (b2 * a32 - a22 * b3)
                    dB = a11 * (b2 * a33 - a23 * b3) - b1 * (a21 * a33 - a23 * a31) + a13 * (a21 * b3 - b2 * a31)
                    dC = a11 * (a22 * b3 - b2 * a32) - a12 * (a21 * b3 - b2 * a31) + b1 * (a21 * a32 - a22 * a31)
                    
                    val = 2 * x4 * dA + 2 * y4 * dB + dC + (x4 * x4 + y4 * y4) * det
                    if val == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Seed points (5 points)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = seed_points
    
    # Search for more points
    # We will try to add points to the seed set
    # To speed up, we pre-compute candidates that have integer distance to all seed points
    
    candidates = []
    R = 100
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            p = (x, y)
            if p in seed_points:
                continue
            valid_dist = True
            for sp in seed_points:
                if not _is_square(_sq_dist(p, sp)):
                    valid_dist = False
                    break
            if valid_dist:
                candidates.append(p)
    
    # Shuffle candidates
    random.shuffle(candidates)
    
    # Greedy add
    current = list(seed_points)
    for p in candidates:
        if time.time() > deadline:
            break
        new_set = current + [p]
        if _check_valid(new_set):
            current = new_set
            if len(current) > len(best):
                best = current
            # Try to add more
            # Re-shuffle or continue?
            # We can restart the candidate list or just continue
    
    # If we didn't find more, try random restarts with different seeds?
    # For now, return the best found
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
