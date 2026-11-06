<<<<<<< SEARCH
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
        for j in range(i + 1, n):
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _line_key_from_two(p1, p2):
    """Compute normalized line key from two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key_from_two(points[i], points[j])
            if key is None:
                continue
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

def _score_points_delta(points, old_pt, new_pt, idx, prev_lines):
    """
    Fast delta scoring: update line counts when one point moves.
    prev_lines: dict mapping line_key -> set of point indices on that line
    Returns (new_score, new_has_5, new_lines)
    """
    n = 16
    # Build new lines from scratch but efficiently
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key_from_two(points[i], points[j])
            if key is None:
                continue
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
    return exact_4, has_5, dict(lines)
>>>>>>> REPLACE

