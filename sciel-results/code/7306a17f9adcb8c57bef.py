import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0

    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = int(math.isqrt(d2))
        return r*r == d2

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))

        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
        ]

        def det4(m):
            return (m[0][3]*det3([[m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]], [m[3][0], m[3][1], m[3][2]]]) -
                    m[1][3]*det3([[m[0][0], m[0][1], m[0][2]], [m[2][0], m[2][1], m[2][2]], [m[3][0], m[3][1], m[3][2]]]) +
                    m[2][3]*det3([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[3][0], m[3][1], m[3][2]]]) -
                    m[3][3]*det3([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]))
        return det4(m) == 0

    def find_extension(points, max_range, deadline):
        n = len(points)
        min_x = min(p[0] for p in points) - max_range
        max_x = max(p[0] for p in points) + max_range
        min_y = min(p[1] for p in points) - max_range
        max_y = max(p[1] for p in points) + max_range

        # Precompute point set for O(1) lookup
        point_set = set(points)

        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                return None
            for y in range(min_y, max_y + 1):
                p_new = (x, y)
                if p_new in point_set:
                    continue

                # Check distances first (most restrictive)
                valid_dist = True
                for p in points:
                    if not is_int_dist(p_new, p):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue

                # Check collinearity
                valid_col = True
                for i in range(n):
                    for j in range(i+1, n):
                        if is_collinear(points[i], points[j], p_new):
                            valid_col = False
                            break
                    if not valid_col:
                        break
                if not valid_col:
                    continue

                # Check concyclicity
                valid_con = True
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            if is_concyclic(points[i], points[j], points[k], p_new):
                                valid_con = False
                                break
                        if not valid_con:
                            break
                    if not valid_con:
                        break
                if not valid_con:
                    continue

                return p_new
        return None

    best_points = seed_points

    # Try to find 6th point with larger range
    p6 = find_extension(best_points, 500, deadline)
    if p6:
        best_points = best_points + [p6]
        # Try to find 7th point
        p7 = find_extension(best_points, 500, deadline)
        if p7:
            best_points = best_points + [p7]

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
