"""Rewritten program for integral point sets in general position."""
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    def is_square(n):
        if n < 0:
            return False
        r = int(n ** 0.5)
        if r * r == n:
            return True
        if (r + 1) * (r + 1) == n:
            return True
        return False
    
    def dist2_int(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    def check_all_int_distances(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if not is_square(dist2_int(points[i], points[j])):
                    return False
        return True
    
    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if cross == 0:
                        return False
        return True
    
    def no_four_concyclic(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        
                        # Determinant for concyclicity
                        a1 = x1 * x1 + y1 * y1
                        a2 = x2 * x2 + y2 * y2
                        a3 = x3 * x3 + y3 * y3
                        a4 = x4 * x4 + y4 * y4
                        
                        # det = | a1, x1, y1, 1 |
                        #       | a2, x2, y2, 1 |
                        #       | a3, x3, y3, 1 |
                        #       | a4, x4, y4, 1 |
                        # Expand along last column:
                        # det = 1*(a1*(x2*y3 - y2*x3) - x1*(a2*y3 - y2*a3) + y1*(a2*x3 - x2*a3))
                        #     - 1*(a1*(x2*y4 - y2*x4) - x1*(a2*y4 - y2*a4) + y1*(a2*x4 - x2*a4))
                        #     + 1*(a1*(x3*y4 - y3*x4) - x1*(a3*y4 - y3*a4) + y1*(a3*x4 - x3*a4))
                        #     - 1*(a2*(x3*y4 - y3*x4) - x2*(a3*y4 - y3*a4) + y2*(a3*x4 - x3*a4))
                        
                        def det3(a, b, c, d, e, f, g, h, i):
                            return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
                        
                        det = (det3(a1, x1, y1, a2, x2, y2, a3, x3, y3)
                             - det3(a1, x1, y1, a2, x2, y2, a4, x4, y4)
                             + det3(a1, x1, y1, a3, x3, y3, a4, x4, y4)
                             - det3(a2, x2, y2, a3, x3, y3, a4, x4, y4))
                        
                        if det == 0:
                            return False
        return True
    
    def validate(points):
        if not check_all_int_distances(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    # Known 5-point set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_set = base5[:]
    
    # Try known 6-point sets
    known_6_sets = [
        [(0, 0), (9, 0), (0, 12), (9, 12), (40, 42), (-14, 48)],
        [(0, 0), (15, 0), (0, 20), (15, 20), (7, 24), (24, 7)],
        [(0, 0), (20, 0), (0, 21), (20, 21), (15, 36), (36, 15)],
    ]
    
    for s in known_6_sets:
        if validate(s) and len(s) > len(best_set):
            best_set = s
    
    # Randomized search to extend the best set
    random.seed(12345)
    search_range = 150
    
    for target in range(len(best_set) + 1, 8):
        found = False
        for attempt in range(30000):
            x = random.randint(-search_range, search_range)
            y = random.randint(-search_range, search_range)
            cand = (x, y)
            if cand in best_set:
                continue
            new_set = best_set + [cand]
            if validate(new_set):
                best_set = new_set
                found = True
                search_range += 30
                break
        if not found:
            break
    
    # If we have 6 points, try harder to find a 7th
    if len(best_set) == 6:
        for attempt in range(100000):
            x = random.randint(-300, 300)
            y = random.randint(-300, 300)
            cand = (x, y)
            if cand in best_set:
                continue
            new_set = best_set + [cand]
            if validate(new_set):
                best_set = new_set
                break
    
    if not validate(best_set):
        best_set = base5[:]
    
    points_formatted = [[[x, 1], [y, 1]] for x, y in best_set]
    return {'D': 1, 'points': points_formatted}
# EVOLVE-BLOCK-END
