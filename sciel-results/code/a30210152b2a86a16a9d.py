"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from fractions import Fraction
from itertools import combinations

def construct():
    """Search for integral point sets with no three collinear and no four concyclic."""
    t_start = time.time()
    DEADLINE = t_start + 130  # 20s safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def is_rational_square(frac):
        """Check if a Fraction is a perfect square of a rational number."""
        if frac < 0:
            return False
        num, den = frac.numerator, frac.denominator
        return is_square(num) and is_square(den)

    def check_collinear(p1, p2, p3):
        """Check if three points are collinear using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        """Check if four points are concyclic using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        
        return det == 0

    def validate_incremental(points, new_point):
        """Check if new_point can be added to points (all constraints satisfied)."""
        # Check distance to all existing points
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_rational_square(Fraction(d2.numerator, d2.denominator)):
                return False
        
        # Check collinearity with all pairs
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        
        return True

    def generate_candidates_from_point(center, max_radius=100, max_den=8):
        """Generate rational points at rational distance from center."""
        candidates = set()
        cx, cy = center
        # For each rational radius r = a/b, find rational (dx, dy) with dx^2 + dy^2 = r^2
        # Use parametric form: dx = r*(1-t^2)/(1+t^2), dy = r*2t/(1+t^2)
        for b in range(1, max_den + 1):
            for a in range(1, max_radius * b + 1):
                r = Fraction(a, b)
                # Try several t values
                for tb in range(1, max_den + 1):
                    for ta in range(-max_den, max_den + 1):
                        if tb == 0:
                            continue
                        t = Fraction(ta, tb)
                        denom = 1 + t*t
                        if denom == 0:
                            continue
                        dx = r * (1 - t*t) / denom
                        dy = r * 2 * t / denom
                        # Check bounds
                        if abs(dx.numerator) > 10000 or abs(dy.numerator) > 10000:
                            continue
                        if dx.denominator > 10000 or dy.denominator > 10000:
                            continue
                        cand = (cx + dx, cy + dy)
                        if cand != center:
                            candidates.add(cand)
        return candidates

    def generate_integer_candidates(center, max_radius=200):
        """Generate integer points at integer distance from center."""
        candidates = set()
        cx, cy = int(center[0]), int(center[1]) if isinstance(center[1], int) else 0
        for r in range(1, max_radius + 1):
            r2 = r * r
            for dx in range(-r, r + 1):
                dy2 = r2 - dx * dx
                if dy2 < 0:
                    continue
                dy = isqrt(dy2)
                if dy * dy != dy2:
                    continue
                for dy_sign in ([1, -1] if dy != 0 else [0]):
                    for dx_sign in ([1, -1] if dx != 0 else [0]):
                        cand = (cx + dx_sign * dx, cy + dy_sign * dy)
                        if cand != center:
                            candidates.add(cand)
        return candidates

    def find_best_candidate(points, point_set, candidates):
        """Find the best candidate to add from a list."""
        for cand in candidates:
            if cand in point_set:
                continue
            if validate_incremental(points, cand):
                return cand
        return None

    # Start with known good 5-point set
    best_points = [(Fraction(0), Fraction(0)),
                   (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)),
                   (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    point_set = set(best_points)
    
    random.seed(42)
    search_range = 150
    stuck_counter = 0
    max_stuck = 8
    best_size_ever = len(best_points)
    
    # Try to add points one at a time, multiple passes
    while time_ok():
        current_size = len(best_points)
        if current_size >= 8:
            break
        
        improved = False
        
        if not time_ok():
            break
        
        # Strategy 1: Generate candidates using rational distance constraints
        all_candidates = set()
        
        # Use rational candidates from each existing point (limited for speed)
        for p in best_points:
            if not time_ok():
                break
            rat_cands = generate_candidates_from_point(p, max_radius=50, max_den=4)
            all_candidates.update(rat_cands)
        
        # Also add integer Pythagorean candidates
        for p in best_points:
            if not time_ok():
                break
            int_cands = generate_integer_candidates(p, max_radius=150)
            all_candidates.update(int_cands)
        
        # Add some random rational candidates
        for _ in range(300):
            x = Fraction(random.randint(-search_range, search_range), random.randint(1, 5))
            y = Fraction(random.randint(-search_range, search_range), random.randint(1, 5))
            all_candidates.add((x, y))
        
        # Remove existing points
        all_candidates -= point_set
        
        # Convert to list and shuffle for diversity
        candidate_list = list(all_candidates)
        random.shuffle(candidate_list)
        
        # Try to find a valid candidate
        found = find_best_candidate(best_points, point_set, candidate_list)
        if found:
            best_points.append(found)
            point_set.add(found)
            improved = True
            stuck_counter = 0
            best_size_ever = max(best_size_ever, len(best_points))
            search_range = min(search_range + 20, 400)
        
        if improved:
            continue
        
        # Strategy 2: Local search with rational offsets
        for p in best_points:
            if not time_ok():
                break
            local_cands = set()
            for dx_n in range(-20, 21):
                for dy_n in range(-20, 21):
                    if dx_n == 0 and dy_n == 0:
                        continue
                    for den in [1, 2, 3]:
                        cand = (p[0] + Fraction(dx_n, den), p[1] + Fraction(dy_n, den))
                        local_cands.add(cand)
            
            local_cands -= point_set
            random.shuffle(list(local_cands))
            found = find_best_candidate(best_points, point_set, list(local_cands))
            if found:
                best_points.append(found)
                point_set.add(found)
                improved = True
                best_size_ever = max(best_size_ever, len(best_points))
                break
        
        if improved:
            continue
        
        # Strategy 3: Symmetric extensions
        for p in list(best_points):
            if not time_ok():
                break
            sym_cands = [(p[0], -p[1]), (-p[0], p[1]), (-p[0], -p[1]),
                         (p[1], p[0]), (p[1], -p[0]), (-p[1], p[0]), (-p[1], -p[0])]
            sym_cands = [c for c in sym_cands if c not in point_set]
            found = find_best_candidate(best_points, point_set, sym_cands)
            if found:
                best_points.append(found)
                point_set.add(found)
                improved = True
                best_size_ever = max(best_size_ever, len(best_points))
                break
        
        if improved:
            continue
        
        # Strategy 4: Backtracking
        stuck_counter += 1
        if stuck_counter >= max_stuck and len(best_points) > 3:
            idx = len(best_points) - 1 if random.random() < 0.5 else random.randint(0, len(best_points)-1)
            removed = best_points.pop(idx)
            point_set.discard(removed)
            stuck_counter = 0
            all_candidates = set()
            for p in best_points:
                rat_cands = generate_candidates_from_point(p, max_radius=60, max_den=5)
                all_candidates.update(rat_cands)
            all_candidates -= point_set
            candidate_list = list(all_candidates)
            random.shuffle(candidate_list)
            found = find_best_candidate(best_points, point_set, candidate_list)
            if found:
                best_points.append(found)
                point_set.add(found)
            else:
                best_points.append(removed)
                point_set.add(removed)
        
        if not improved and stuck_counter < max_stuck:
            search_range += 20
            if search_range > 600:
                break
    
    # Final validation
    valid = True
    for i in range(len(best_points)):
        for j in range(i+1, len(best_points)):
            dx = best_points[i][0] - best_points[j][0]
            dy = best_points[i][1] - best_points[j][1]
            d2 = dx*dx + dy*dy
            if not is_rational_square(Fraction(d2.numerator, d2.denominator)):
                valid = False
                break
        if not valid:
            break
        for k in range(j+1, len(best_points)):
            if check_collinear(best_points[i], best_points[j], best_points[k]):
                valid = False
                break
        if not valid:
            break
    if valid:
        for i in range(len(best_points)):
            for j in range(i+1, len(best_points)):
                for k in range(j+1, len(best_points)):
                    for l in range(k+1, len(best_points)):
                        if check_concyclic(best_points[i], best_points[j], best_points[k], best_points[l]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            best_points = [(Fraction(0), Fraction(0)),
                          (Fraction(-56), Fraction(33)),
                          (Fraction(-16), Fraction(63)),
                          (Fraction(0), Fraction(33)),
                          (Fraction(16), Fraction(63))]
    
    # Format the result
    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
# EVOLVE-BLOCK-END
