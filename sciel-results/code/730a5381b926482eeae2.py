"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations
from fractions import Fraction

def is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def check_no_three_collinear(points):
    """Check no three points are collinear. points is list of (x, y) tuples of Fractions."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def check_no_four_concyclic(points, D=1):
    """Check no four points are concyclic. For D=1, circle: x^2+y^2+Ax+By+C=0"""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # For D=1: determinant of [x^2+y^2, x, y, 1] for 4 points
                    # Use exact arithmetic
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    # Subtract first row from others to simplify
                    b2 = a2 - a1
                    c2 = x2 - x1
                    d2 = y2 - y1
                    b3 = a3 - a1
                    c3 = x3 - x1
                    d3 = y3 - y1
                    b4 = a4 - a1
                    c4 = x4 - x1
                    d4 = y4 - y1
                    # 3x3 determinant
                    det3 = b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4)
                    if det3 == 0:
                        return False
    return True

def dist_sq(p1, p2, D=1):
    """Squared distance between two points (x, y*sqrt(D)) with rational coords."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + D*dy*dy

def is_rational_square(val):
    """Check if a Fraction is a perfect square of a rational."""
    if val < 0:
        return False
    # val = num/den, check if num*den is a perfect square
    num = val.numerator
    den = val.denominator
    if num == 0:
        return True
    return is_perfect_square(num * den)

def is_valid_set(points, D=1):
    """Check all constraints for a set of points."""
    n = len(points)
    if n < 3:
        return False
    pts_set = set(points)
    if len(pts_set) != n:
        return False
    # Check pairwise distances are rational squares
    for i in range(n):
        for j in range(i+1, n):
            dsq = dist_sq(points[i], points[j], D)
            if not is_rational_square(dsq):
                return False
    if not check_no_three_collinear(points):
        return False
    if not check_no_four_concyclic(points, D):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Known 5-point solution with D=1
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # All pairwise distances are integers
    best_points = [(Fraction(0), Fraction(0)),
                   (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)),
                   (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    best_count = len(best_points)
    D = 1
    
    def generate_pythagorean_candidates(max_radius=2000):
        """Generate candidate points with integer distance to origin."""
        candidates = set()
        for m in range(1, int(math.sqrt(max_radius*max_radius)) + 1):
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m*m - n*n
                    b = 2*m*n
                    c = m*m + n*n
                    if c <= max_radius:
                        for sa in [-1, 1]:
                            for sb in [-1, 1]:
                                candidates.add((Fraction(sa*a), Fraction(sb*b)))
                                candidates.add((Fraction(sa*b), Fraction(sb*a)))
        # Axis points
        for r in range(1, max_radius+1):
            candidates.add((Fraction(r), Fraction(0)))
            candidates.add((Fraction(-r), Fraction(0)))
            candidates.add((Fraction(0), Fraction(r)))
            candidates.add((Fraction(0), Fraction(-r)))
        return list(candidates)
    
    def quick_distance_check(candidate, current_points, D=1):
        """Quick check: candidate has rational distance to all current points."""
        for px, py in current_points:
            dsq = dist_sq((candidate[0], candidate[1]), (px, py), D)
            if not is_rational_square(dsq):
                return False
        return True
    
    # Strategy 1: Systematic search for 6th point
    current = list(best_points)
    D = 1
    current_set = set(current)
    
    # Generate a larger pool of candidates more efficiently
    all_candidates = set()
    max_r = 10000
    for m in range(1, int(math.sqrt(max_r*max_r)) + 1):
        for n in range(1, m):
            if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                a = m*m - n*n
                b = 2*m*n
                c = m*m + n*n
                if c <= max_r:
                    for sa in [-1, 1]:
                        for sb in [-1, 1]:
                            all_candidates.add((Fraction(sa*a), Fraction(sb*b)))
                            all_candidates.add((Fraction(sa*b), Fraction(sb*a)))
    # Add axis-aligned points
    for r in range(1, max_r+1, 1):
        all_candidates.add((Fraction(r), Fraction(0)))
        all_candidates.add((Fraction(-r), Fraction(0)))
        all_candidates.add((Fraction(0), Fraction(r)))
        all_candidates.add((Fraction(0), Fraction(-r)))
    
    # Sort candidates by distance to origin for systematic search
    candidates = sorted(all_candidates, key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    # First pass: quick distance check to all current points
    viable_candidates = []
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in current_set:
            continue
        if not quick_distance_check(cand, current, D):
            continue
        viable_candidates.append(cand)
    
    # Second pass: check full validity for viable candidates
    for cand in viable_candidates:
        if time.time() > deadline:
            break
        new_set = current + [cand]
        if is_valid_set(new_set, D):
            current.append(cand)
            current_set.add(cand)
            if len(current) > best_count:
                best_count = len(current)
                best_points = list(current)
            # If we found a 6th point, try to find a 7th
            if len(current) >= 6:
                # Regenerate candidates excluding current points
                for cand2 in candidates:
                    if time.time() > deadline:
                        break
                    if cand2 in current_set:
                        continue
                    if not quick_distance_check(cand2, current, D):
                        continue
                    new_set2 = current + [cand2]
                    if is_valid_set(new_set2, D):
                        current.append(cand2)
                        current_set.add(cand2)
                        if len(current) > best_count:
                            best_count = len(current)
                            best_points = list(current)
    
    # Strategy 2: Try different D values with more systematic search
    D = 1  # Reset D
    if best_count < 6 and time.time() < deadline - 30:
        for D_val in [2, 3, 5, 7, 11, 13, 17, 19]:
            if time.time() > deadline - 10:
                break
            D = D_val
            # For general D, try to build a set from scratch
            base_points = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
            if not is_valid_set(base_points, D):
                continue
            
            current = list(base_points)
            current_set = set(current)
            
            # Generate candidates for this D
            candidates_D = set()
            max_r_D = 500
            for m in range(1, int(math.sqrt(max_r_D*max_r_D)) + 1):
                for n in range(1, m):
                    if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                        a = m*m - n*n
                        b = 2*m*n
                        c = m*m + n*n
                        if c <= max_r_D:
                            for sa in [-1, 1]:
                                for sb in [-1, 1]:
                                    candidates_D.add((Fraction(sa*a), Fraction(sb*b)))
                                    candidates_D.add((Fraction(sa*b), Fraction(sb*a)))
            
            candidates_D = sorted(candidates_D, key=lambda p: p[0]*p[0] + D*p[1]*p[1])
            
            # Try to extend
            for cand in candidates_D:
                if time.time() > deadline - 10:
                    break
                if cand in current_set:
                    continue
                if not quick_distance_check(cand, current, D):
                    continue
                new_set = current + [cand]
                if is_valid_set(new_set, D):
                    current.append(cand)
                    current_set.add(cand)
                    if len(current) > best_count:
                        best_count = len(current)
                        best_points = list(current)
                        # Save D for this best set
                        best_D = D
    
    # Strategy 3: Local search - try to replace points in best set
    D = 1
    if time.time() < deadline - 30:
        current = list(best_points)
        candidates = generate_pythagorean_candidates(3000)
        for iteration in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for cand in candidates:
                    if cand in current:
                        continue
                    new_set = current[:i] + current[i+1:] + [cand]
                    if is_valid_set(new_set, D):
                        if len(new_set) > len(best_points):
                            best_points = new_set
                            best_count = len(best_points)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Strategy 4: Final attempt with even larger search space
    D = 1
    if time.time() < deadline - 15 and best_count < 6:
        current = list(best_points)
        current_set = set(current)
        
        # Generate candidates with much larger radius
        max_r_final = 20000
        final_candidates = set()
        for m in range(1, int(math.sqrt(max_r_final*max_r_final)) + 1):
            for n in range(1, m):
                if (m - n) % 2 == 1 and math.gcd(m, n) == 1:
                    a = m*m - n*n
                    b = 2*m*n
                    c = m*m + n*n
                    if c <= max_r_final:
                        for sa in [-1, 1]:
                            for sb in [-1, 1]:
                                final_candidates.add((Fraction(sa*a), Fraction(sb*b)))
                                final_candidates.add((Fraction(sa*b), Fraction(sb*a)))
        
        final_candidates = sorted(final_candidates, key=lambda p: p[0]*p[0] + p[1]*p[1])
        
        for cand in final_candidates:
            if time.time() > deadline - 5:
                break
            if cand in current_set:
                continue
            if not quick_distance_check(cand, current, D):
                continue
            new_set = current + [cand]
            if is_valid_set(new_set, D):
                current.append(cand)
                current_set.add(cand)
                if len(current) > best_count:
                    best_count = len(current)
                    best_points = list(current)
    
    # Final verification
    # Use D=1 for the known 5-point set
    if is_valid_set(best_points, 1):
        # Convert to required format
        points_out = []
        for x, y in best_points:
            points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': 1, 'points': points_out}
    
    # Fallback to known good set
    points_out = []
    for x, y in [(Fraction(0), Fraction(0)),
                 (Fraction(-56), Fraction(33)),
                 (Fraction(-16), Fraction(63)),
                 (Fraction(0), Fraction(33)),
                 (Fraction(16), Fraction(63))]:
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': 1, 'points': points_out}
# EVOLVE-BLOCK-END
