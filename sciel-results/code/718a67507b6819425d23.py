"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _all_int_distances(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_square(_dist2(points[i], points[j])):
                return False
    return True

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    return True

def _are_concyclic(p1, p2, p3, p4):
    a1 = p1[0]; b1 = p1[1]; c1 = a1*a1 + b1*b1
    a2 = p2[0]; b2 = p2[1]; c2 = a2*a2 + b2*b2
    a3 = p3[0]; b3 = p3[1]; c3 = a3*a3 + b3*b3
    a4 = p4[0]; b4 = p4[1]; c4 = a4*a4 + b4*b4
    d1 = a2 - a1; e1 = b2 - b1; f1 = c2 - c1
    d2 = a3 - a1; e2 = b3 - b1; f2 = c3 - c1
    d3 = a4 - a1; e3 = b4 - b1; f3 = c4 - c1
    det = d1*(e2*f3 - e3*f2) - e1*(d2*f3 - d3*f2) + f1*(d2*e3 - d3*e2)
    return det == 0

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _is_valid(points):
    if not _all_int_distances(points):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _generate_candidates(max_param=200, max_radius=1000):
    """Generate candidate points using rational parametrization of circles."""
    candidates = set()
    # Add origin
    candidates.add((0, 0))
    
    # For each radius r, generate points on circle x^2 + y^2 = r^2
    # using rational parametrization: x = r*(1-t^2)/(1+t^2), y = r*2t/(1+t^2)
    for r in range(1, max_radius + 1):
        if r > max_radius:
            break
        for t_num in range(-max_param, max_param + 1):
            for t_den in range(1, max_param + 1):
                if math.gcd(t_num, t_den) != 1:
                    continue
                # x = r * (1 - t^2) / (1 + t^2), y = r * 2t / (1 + t^2)
                # where t = t_num/t_den
                # x = r * (den^2 - num^2) / (den^2 + num^2)
                # y = r * 2*num*den / (den^2 + num^2)
                num_sq = t_num * t_num
                den_sq = t_den * t_den
                denom = den_sq + num_sq
                if denom == 0:
                    continue
                x_num = r * (den_sq - num_sq)
                y_num = r * 2 * t_num * t_den
                # Check if these divide evenly to give integers
                if x_num % denom == 0 and y_num % denom == 0:
                    x = x_num // denom
                    y = y_num // denom
                    # Verify it's on the circle
                    if x * x + y * y == r * r:
                        candidates.add((x, y))
    return candidates

def _try_add_point(current_points, candidate, deadline):
    """Try to add candidate to current_points if all constraints are satisfied."""
    n = len(current_points)
    
    # Check integer distances to all existing points
    for p in current_points:
        d2 = _dist2(candidate, p)
        if not _is_square(d2):
            return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            if _are_collinear(candidate, current_points[i], current_points[j]):
                return False
    
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_concyclic(candidate, current_points[i], current_points[j], current_points[k]):
                    return False
    
    return True

def _greedy_expand(seed_points, candidates, deadline):
    """Greedily add points from candidates to seed_points."""
    points = seed_points[:]
    used = set(seed_points)
    
    while time.time() < deadline:
        added = False
        for cand in candidates:
            if cand in used:
                continue
            if _try_add_point(points, cand, deadline):
                points.append(cand)
                used.add(cand)
                added = True
                break
        if not added:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = base_points[:]
    
    # Verify base is valid
    assert _is_valid(best), "Base configuration should be valid"
    
    # Generate candidate points using rational parametrization
    candidates = _generate_candidates(max_param=150, max_radius=800)
    candidate_list = list(candidates)
    
    # Also add some grid points for broader coverage
    for x in range(-100, 101):
        for y in range(-100, 101):
            candidate_list.append((x, y))
    
    # Remove duplicates
    candidate_list = list(set(candidate_list))
    
    # Sort candidates by distance from origin for more structured search
    candidate_list.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    # Greedy expansion
    expanded = _greedy_expand(best, candidate_list, deadline)
    
    if len(expanded) > len(best):
        best = expanded
    
    # Try different starting points for diversity
    if time.time() < deadline - 10:
        # Try starting from subsets of different sizes
        for start_size in [3, 4]:
            if time.time() > deadline - 5:
                break
            # Take first start_size points from base
            sub_seed = base_points[:start_size]
            if not _is_valid(sub_seed):
                continue
            expanded2 = _greedy_expand(sub_seed, candidate_list, deadline)
            if len(expanded2) > len(best):
                best = expanded2
    
    # Final validation
    if not _is_valid(best):
        best = base_points[:]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}

# EVOLVE-BLOCK-END
