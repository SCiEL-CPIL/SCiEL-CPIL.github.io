"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _no_four_concyclic_fast(points, D):
    """Check no four points are concyclic using determinant test.
    Points are (x, y_coeff) where actual y = y_coeff * sqrt(D).
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1c = points[i]
                    x2, y2c = points[j]
                    x3, y3c = points[k]
                    x4, y4c = points[l]
                    
                    def minor3(i1, i2, i3):
                        a, b = points[i1]
                        c, d = points[i2]
                        e, f = points[i3]
                        va = a*a + D*b*b
                        vc = c*c + D*d*d
                        ve = e*e + D*f*f
                        bracket = a*(d*ve - vc*f) - b*(c*ve - vc*e) + va*(c*f - d*e)
                        return bracket
                    
                    m1 = minor3(i, j, k)
                    m2 = minor3(i, j, l)
                    m3 = minor3(i, k, l)
                    m4 = minor3(j, k, l)
                    
                    total = m1 - m2 + m3 - m4
                    if total == 0:
                        return False
    return True

def _precompute_int_dist_offsets(max_coord):
    """Precompute all (dx, dy) with dx^2 + dy^2 a perfect square, |dx|,|dy| <= max_coord."""
    offsets = {}
    for dx in range(-max_coord, max_coord + 1):
        for dy in range(-max_coord, max_coord + 1):
            d2 = dx*dx + dy*dy
            if d2 > 0 and _is_square(d2):
                offsets[(dx, dy)] = d2
    return offsets

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Known 5-point integral point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    best = base_frac
    best_count = 5
    
    # Precompute integer-distance offsets for a reasonable range
    max_coord = 3000
    int_offsets = _precompute_int_dist_offsets(max_coord)
    
    # Strategy: For each base point p_i = (a_i, b_i), a candidate (x, y) must satisfy
    # (x - a_i)^2 + (y - b_i)^2 is a perfect square for ALL i.
    # Equivalently, (dx_i, dy_i) = (x - a_i, y - b_i) must be in int_offsets.
    # So x - a_i = dx_i and y - b_i = dy_i for some (dx_i, dy_i) in int_offsets.
    # This means x = a_i + dx_i and y = b_i + dy_i for each i.
    # For consistency: a_i + dx_i = a_j + dx_j for all i,j => dx_i - dx_j = a_j - a_i.
    
    # More efficient: pick two base points, say p_0 and p_1.
    # x = a_0 + dx_0 = a_1 + dx_1 => dx_1 = dx_0 + (a_0 - a_1)
    # y = b_0 + dy_0 = b_1 + dy_1 => dy_1 = dy_0 + (b_0 - b_1)
    # So for each (dx_0, dy_0) in int_offsets, check if (dx_0 + (a_0-a_1), dy_0 + (b_0-b_1)) is also in int_offsets.
    # Then verify against all other base points.
    
    found_six = False
    a0, b0 = base_points[0]
    a1, b1 = base_points[1]
    da = a0 - a1  # = 56
    db = b0 - b1  # = -33
    
    # For each (dx0, dy0) in int_offsets, check if (dx0+da, dy0+db) is also in int_offsets
    # Then compute candidate (x, y) = (a0+dx0, b0+dy0) and verify all constraints.
    
    for dx0, dy0 in int_offsets:
        if time.time() > deadline:
            break
        
        dx1 = dx0 + da
        dy1 = dy0 + db
        
        if (dx1, dy1) not in int_offsets:
            continue
        
        # Candidate point
        x = a0 + dx0
        y = b0 + dy0
        
        # Check against all other base points
        all_ok = True
        for i in range(2, 5):
            ai, bi = base_points[i]
            dxi = x - ai
            dyi = y - bi
            if (dxi, dyi) not in int_offsets:
                all_ok = False
                break
        if not all_ok:
            continue
        
        # Check not collinear with any pair of base points
        collinear = False
        for i in range(5):
            for j in range(i+1, 5):
                x1, y1 = base_frac[i]
                x2_val, y2_val = base_frac[j]
                cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                if cp == 0:
                    collinear = True
                    break
            if collinear:
                break
        if collinear:
            continue
        
        # Check no four concyclic
        candidate = base_frac + [(Fraction(x), Fraction(y))]
        if not _no_four_concyclic_fast(candidate, D):
            continue
        
        # Found a 6-point set!
        best = candidate
        best_count = 6
        found_six = True
        break
    
    # If we found 6 points, try to find a 7th
    if found_six:
        found_seven = False
        a0, b0 = best[0]
        a1, b1 = best[1]
        da = a0 - a1
        db = b0 - b1
        
        for dx0, dy0 in int_offsets:
            if time.time() > deadline:
                break
            
            dx1 = dx0 + da
            dy1 = dy0 + db
            
            if (dx1, dy1) not in int_offsets:
                continue
            
            x = a0 + dx0
            y = b0 + dy0
            
            # Check against all other points in best
            all_ok = True
            for i in range(2, 6):
                ai, bi = best[i]
                dxi = x - ai
                dyi = y - bi
                if (dxi, dyi) not in int_offsets:
                    all_ok = False
                    break
            if not all_ok:
                continue
            
            # Check not collinear with any pair
            collinear = False
            for i in range(6):
                for j in range(i+1, 6):
                    x1, y1 = best[i]
                    x2_val, y2_val = best[j]
                    cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                    if cp == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check no four concyclic
            candidate = best + [(Fraction(x), Fraction(y))]
            if not _no_four_concyclic_fast(candidate, D):
                continue
            
            best = candidate
            best_count = 7
            found_seven = True
            break
    
    # If still only 5 or 6 points, try a different pair of base points to expand search
    if best_count < 7:
        # Try using different anchor pairs to find more candidates
        for anchor_i in range(min(3, best_count)):
            for anchor_j in range(anchor_i+1, min(4, best_count)):
                if time.time() > deadline:
                    break
                
                a0, b0 = best[anchor_i]
                a1, b1 = best[anchor_j]
                da = a0 - a1
                db = b0 - b1
                
                for dx0, dy0 in int_offsets:
                    if time.time() > deadline:
                        break
                    
                    dx1 = dx0 + da
                    dy1 = dy0 + db
                    
                    if (dx1, dy1) not in int_offsets:
                        continue
                    
                    x = a0 + dx0
                    y = b0 + dy0
                    
                    # Check against all other points in best
                    all_ok = True
                    for i in range(len(best)):
                        if i == anchor_i or i == anchor_j:
                            continue
                        ai, bi = best[i]
                        dxi = x - ai
                        dyi = y - bi
                        if (dxi, dyi) not in int_offsets:
                            all_ok = False
                            break
                    if not all_ok:
                        continue
                    
                    # Check not collinear with any pair
                    collinear = False
                    for i in range(len(best)):
                        for j in range(i+1, len(best)):
                            x1, y1 = best[i]
                            x2_val, y2_val = best[j]
                            cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                            if cp == 0:
                                collinear = True
                                break
                        if collinear:
                            break
                    if collinear:
                        continue
                    
                    # Check no four concyclic
                    candidate = best + [(Fraction(x), Fraction(y))]
                    if not _no_four_concyclic_fast(candidate, D):
                        continue
                    
                    best = candidate
                    best_count += 1
                    break
                if time.time() > deadline:
                    break
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
