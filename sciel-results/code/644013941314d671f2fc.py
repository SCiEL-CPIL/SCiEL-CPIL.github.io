import time
import math
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit

    best_result = {
        'D': 1,
        'points': [[[0, 1], [0, 1]], 
                   [[-56, 1], [33, 1]], 
                   [[-16, 1], [63, 1]], 
                   [[0, 1], [33, 1]], 
                   [[16, 1], [63, 1]]]
    }
    best_size = 5

    def is_rational_sqrt(num, den):
        """Check if num/den is a perfect square of a rational."""
        if num == 0:
            return True
        if den < 0:
            num = -num
            den = -den
        g = math.gcd(abs(num), abs(den))
        n = abs(num) // g
        d = abs(den) // g
        root_n = int(math.isqrt(n))
        root_d = int(math.isqrt(d))
        if root_n * root_n != n or root_d * root_d != d:
            return False
        return True

    def dist_sq_rational(p1, p2, D):
        """p1, p2 are (Fraction, Fraction). Returns True if distance squared is a rational square."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + D * dy*dy
        if d2 < 0:
            return False
        return is_rational_sqrt(d2.numerator, d2.denominator)

    def check_no_3_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    val = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    if val == 0:
                        return False
        return True

    def det4(m):
        a,b,c,d = m[0]
        e,f,g,h = m[1]
        i,j,k,l = m[2]
        m_,n,o,p = m[3]
        return (
            a*(f*(k*p-l*o) - g*(j*p-l*n) + h*(j*o-k*n)) -
            b*(e*(k*p-l*o) - g*(i*p-l*m_) + h*(i*o-k*m_)) +
            c*(e*(j*p-l*n) - f*(i*p-l*m_) + h*(i*n-j*m_)) -
            d*(e*(j*o-k*n) - f*(i*o-k*m_) + g*(i*n-j*m_))
        )

    def check_no_4_concyclic_with_D(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        r1 = [p1[0]**2 + D*p1[1]**2, p1[0], p1[1], Fraction(1)]
                        r2 = [p2[0]**2 + D*p2[1]**2, p2[0], p2[1], Fraction(1)]
                        r3 = [p3[0]**2 + D*p3[1]**2, p3[0], p3[1], Fraction(1)]
                        r4 = [p4[0]**2 + D*p4[1]**2, p4[0], p4[1], Fraction(1)]
                        M = [r1, r2, r3, r4]
                        if det4(M) == 0:
                            return False
        return True

    def is_valid_set(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_rational(points[i], points[j], D):
                    return False
        if not check_no_3_collinear(points):
            return False
        if not check_no_4_concyclic_with_D(points, D):
            return False
        return True

    def to_output_format(points, D):
        return {
            'D': D,
            'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]
        }

    # Known good 5-point set for D=1
    known_5pts = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    # Verify
    if is_valid_set(known_5pts, 1):
        best_size = 5
        best_result = to_output_format(known_5pts, 1)

    # Generate candidate points with rational distances to a set of anchor points
    def generate_candidates_around(anchors, D, num_candidates=2000):
        """Generate candidate points that have rational distance to all anchors."""
        candidates = []
        # Strategy: for each pair of anchors, find points on the intersection of two circles
        # Circle 1: (x-x1)^2 + D(y-y1)^2 = r1^2
        # Circle 2: (x-x2)^2 + D(y-y2)^2 = r2^2
        # This gives a line in (x, y*sqrt(D)) space, which we can parameterize.
        
        # Simpler approach: random search with bias toward small rationals
        for _ in range(num_candidates):
            # Try small rational coordinates
            x_num = random.randint(-20, 20)
            x_den = random.randint(1, 5)
            y_num = random.randint(-20, 20)
            y_den = random.randint(1, 5)
            x = Fraction(x_num, x_den)
            y = Fraction(y_num, y_den)
            cand = (x, y)
            
            # Check rational distance to all anchors
            valid = True
            for a in anchors:
                if not dist_sq_rational(cand, a, D):
                    valid = False
                    break
            if valid:
                candidates.append(cand)
        
        return candidates

    def generate_candidates_on_line(anchors, D, num_candidates=500):
        """Generate points by parameterizing lines through pairs of anchors."""
        candidates = []
        for i in range(len(anchors)):
            for j in range(i+1, len(anchors)):
                a1, a2 = anchors[i], anchors[j]
                # Line through a1 and a2: parameterize as a1 + t*(a2-a1)
                # For rational t, we get rational coordinates.
                # But we need rational distance to all anchors, not just these two.
                # So this only works if we're building up from 2 points.
                for t_num in range(-10, 11):
                    for t_den in range(1, 4):
                        if t_den == 0:
                            continue
                        t = Fraction(t_num, t_den)
                        x = a1[0] + t * (a2[0] - a1[0])
                        y = a1[1] + t * (a2[1] - a1[1])
                        cand = (x, y)
                        # Check distinct from anchors
                        if cand in anchors:
                            continue
                        # Check rational distance to all anchors
                        valid = True
                        for a in anchors:
                            if not dist_sq_rational(cand, a, D):
                                valid = False
                                break
                        if valid:
                            candidates.append(cand)
        return candidates

    def greedy_expand(points, D, max_size=12):
        """Greedily add points to maximize set size."""
        current = list(points)
        if not current:
            return current
        
        while len(current) < max_size:
            if time.time() > deadline:
                break
            
            # Generate candidates around current points
            anchors = current[:min(3, len(current))]  # Use first 3 as anchors for speed
            cands = generate_candidates_around(anchors, D, 500)
            cands += generate_candidates_on_line(anchors, D, 200)
            
            # Remove duplicates and points already in current
            seen = set(current)
            unique_cands = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    unique_cands.append(c)
            
            # Sort by some heuristic (e.g., distance to centroid)
            if unique_cands:
                cx = sum(p[0] for p in current) / len(current)
                cy = sum(p[1] for p in current) / len(current)
                unique_cands.sort(key=lambda p: (p[0]-cx)**2 + D*(p[1]-cy)**2)
            
            added = False
            for cand in unique_cands:
                if time.time() > deadline:
                    break
                new_pts = current + [cand]
                if is_valid_set(new_pts, D):
                    current = new_pts
                    added = True
                    break
            
            if not added:
                break
        
        return current

    # Try different D values with structured search
    D_candidates = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30]
    
    for D in D_candidates:
        if time.time() > deadline:
            break
        
        # Try multiple starting configurations
        best_for_D = None
        
        if D == 1:
            # Start with known 5-point set
            start_configs = [known_5pts]
            # Also try smaller seeds
            start_configs.append([(Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)), (Fraction(3), Fraction(4))])
        else:
            # For D > 1, start with simple configurations
            # (0,0) and (D,0) have distance D (rational)
            start_configs = [
                [(Fraction(0), Fraction(0)), (Fraction(D), Fraction(0))],
                [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))],
                [(Fraction(0), Fraction(0)), (Fraction(D*D), Fraction(0))],
            ]
        
        for seed in start_configs:
            if time.time() > deadline:
                break
            
            # Verify seed is valid
            if not is_valid_set(seed, D):
                continue
            
            # Run greedy expansion with multiple random restarts
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Shuffle seed for diversity
                shuffled = list(seed)
                random.shuffle(shuffled)
                
                result = greedy_expand(shuffled, D, max_size=12)
                
                if result and (best_for_D is None or len(result) > len(best_for_D)):
                    best_for_D = result
        
        if best_for_D and len(best_for_D) > best_size:
            best_size = len(best_for_D)
            best_result = to_output_format(best_for_D, D)
    
    # If we still have time, try more aggressive search on best D
    if time.time() < deadline - 5:
        # Try to improve the best result found
        if best_size >= 5:
            # Extract D and points from best_result
            D_best = best_result['D']
            pts_best = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) 
                        for p in best_result['points']]
            
            # Try more iterations of greedy expansion
            improved = greedy_expand(pts_best, D_best, max_size=12)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = to_output_format(improved, D_best)
    
    return best_result

# EVOLVE-BLOCK-END
