import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    deadline = time.time() + 550  # safety margin

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Represent field elements as integers 0..48: element a + b*alpha = a + b*p
    # Field multiplication: (a1+b1*alpha)(a2+b2*alpha) = a1*a2 + b1*b2*3 + (a1*b2+a2*b1)*alpha
    
    # Precompute multiplication table for speed
    mul_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = x % p, x // p
        for y in range(q):
            ya, yb = y % p, y // p
            ra = (xa * ya + xb * yb * 3) % p
            rb = (xa * yb + xb * ya) % p
            mul_table[x, y] = ra + rb * p
    
    # Precompute subtraction table
    sub_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = x % p, x // p
        for y in range(q):
            ya, yb = y % p, y // p
            sub_table[x, y] = (xa - ya) % p + ((xb - yb) % p) * p
    
    def field_sub(x, y):
        return int(sub_table[x, y])
    
    def field_mul(x, y):
        return int(mul_table[x, y])
    
    def check_permutation(vals):
        """Check if vals (list/array of field elements) is a permutation of F_49"""
        return len(set(vals)) == q
    
    def check_function(f_vals):
        """
        Check if f: x -> f_vals[x] gives a valid excluded set.
        f_vals is a list of length q, f_vals[x] = f(x).
        Need: f is permutation AND f(x) - m*x is permutation for all m in F_49.
        """
        # Check f is a permutation (m=0 constraint)
        if len(set(f_vals)) != q:
            return False
        # Check f(x) - m*x is permutation for all m
        for m in range(q):
            vals = []
            for x in range(q):
                val = field_sub(f_vals[x], field_mul(m, x))
                vals.append(val)
            if len(set(vals)) != q:
                return False
        return True
    
    def check_function_fast(f_vals):
        """
        Faster check using numpy. f_vals is a list of length q.
        """
        f_arr = np.array(f_vals, dtype=np.int32)
        
        # Check f is a permutation
        if len(set(f_arr.tolist())) != q:
            return False
        
        for m in range(q):
            # Compute f(x) - m*x for all x
            mx = np.array([field_mul(m, x) for x in range(q)], dtype=np.int32)
            vals = np.array([field_sub(f_arr[x], mx[x]) for x in range(q)], dtype=np.int32)
            if len(set(vals.tolist())) != q:
                return False
        return True
    
    # Try to find a valid f by searching
    # A complete orthomorphism: permutation f such that f(x) - m*x is permutation for all m
    # This is a "planar permutation" or related to perfect nonlinear functions
    
    found = None
    best_trial = None
    
    # Strategy 1: Try to find a complete orthomorphism via randomized search
    random.seed(42)
    
    # First, try some structured candidates
    # Try f(x) = x^d for various exponents d
    # In F_49, the multiplicative group has order 48
    # x^d is a permutation iff gcd(d, 48) = 1 (for x != 0), and f(0)=0
    # So d must be coprime to 48: d in {1, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47}
    
    # Check if x^d - m*x is a permutation for all m
    # This requires x^d - m*x to be a permutation polynomial
    
    for d in [5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47]:
        if time.time() > deadline - 200:
            break
        f_vals = []
        for x in range(q):
            if x == 0:
                f_vals.append(0)
            else:
                # Compute x^d in F_49
                result = 1
                base = x
                exp = d
                while exp > 0:
                    if exp % 2 == 1:
                        result = field_mul(result, base)
                    base = field_mul(base, base)
                    exp //= 2
                f_vals.append(result)
        
        if check_function(f_vals):
            found = f_vals
            break
    
    # Strategy 2: Randomized search over permutations
    if found is None:
        for trial in range(5000):
            if time.time() > deadline - 100:
                break
            perm = list(range(q))
            random.shuffle(perm)
            if check_function(perm):
                found = perm
                break
    
    # Strategy 3: Greedy construction
    if found is None:
        # Build f greedily: for each x, pick f(x) such that all constraints are satisfied
        # Constraints: f is a bijection, and for each m, f(x) - m*x is a bijection
        # This means: for each m, the set {f(x) - m*x : x in F_49} must be all of F_49
        
        # Use backtracking/greedy with constraint propagation
        # f_vals[x] = y means:
        #   - y not used by any other x' (permutation constraint)
        #   - for each m, y - m*x not used by any other x' (slope constraint)
        
        # For a partial assignment, maintain:
        #   used_y: set of y values used
        #   used_val[m]: set of values f(x') - m*x' used for each m
        
        used_y = set()
        used_val = [set() for _ in range(q)]
        f_vals = [-1] * q
        
        def can_assign(x, y):
            if y in used_y:
                return False
            for m in range(q):
                val = field_sub(y, field_mul(m, x))
                if val in used_val[m]:
                    return False
            return True
        
        def assign(x, y):
            used_y.add(y)
            for m in range(q):
                val = field_sub(y, field_mul(m, x))
                used_val[m].add(val)
            f_vals[x] = y
        
        def unassign(x, y):
            used_y.discard(y)
            for m in range(q):
                val = field_sub(y, field_mul(m, x))
                used_val[m].discard(val)
            f_vals[x] = -1
        
        # Greedy with randomization: process x in random order
        success = False
        for attempt in range(100):
            if time.time() > deadline - 50:
                break
            
            # Reset
            used_y = set()
            used_val = [set() for _ in range(q)]
            f_vals = [-1] * q
            
            order = list(range(q))
            random.shuffle(order)
            
            ok = True
            for x in order:
                # Find all valid y values
                valid_ys = [y for y in range(q) if can_assign(x, y)]
                if not valid_ys:
                    ok = False
                    break
                # Pick randomly from valid options
                y = random.choice(valid_ys)
                assign(x, y)
            
            if ok:
                found = f_vals
                success = True
                break
    
    if found is not None:
        # Build S from f: S = {(x, f(x)) : x in F_49}
        S = set()
        for x in range(q):
            y = found[x]
            S.add(x + y * q)
        
        all_points = set(range(q * q))
        N = all_points - S
        points = []
        for pt in sorted(N):
            y = pt // q
            x = pt % q
            xa, xb = x % p, x // p
            ya, yb = y % p, y // p
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    else:
        # Fallback: use a smaller excluded set based on a unital-like construction
        # Or use a simple valid construction
        
        # Simple valid construction: exclude points on a line not containing any other constraint
        # Actually, let's use the Hermitian curve approach
        # Unital: Tr(x) + Nm(y) = 0 gives 49 points, but we need to verify it works
        
        # Let's use a different approach: exclude a set of points such that
        # for every point p, there's a line through p with all other points in N
        
        # Minimal known construction: use a set of 49 points that forms a "cap"
        # or use the complement of a unital
        
        # For now, use a simple valid set: exclude all points where x = 0 (49 points)
        # Check: for point (0, y), the vertical line x=0 has all other points (0, y') not in N
        # Wait, that's wrong - we need the line through p to have all OTHER points in N
        
        # Let me reconsider. N is the set of points. Every point x (in or not in N) must lie
        # on a line where all points except x are in N.
        
        # If we exclude S = {(x, f(x))}, then for a point p = (x, f(x)) in S:
        # We need a line through p such that all other points on that line are in N (not in S).
        # The line with direction (1, m) through p is {(x + t, f(x) + m*t) : t in F_49}.
        # For this to work, we need (x + t, f(x) + m*t) not in S for all t != 0.
        # (x+t, f(x)+m*t) in S means f(x+t) = f(x) + m*t, i.e., f(x+t) - f(x) = m*t.
        # So we need: for each x, there exists m such that f(x+t) - f(x) != m*t for all t != 0.
        # This is equivalent to: the difference f(x+t) - f(x) divided by t is not constant.
        # i.e., (f(x+t) - f(x))/t takes at least 2 distinct values as t varies over F_49\{0}.
        
        # For a point p = (x, y) NOT in S:
        # We need a line through p with all other points in N.
        # Line (1, m) through p: {(x+t, y+m*t) : t in F_49}.
        # All other points must be in N, i.e., not in S.
        # (x+t, y+m*t) in S means f(x+t) = y + m*t.
        # So we need: for each (x,y) not in S, there exists m such that f(x+t) != y + m*t for all t != 0.
        
        # This is a strong condition. Let me just use a known valid construction.
        
        # Use the construction from the original: exclude points on the Hermitian unital
        # where the y-coordinate is in a specific set.
        
        # Actually, let me just return a simple valid set: N = all points except one line.
        # If N = F_49^2 \ {(0, y) : y in F_49}, then:
        # For p = (0, y) in S: vertical line x=0, other points (0, y') are in S, not N. FAIL.
        
        # Let me use: N = F_49^2 \ {(x, 0) : x in F_49}
        # For p = (x, 0) in S: line with slope m through p: {(x+t, m*t) : t in F_49}.
        # Other points: (x+t, m*t) for t != 0. In S iff m*t = 0 iff t=0 (since m*t=0 means m=0 or t=0).
        # If m != 0, then m*t = 0 only when t=0. So all other points are in N. GOOD.
        # For p = (x, y) not in S (y != 0): need a line through p with all others in N.
        # Line (1, m) through p: {(x+t, y+m*t) : t in F_49}.
        # In S iff y+m*t = 0 iff t = -y/m (if m != 0).
        # So there's exactly one t value where the point is in S. We need to avoid that.
        # If we pick m such that -y/m is not a valid t... but t ranges over all F_49.
        # So for any m != 0, there's exactly one point on the line in S.
        # We need a line where NO other point is in S.
        # The only way is if the line doesn't intersect S at all (except possibly at p, but p not in S).
        # Line (1,m) through (x,y): intersects S={(x',0)} when y+m*t=0, i.e., t=-y/m.
        # So it always intersects S at exactly one point (for m!=0).
        # For m=0: line {(x+t, y) : t in F_49}, y!=0, never in S. GOOD!
        # So for p not in S, use horizontal line (m=0). All points on it have y-coordinate y!=0, so none in S.
        
        # This works! S = {(x, 0) : x in F_49}, |S| = 49, |N| = 2401 - 49 = 2352.
        
        S = set()
        for x in range(q):
            S.add(x + 0 * q)  # (x, 0)
        
        all_points = set(range(q * q))
        N = all_points - S
        points = []
        for pt in sorted(N):
            y = pt // q
            x = pt % q
            xa, xb = x % p, x // p
            ya, yb = y % p, y // p
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
