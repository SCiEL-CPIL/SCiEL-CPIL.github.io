import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin under 90s wall-clock
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)
    
    # Start with known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Target: find A,B such that A+B has 9 pairwise coprime distinct values
    # Strategy: pick target sums that are pairwise coprime, then solve for A,B
    
    # Precompute small primes and 1 for potential sum targets
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Generate candidate target sum sets (9 pairwise coprime values)
    # Use 1 and 8 primes, or 9 primes
    target_sums_list = []
    
    # Option 1: {1} + 8 small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Try combinations of 8 primes from small_primes, add 1
    for combo in combinations(small_primes[:20], 8):
        target = [1] + list(combo)
        target.sort()
        # Check if all pairwise coprime (1 is coprime to all, primes are coprime to each other)
        # This is always true for distinct primes + 1
        target_sums_list.append(target)
    
    # Also try 9 primes without 1
    for combo in combinations(small_primes[:15], 9):
        target = list(combo)
        target.sort()
        target_sums_list.append(target)
    
    # For each target sum set, try to find A and B
    # A = {a1, a2, a3}, B = {b1, b2, b3}
    # A+B must equal target set (as a set)
    # So we need: {a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3} = target
    
    # Sort target: t1 < t2 < ... < t9
    # a1+b1 = t1 (min sum)
    # a3+b3 = t9 (max sum)
    # a1+b3, a3+b1 are next extremes
    
    for target in target_sums_list:
        if time.time() > deadline - 5:
            break
        if len(target) != 9:
            continue
        t = sorted(target)
        t1, t2, t3, t4, t5, t6, t7, t8, t9 = t
        
        # Try different assignments
        # a1+b1 = t1, a3+b3 = t9
        # Let a1 = x, b1 = t1 - x
        # a3 = y, b3 = t9 - y
        # Need 1 <= x, y, t1-x, t9-y <= 128
        
        # The other sums are:
        # a1+b2 = x + b2
        # a1+b3 = x + (t9 - y) = x + t9 - y
        # a2+b1 = a2 + (t1 - x)
        # a2+b2 = a2 + b2
        # a2+b3 = a2 + (t9 - y)
        # a3+b1 = y + (t1 - x) = y + t1 - x
        # a3+b2 = y + b2
        
        # Let b2 = z, a2 = w
        # Sums: x+z, x+t9-y, w+t1-x, w+z, w+t9-y, y+t1-x, y+z
        # These 7 + t1 + t9 = 9 sums must equal t
        
        # Try ranges for x, y, z, w
        # x in [1, 128], t1-x in [1, 128] => x in [t1-128, t1-1] intersect [1,128]
        # y in [1, 128], t9-y in [1, 128] => y in [t9-128, t9-1] intersect [1,128]
        
        x_min = max(1, t1 - 128)
        x_max = min(128, t1 - 1)
        y_min = max(1, t9 - 128)
        y_max = min(128, t9 - 1)
        
        # Limit search for efficiency
        x_range = range(x_min, x_max + 1)
        y_range = range(y_min, y_max + 1)
        
        # To keep it manageable, sample or limit
        if len(x_range) > 50:
            x_range = list(x_range)[::max(1, len(x_range)//50)]
        if len(y_range) > 50:
            y_range = list(y_range)[::max(1, len(y_range)//50)]
        
        for x in x_range:
            if time.time() > deadline - 3:
                break
            b1 = t1 - x
            if b1 < 1 or b1 > 128:
                continue
            for y in y_range:
                if time.time() > deadline - 3:
                    break
                b3 = t9 - y
                if b3 < 1 or b3 > 128:
                    continue
                # Now find z (b2) and w (a2)
                # The 7 middle sums must be {t2,...,t8}
                # Sums: x+z, x+t9-y, w+t1-x, w+z, w+t9-y, y+t1-x, y+z
                # Let's fix z and solve for w, or iterate
                
                # x+t9-y is fixed
                s_fixed = x + t9 - y
                if s_fixed not in t[1:8]:
                    continue
                
                # y+t1-x is fixed
                s_fixed2 = y + t1 - x
                if s_fixed2 not in t[1:8]:
                    continue
                
                # Remaining 5 sums: x+z, w+t1-x, w+z, w+t9-y, y+z
                # These must be the remaining 5 values from t[1:8] minus {s_fixed, s_fixed2}
                remaining = set(t[1:8]) - {s_fixed, s_fixed2}
                
                # Try z in reasonable range
                z_min = max(1, t2 - x, t2 - y)
                z_max = min(128, t8 - x, t8 - y)
                if z_max < z_min:
                    continue
                
                # Sample z values
                z_step = max(1, (z_max - z_min) // 30)
                for z in range(z_min, z_max + 1, z_step):
                    if time.time() > deadline - 2:
                        break
                    # x+z must be in remaining
                    if x + z not in remaining:
                        continue
                    # y+z must be in remaining
                    if y + z not in remaining:
                        continue
                    
                    # Now w+t1-x, w+z, w+t9-y must be the remaining 3
                    rem3 = remaining - {x + z, y + z}
                    if len(rem3) != 3:
                        continue
                    
                    # w = rem_val - (t1-x) for each rem_val, check consistency
                    candidates_w = set()
                    for rv in rem3:
                        w1 = rv - (t1 - x)
                        w2 = rv - z
                        w3 = rv - (t9 - y)
                        # w must satisfy all three
                        if w1 == w2 == w3:
                            candidates_w.add(w1)
                    
                    # Actually, w+t1-x, w+z, w+t9-y are three different expressions
                    # They must equal three values in rem3
                    # Let me think differently: w is determined by one equation
                    # w = val - (t1-x) for some val in rem3
                    # Then check if w+z and w+t9-y are the other two in rem3
                    
                    for w in candidates_w:
                        if w < 1 or w > 128:
                            continue
                        s1 = w + (t1 - x)
                        s2 = w + z
                        s3 = w + (t9 - y)
                        if {s1, s2, s3} == rem3:
                            # Found a valid solution!
                            A = sorted([x, w, y])
                            B = sorted([b1, z, b3])
                            # Verify
                            if len(set(A)) == 3 and len(set(B)) == 3:
                                if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
                                    sc = score_of(A, B)
                                    if sc >= 9:
                                        return A, B
                                    if sc > best_score:
                                        best_score = sc
                                        best_A = A
                                        best_B = B
    
    # Fallback: broader search if target-based didn't find 9
    if best_score < 9:
        import random
        random.seed(42)
        
        # Smarter random: bias towards structured sets
        while time.time() < deadline - 2:
            # Try arithmetic progressions or near-arithmetic
            step_a = random.randint(5, 30)
            step_b = random.randint(3, 20)
            a1 = random.randint(1, 128 - 2*step_a)
            b1 = random.randint(1, 128 - 2*step_b)
            
            A = [a1, a1 + step_a, a1 + 2*step_a]
            B = [b1, b1 + step_b, b1 + 2*step_b]
            
            if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
                sc = score_of(A, B)
                if sc > best_score:
                    best_score = sc
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        break
            
            # Also pure random
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
            sc = score_of(A, B)
            if sc > best_score:
                best_score = sc
                best_A = A
                best_B = B
                if best_score == 9:
                    break
    
    return best_A, best_B
