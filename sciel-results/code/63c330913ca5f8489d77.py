"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Try small k values first with a bounded search
    # k=2: impossible (shown analytically)
    # k=3: search with tight bounds
    # k=4: search with tight bounds
    
    def find_small_k(k, max_den=5000):
        """Search for k distinct denominators >= 5 summing to 1."""
        if time.time() > deadline:
            return None
        
        if k == 3:
            # 1/a + 1/b + 1/c = 1, 5 <= a < b < c
            for a in range(5, max_den):
                if time.time() > deadline:
                    return None
                rem1 = 1 - Fraction(1, a)
                if rem1 <= 0:
                    break
                # 1/b + 1/c = rem1
                # c = b / (b * rem1 - 1) ... let's use: 1/c = rem1 - 1/b = (b*rem1 - 1)/b
                # So c = b / (b * rem1 - 1) must be integer and > b
                for b in range(a + 1, max_den):
                    if time.time() > deadline:
                        return None
                    num = b * rem1.numerator
                    den = b * rem1.denominator
                    # 1/c = rem1 - 1/b = (b*rem1.num - rem1.den) / (b*rem1.den)
                    c_num = b * rem1.numerator - rem1.denominator
                    c_den = b * rem1.denominator
                    if c_num <= 0:
                        break
                    if c_den % c_num == 0:
                        c = c_den // c_num
                        if c > b and c >= 5:
                            return [a, b, c]
            return None
        elif k == 4:
            # 1/a + 1/b + 1/c + 1/d = 1
            for a in range(5, max(10, max_den // 10)):
                if time.time() > deadline:
                    return None
                rem1 = 1 - Fraction(1, a)
                if rem1 <= 0:
                    break
                for b in range(a + 1, max(15, max_den // 5)):
                    if time.time() > deadline:
                        return None
                    rem2 = rem1 - Fraction(1, b)
                    if rem2 <= 0:
                        continue
                    # 1/c + 1/d = rem2
                    for c in range(b + 1, max(20, max_den // 2)):
                        if time.time() > deadline:
                            return None
                        c_num = c * rem2.numerator - rem2.denominator
                        c_den = c * rem2.denominator
                        if c_num <= 0:
                            break
                        if c_den % c_num == 0:
                            d = c_den // c_num
                            if d > c and d >= 5:
                                return [a, b, c, d]
            return None
        else:
            return None
    
    best = None
    
    # Try k=3
    result = find_small_k(3)
    if result is not None:
        best = result
    else:
        # Try k=4
        result = find_small_k(4)
        if result is not None:
            best = result
    
    if best is None:
        # Use DP with a well-chosen common denominator
        # Try several targets and pick the one giving fewest terms
        candidates = [2520, 5040, 7560, 10080, 15120, 20160, 25200, 50400, 100800, 252000]
        
        best_count = float('inf')
        best_result = None
        
        for target in candidates:
            if time.time() > deadline:
                break
            
            # Get all divisors of target that are >= 5
            divisors = []
            for d in range(5, target + 1):
                if target % d == 0:
                    divisors.append(d)
            
            if len(divisors) < 3:
                continue
            
            # DP: subset sum to 'target' using weights target/d
            # reached[sum] = list of denominators used
            reached = {0: []}
            
            for d in divisors:
                if time.time() > deadline:
                    break
                w = target // d
                for total in list(reached.keys()):
                    new_total = total + w
                    if new_total <= target and new_total not in reached:
                        reached[new_total] = reached[total] + [d]
            
            if target in reached:
                sol = reached[target]
                if len(sol) < best_count:
                    best_count = len(sol)
                    best_result = sol
            
            if best_count <= 4:
                break
        
        if best_result is not None:
            best = best_result
    
    if best is None:
        raise RuntimeError("no representation found")
    
    # Verify the solution
    total = Fraction(0, 1)
    for d in best:
        total += Fraction(1, d)
    assert total == 1, f"Sum is {total}, not 1"
    assert len(set(best)) == len(best), "Denominators not distinct"
    assert all(5 <= d <= 10000000 for d in best), "Denominator out of range"
    
    return best

# EVOLVE-BLOCK-END
