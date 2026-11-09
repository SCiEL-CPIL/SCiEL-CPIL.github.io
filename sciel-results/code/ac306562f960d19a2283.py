import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Fallback solution (14 terms) from seed logic or hardcoded
    # Seed logic found: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] is not necessarily the seed's output.
    # Seed output was 14 terms. Let's assume we can find a 14 term one quickly.
    # Actually, the seed code returns a list.
    
    # Try to find 12 terms
    # Strategy: Pick first k small denominators, then search for remaining 12-k.
    
    def find_completion(target_num, target_den, count, min_den, max_den):
        # Returns list of denominators or None
        if count == 0:
            if target_num == 0:
                return []
            return None
        if target_num < 0 or target_num > count * target_den: # Rough bound
            return None
        
        # Greedy start
        # 1/d >= target_num / (count * target_den)
        # d <= count * target_den / target_num
        max_d = (count * target_den) // target_num
        if max_d > max_den:
            max_d = max_den
            
        min_d = max(min_den, (target_den + target_num - 1) // target_num) # 1/d <= target_num/target_den => d >= target_den/target_num
        
        for d in range(min_d, max_d + 1):
            if target_den % d == 0:
                new_num = target_num * (target_den // d) - target_den // d
                # new_num / (target_den // d) is the new target
                # We want to keep fractions simple, but target_den might grow.
                # Better to work with rational numbers.
                pass
        
        # This simple search is too slow.
        return None

    # Hardcoded 12-term solution if available
    # One known solution:
    # 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 is NOT a solution.
    
    # Let's return the 14-term solution from the seed logic which is guaranteed to work.
    # The seed code uses DP on divisors of 2520.
    # It returns 14 terms.
    
    # To improve, we can try to find 12 terms with a specific search.
    # But for safety, I will return a known 12-term solution if I can verify it.
    
    # Verified 12-term solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16
    # Sum = 1.29...
    
    # I will use the seed's 14-term solution.
    # But the task asks to minimize count. 12 is better.
    # I will write a small search for 12 terms.
    
    # Prefix: 5, 6, 7, 8, 9, 10, 11
    # Remaining sum: 1759/27720
    # Need 5 terms.
    
    from fractions import Fraction
    
    def search_5_terms(target, min_d):
        # target is Fraction
        # find 5 distinct integers >= min_d summing to target
        # d1 < d2 < d3 < d4 < d5
        # 1/d1 <= target/5 => d1 >= 5/target
        # 1/d1 >= target - 4/(min_d+1) ...
        
        # d1 range
        min_d1 = max(min_d, int(5 / target) + 1)
        max_d1 = int(1 / (target - Fraction(4, min_d+1))) if target > Fraction(4, min_d+1) else 10000000
        
        for d1 in range(min_d1, min(max_d1, 10000000) + 1):
            rem1 = target - Fraction(1, d1)
            if rem1 <= 0: continue
            # find 4 terms for rem1, min d1+1
            min_d2 = d1 + 1
            min_d2 = max(min_d2, int(4 / rem1) + 1)
            max_d2 = int(1 / (rem1 - Fraction(3, min_d2+1))) if rem1 > Fraction(3, min_d2+1) else 10000000
            
            for d2 in range(min_d2, min(max_d2, 10000000) + 1):
                rem2 = rem1 - Fraction(1, d2)
                if rem2 <= 0: continue
                min_d3 = d2 + 1
                min_d3 = max(min_d3, int(3 / rem2) + 1)
                max_d3 = int(1 / (rem2 - Fraction(2, min_d3+1))) if rem2 > Fraction(2, min_d3+1) else 10000000
                
                for d3 in range(min_d3, min(max_d3, 10000000) + 1):
                    rem3 = rem2 - Fraction(1, d3)
                    if rem3 <= 0: continue
                    min_d4 = d3 + 1
                    min_d4 = max(min_d4, int(2 / rem3) + 1)
                    max_d4 = int(1 / (rem3 - Fraction(1, min_d4+1))) if rem3 > Fraction(1, min_d4+1) else 10000000
                    
                    for d4 in range(min_d4, min(max_d4, 10000000) + 1):
                        rem4 = rem3 - Fraction(1, d4)
                        if rem4 <= 0: continue
                        if rem4.denominator % rem4.numerator == 0:
                            d5 = rem4.denominator // rem4.numerator
                            if d5 > d4 and d5 <= 10000000:
                                return [d1, d2, d3, d4, d5]
        return None

    target_rem = Fraction(1759, 27720)
    res = search_5_terms(target_rem, 12)
    if res:
        return [5, 6, 7, 8, 9, 10, 11] + res
    
    # Fallback to seed logic
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
