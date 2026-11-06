"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def max_sum_from(start, count):
        """Max possible sum of 'count' distinct reciprocals starting from 'start'."""
        if count <= 0:
            return Fraction(0, 1)
        # Use the smallest 'count' denominators
        s = Fraction(0, 1)
        for i in range(count):
            s += Fraction(1, start + i)
        return s

    def find_last_two(r, prev_d):
        """Find two denominators d7, d8 > prev_d such that 1/d7 + 1/d8 = r."""
        a, b = r.numerator, r.denominator
        if a <= 0 or b <= 0:
            return None
        min_d7 = max(prev_d + 1, b // a + (1 if b % a else 0))
        max_d7 = min(10000000, 2 * b // a)
        for d7 in range(min_d7, max_d7 + 1):
            num = a * d7 - b
            if num <= 0:
                continue
            if (d7 * b) % num != 0:
                continue
            d8 = (d7 * b) // num
            if d8 <= d7 or d8 > 10000000:
                continue
            return d7, d8
        return None

    # Best-first search using priority queue
    # State: (priority, idx, prev_d, current_list, remaining_frac)
    # Priority: how close remaining_frac is to max achievable (higher = more urgent)
    
    def best_first(d1, target_frac):
        """Find 7 denominators > d1 summing to target_frac."""
        # Priority queue: (negative_slack, idx, prev_d, current, remaining)
        # slack = max_possible - remaining; smaller slack means tighter, explore first
        pq = []
        max7 = max_sum_from(d1 + 1, 7)
        if max7 < target_frac:
            return None
        slack = max7 - target_frac
        heapq.heappush(pq, (float(slack), 0, d1, [d1], target_frac))
        
        while pq:
            if time.time() > deadline:
                return None
            neg_slack, idx, prev_d, current, remaining = heapq.heappop(pq)
            
            if idx == 5:
                # Need 2 more terms
                result = find_last_two(remaining, prev_d)
                if result is not None:
                    return current + [result[0], result[1]]
                continue
            
            rem = 6 - idx  # number of terms to choose (including current level)
            max_possible = max_sum_from(prev_d + 1, rem)
            if max_possible < remaining:
                continue
            
            min_d = max(prev_d + 1, remaining.denominator // remaining.numerator + (1 if remaining.denominator % remaining.numerator else 0))
            max_d = min(10000000, rem * remaining.denominator // remaining.numerator)
            if max_d < min_d:
                continue
            
            # Try denominators, prioritizing those that leave tightest slack
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return None
                new_rem = remaining - Fraction(1, d)
                if new_rem <= 0:
                    continue
                rem2 = rem - 1
                if rem2 > 0:
                    max_possible2 = max_sum_from(d + 1, rem2)
                    if max_possible2 < new_rem:
                        continue
                    slack2 = max_possible2 - new_rem
                    heapq.heappush(pq, (float(slack2), idx + 1, d, current + [d], new_rem))
                else:
                    # rem2 == 0, but this shouldn't happen since idx < 5
                    pass
        return None

    # Try d1 from 5 upward
    for d1 in range(5, 25):
        if time.time() > deadline:
            break
        r_after_d1 = Fraction(1) - Fraction(1, d1)
        if r_after_d1 <= 0:
            continue
        max7 = max_sum_from(d1 + 1, 7)
        if max7 < r_after_d1:
            continue
        result = best_first(d1, r_after_d1)
        if result is not None:
            return result

    # Fallback: hardcoded 8-term solution
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
# EVOLVE-BLOCK-END
