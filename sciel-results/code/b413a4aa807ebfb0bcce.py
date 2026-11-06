from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    # Known 11-term solution attempt: try to find solutions with fewer than 12 terms
    # Strategy: Use the identity that if we have sum(1/d_i) = 1, we can split terms
    # 1/n = 1/(n+1) + 1/(n(n+1))
    # This can reduce the number of terms in some cases.
    
    # First, let's try a direct approach: pick k-1 denominators, compute the 12th
    # For k terms: 1/d_1 + ... + 1/d_{k-1} + 1/d_k = 1
    # => 1/d_k = 1 - sum(1/d_i for i=1..k-1)
    # => d_k = 1 / (1 - sum(1/d_i))
    
    # We need d_k to be an integer > max(d_1..d_{k-1}) and <= 10^7
    
    # Better: Use a structured search. For small k, we can enumerate more effectively.
    
    # Let's try k=11 first with a smarter search
    # Key insight: if sum(1/d_i) = 1 with all d_i >= 5, then we need at least 5 terms
    # (since 5 * 1/5 = 1, but they must be distinct, so we need more)
    # Minimum possible: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15
    # Let's compute this sum
    
    def compute_sum(denoms):
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s
    
    # Try a different approach: use the fact that we can construct solutions
    # by starting with a known identity and modifying it.
    
    # Known: 1 = 1/2 + 1/3 + 1/6, but we need all >= 5
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/2520
    # Let me verify this properly
    
    denoms_12 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2520]
    s12 = compute_sum(denoms_12)
    # print(f"Sum of 12-term: {s12} = {float(s12)}")
    
    # If that's not 1, find the correct 12-term solution
    def get_fallback():
        target = 2520
        reached = {0: []}
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
    
    fallback = get_fallback()
    
    # Now try to find solutions with fewer terms
    # For k terms, use backtracking with strong pruning
    
    def try_find_k_terms(k, min_den=5, max_den=10000000):
        """Find k distinct denominators in [min_den, max_den] with reciprocal sum = 1."""
        best = [None]
        visited = set()
        
        def backtrack(remaining, min_d, current, k_left):
            if time.time() > deadline:
                return
            if k_left == 0:
                if remaining == 0:
                    best[0] = current[:]
                return
            
            if k_left == 1:
                # Need 1/d = remaining
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d >= min_d and d <= max_den and d not in current:
                            best[0] = current + [d]
                return
            
            # Pruning: max possible sum with k_left terms starting from min_d
            # Each term <= 1/min_d, so max sum = k_left / min_d
            if remaining > k_left / min_d:
                return
            
            # Min possible sum: each term >= 1/max_den
            if remaining < k_left / max_den:
                return
            
            # Lower bound on d: 1/d < remaining (strict, since we need room for others)
            # Actually: 1/d <= remaining - (k_left-1)/max_den
            # And 1/d < remaining (since other terms are positive)
            d_low = max(min_d, int(1 / float(remaining)) + 1)
            
            # Upper bound: after picking d, need k_left-1 terms from d+1 to max_den
            # to sum to remaining - 1/d
            # Max sum of k_left-1 terms from d+1: (k_left-1)/(d+1)
            # So: remaining - 1/d <= (k_left-1)/(d+1)
            # This gives a constraint on d
            # Also: remaining - 1/d >= (k_left-1)/max_den
            # => 1/d <= remaining - (k_left-1)/max_den
            # => d >= 1/(remaining - (k_left-1)/max_den)
            
            rem_after_min = remaining - (k_left - 1) / max_den
            if rem_after_min <= 0:
                return
            d_high_from_min = int(1 / float(rem_after_min))
            
            # Also from the max sum constraint:
            # remaining - 1/d <= (k_left-1)/(d+1)
            # Let's solve for d:
            # remaining * d * (d+1) - (d+1) <= (k_left-1) * d
            # This is a quadratic, but let's just use a reasonable upper bound
            # d should not be too large, otherwise remaining - 1/d is too small
            # For k_left terms averaging remaining/k_left, d ~ k_left/remaining
            avg_d = k_left / float(remaining)
            d_high = min(int(avg_d * 3) + 100, max_den, d_high_from_min)
            
            if d_low > d_high:
                return
            
            # Generate candidates: focus on values that make the remainder "nice"
            # i.e., remaining - 1/d has a denominator that can be achieved with k_left-1 terms
            candidates = set()
            
            # Add a range around the expected value
            center = int(k_left / float(remaining))
            for offset in range(-50, 51):
                d = center + offset
                if d_low <= d <= d_high:
                    candidates.add(d)
            
            # Also add d_low and d_high
            candidates.add(d_low)
            candidates.add(d_high)
            
            # Sort candidates: try values that might lead to integer solutions first
            # Heuristic: prefer d where (remaining - 1/d) has a "simple" denominator
            def candidate_score(d):
                rem = remaining - Fraction(1, d)
                if rem <= 0:
                    return float('inf')
                # Prefer smaller denominators (easier to achieve with few terms)
                return rem.denominator
            
            sorted_candidates = sorted(candidates, key=candidate_score)
            
            for d in sorted_candidates:
                if d < min_d or d > max_den:
                    continue
                if d in current:
                    continue
                term = Fraction(1, d)
                if term >= remaining:
                    continue
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                current.append(d)
                backtrack(new_remaining, d + 1, current, k_left - 1)
                current.pop()
                if best[0] is not None:
                    return
        
        backtrack(Fraction(1, 1), min_den, [], k)
        return best[0]
    
    # Try k from 11 down to 5
    for k in range(11, 4, -1):
        if time.time() > deadline - 10:
            break
        sol = try_find_k_terms(k)
        if sol is not None:
            # Verify
            s = compute_sum(sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    return fallback
