from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    def get_fallback():
        """Get the known 12-term solution via subset DP."""
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
    
    # Get fallback quickly
    fallback = get_fallback()
    
    def try_find_solution(k, min_den=5, max_den=10000000):
        """Find k distinct denominators in [min_den, max_den] with reciprocal sum = 1."""
        best = [None]
        failed_states = set()
        
        def backtrack(remaining, min_d, current, k_left):
            if time.time() > deadline:
                return
            if k_left == 0:
                if remaining == 0:
                    best[0] = current[:]
                return
            
            if k_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d >= min_d and d <= max_den and d not in current:
                            best[0] = current + [d]
                return
            
            # Check if this state has been tried before
            state_key = (remaining, min_d, k_left)
            if state_key in failed_states:
                return
            
            # Pruning bounds
            if remaining > k_left / min_d:
                failed_states.add(state_key)
                return
            if remaining < k_left / max_den:
                failed_states.add(state_key)
                return
            
            # Tighter lower bound: the next denominator must be at least
            # such that we can still reach the remaining sum with k_left terms
            d_low = max(min_d, int(1 / float(remaining)) + 1)
            if d_low < min_d:
                d_low = min_d
            
            # Upper bound: the next denominator cannot be so large that
            # even with k_left-1 terms of size 1/max_den we can't reach remaining
            if remaining - (k_left - 1) / max_den <= 0:
                failed_states.add(state_key)
                return
            d_high = int(1 / (remaining - (k_left - 1) / max_den))
            d_high = min(d_high, max_den)
            
            if d_low > d_high:
                failed_states.add(state_key)
                return
            
            # Also need: sum of k_left terms starting from d_low must be >= remaining
            # 1/d_low + 1/(d_low+1) + ... + 1/(d_low+k_left-1) >= remaining
            # This is hard to compute exactly, so use a loose bound
            if remaining > k_left / d_low:
                failed_states.add(state_key)
                return
            
            # Try candidates: focus on a range around the "expected" denominator
            avg_d = k_left / float(remaining)
            
            # Generate candidates in a smart order
            # Start with denominators near the average, then expand
            search_center = int(avg_d)
            candidates = []
            
            # Add candidates from center outward
            for offset in range(0, 200):
                d1 = search_center + offset
                d2 = search_center - offset
                if d1 >= d_low and d1 <= d_high:
                    candidates.append(d1)
                if d2 >= d_low and d2 <= d_high and d2 != d1:
                    candidates.append(d2)
            
            # Also add boundary candidates
            candidates.append(d_low)
            candidates.append(d_high)
            
            # Remove duplicates and sort
            candidates = sorted(set(candidates))
            
            for d in candidates:
                if d < min_d or d > max_den or d in current:
                    continue
                term = Fraction(1, d)
                if term >= remaining:
                    continue
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                
                # Additional pruning: check if new_remaining is achievable
                if new_remaining > (k_left - 1) / (d + 1):
                    continue
                if new_remaining < (k_left - 1) / max_den:
                    continue
                
                current.append(d)
                backtrack(new_remaining, d + 1, current, k_left - 1)
                current.pop()
                if best[0] is not None:
                    return
            
            failed_states.add(state_key)
        
        backtrack(Fraction(1, 1), min_den, [], k)
        return best[0]
    
    # Try to find solutions with fewer terms
    for k in range(11, 4, -1):
        if time.time() > deadline - 5:
            break
        sol = try_find_solution(k)
        if sol is not None:
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    return fallback
