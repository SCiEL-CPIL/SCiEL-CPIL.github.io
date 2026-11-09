import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe
    
    # Fallback solution from seed logic (DP on 2520)
    # This ensures we always return a valid result
    def get_fallback():
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
        return [5, 6, 7, 42, 140, 420, 840, 2520] # Last resort
    
    best_solution = get_fallback()
    
    # DFS to find a shorter solution
    target = Fraction(1)
    max_den = 10000000
    min_den = 5
    
    # We want to minimize count, so we search for solutions with small count
    # We use a recursive search with pruning
    
    def search(rem, last_d, path):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if rem == 0:
            if len(path) < len(best_solution):
                best_solution = path[:]
            return
            
        if len(path) >= len(best_solution):
            return
            
        # Pruning: if rem is too small to be covered by remaining terms
        # Max remaining terms = len(best_solution) - len(path)
        # Min possible value of next term is 1/max_den
        # This pruning is weak, rely on denominator bound
        
        # Next denominator d must satisfy:
        # 1. d > last_d
        # 2. 1/d <= rem  => d >= 1/rem
        # 3. d <= max_den
        
        # If rem is very small, d must be very large
        # If d > max_den, we can't add any more terms, so if rem != 0, fail
        min_d = max(last_d + 1, int(1 / rem) + 1 if rem > 0 else max_den + 1)
        
        if min_d > max_den:
            return
            
        # Optimization: If we have few terms left, we can bound d more tightly
        # For the last term, d must be exactly 1/rem
        # For the second to last, etc.
        
        # To keep search space small, we don't iterate all d up to max_den
        # We iterate d from min_d to some limit
        # A good heuristic: if rem is large, d is small. If rem is small, d is large.
        # We can limit the number of candidates for d.
        
        # Let's try a range of d values.
        # If rem is close to 1/k, d should be close to k.
        
        # Iterate d
        # We cannot iterate to 10^7.
        # We only need to check d such that 1/d <= rem.
        # Also, if we have `k` terms left, the sum of reciprocals of `k` distinct integers > last_d
        # must be at least rem.
        
        # Simple iteration with early exit
        # We limit the search depth and branching
        # For each d, we check if it's a valid candidate
        
        # To avoid infinite loops or huge ranges, we step through d.
        # However, the range [min_d, max_den] is huge.
        # We can use the property that d must be a divisor of something? No.
        
        # Heuristic: Try d in [min_d, min_d + 1000] and also d near 1/rem.
        # Actually, if we just pick the smallest valid d, we follow the greedy path.
        # Greedy path might fail due to max_den.
        # So we need to branch slightly.
        
        # Let's try a limited branch:
        # 1. The greedy choice d_greedy = ceil(1/rem)
        # 2. d_greedy + 1, d_greedy + 2, ... up to some limit
        
        d_greedy = int(1 / rem) + 1
        if d_greedy < min_d:
            d_greedy = min_d
            
        # Try a window of denominators around d_greedy
        # If the window is too large, we might miss solutions, but we have a fallback.
        # We also try a few larger d values to see if they lead to a shorter path (unlikely)
        
        # Actually, larger d means smaller 1/d, so rem decreases slower, requiring MORE terms.
        # So we want the SMALLEST valid d to minimize count.
        # So we should primarily explore the smallest d.
        
        # Let's try d from min_d to min_d + 20.
        # If none work, try d_greedy to d_greedy + 20.
        
        candidates = set()
        for d in range(min_d, min_d + 50):
            if d <= max_den:
                candidates.add(d)
        
        for d in range(d_greedy, d_greedy + 50):
            if d <= max_den and d > last_d:
                candidates.add(d)
                
        # Also add some larger d values just in case? No, larger d is worse for count.
        
        for d in sorted(candidates):
            if time.time() > deadline:
                return
            if d <= last_d:
                continue
            if d > max_den:
                break
            term = Fraction(1, d)
            if term > rem:
                break
            new_rem = rem - term
            path.append(d)
            search(new_rem, d, path)
            path.pop()

    search(target, 4, [])
    
    return best_solution
