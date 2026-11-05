"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Breadth-first search over rational remainders
    # State: (numerator, denominator, list_of_denominators)
    # Start with remainder 1/1, subtract reciprocals of smallest available d >= 5
    
    # BFS: explore states level by level (each level = one more denominator)
    # State key: (num, den) where num/den is the remaining fraction
    # We want to reach (0, 1) with minimum number of denominators
    
    visited = {}  # (num, den) -> (depth, [denominators])
    queue = deque()
    
    # Initial state: remainder = 1/1, no denominators used yet
    queue.append((1, 1, []))
    visited[(1, 1)] = (0, [])
    
    max_depth = 256  # Maximum number of denominators allowed
    best_result = None
    
    while queue:
        if time.time() > deadline:
            break
        
        num, den, used_dens = queue.popleft()
        depth = len(used_dens)
        
        if depth >= max_depth:
            continue
        
        # If we've reached zero remainder, we found a solution
        if num == 0:
            if best_result is None or depth < len(best_result):
                best_result = list(used_dens)
                # BFS guarantees first solution at this depth is optimal
                break
        
        # The minimum denominator we can choose: 1/d <= num/den => d >= den/num
        min_d = (den + num - 1) // num if num > 0 else 5
        if min_d < 5:
            min_d = 5
        
        # Try denominators starting from min_d
        # To keep search tractable, limit the range of d we try
        # d must satisfy: 1/d <= num/den (so d >= den/num)
        # And d must not be in used_dens
        # Also d <= 10000000
        
        # Heuristic: only try a limited set of candidates to keep BFS tractable
        # Try d = min_d, min_d+1, ..., up to some bound
        # Also try some "nice" values
        
        candidates = []
        
        # Always try the minimum valid denominator
        d = min_d
        while d in used_dens:
            d += 1
        if d <= 10000000:
            candidates.append(d)
        
        # Try a few more values around min_d
        for offset in range(1, 20):
            d = min_d + offset
            if d > 10000000:
                break
            if d not in used_dens:
                candidates.append(d)
        
        # Also try some "round" numbers that might be useful
        for d in [6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 48, 60, 120, 240, 720]:
            if d >= min_d and d <= 10000000 and d not in used_dens and d not in candidates:
                candidates.append(d)
        
        for d in candidates:
            if d in used_dens:
                continue
            if d > 10000000:
                continue
            if d < 5:
                continue
            
            # New remainder: num/den - 1/d = (num*d - den)/(den*d)
            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                continue  # d too small
            
            # Reduce fraction
            g = math.gcd(new_num, new_den)
            if g > 0:
                new_num //= g
                new_den //= g
            
            state_key = (new_num, new_den)
            
            # Only visit each state once (at minimum depth)
            if state_key in visited:
                continue
            
            visited[state_key] = (depth + 1, used_dens + [d])
            queue.append((new_num, new_den, used_dens + [d]))
        
        # If we found a solution at this depth, BFS guarantees it's optimal
        if best_result is not None:
            break
    
    if best_result is not None and 1 <= len(best_result) <= 256:
        # Verify the result
        s = sum(Fraction(1, x) for x in best_result)
        if s == 1:
            return best_result
    
    # Fallback: use greedy algorithm
    num, den = 1, 1
    used = set()
    result = []
    
    while num > 0:
        if time.time() > deadline:
            break
        
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
        
        d = min_d
        while d in used:
            d += 1
        
        if d > 10000000:
            break
        
        result.append(d)
        used.add(d)
        
        num = num * d - den
        den = den * d
        
        g = math.gcd(num, den)
        num //= g
        den //= g
    
    if num != 0 or len(result) == 0 or len(result) > 256:
        # Last resort: hardcoded known solution
        # 1/5 + 1/6 + 1/7 + 1/42 = 1? Let's check:
        # 1/5 + 1/6 + 1/7 + 1/42 = (42+35+30+5)/210 = 112/210 = 8/15 != 1
        # Known Egyptian fraction for 1: 1/2+1/3+1/6 but 2,3 < 5
        # 1/5 + 1/6 + 1/7 + 1/43 + 1/1806? This is getting complex
        # Use a simpler known decomposition
        # 1 = 1/5 + 1/6 + 1/7 + 1/42 + ... let me just use greedy and verify
        
        # Try a different approach: use the identity 1/n = 1/(n+1) + 1/(n(n+1))
        # Starting from 1/5: 1/5 = 1/6 + 1/30
        # 1/30 = 1/31 + 1/930
        # etc. But this gives too many terms
        
        # Just return the greedy result if it's valid
        s = sum(Fraction(1, x) for x in result)
        if s == 1 and 1 <= len(result) <= 256:
            return result
    
    # Final fallback: return a known valid set
    # We need distinct denominators in [5, 10000000] summing to 1
    # 1/5 + 1/6 + 1/7 + 1/42 = 112/210 = 8/15
    # Let's use: 1 = 1/5 + 1/6 + 1/7 + 1/43 + 1/1806 (need to verify)
    # Actually, let's just ensure we return something valid
    
    # Simple valid solution: use the greedy result if it works
    if len(result) > 0:
        s = sum(Fraction(1, x) for x in result)
        if s == 1 and 1 <= len(result) <= 256:
            return result
    
    raise RuntimeError("no valid representation")
# EVOLVE-BLOCK-END
