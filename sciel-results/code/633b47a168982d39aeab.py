"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # Sort descending for greedy-like search
    
    # Filter divisors <= target (can't use any divisor > target)
    divisors = [d for d in divisors if d <= target]
    
    # Try increasing counts from 1 upward
    for count in range(1, len(divisors) + 1):
        if time.time() > deadline:
            break
        
        # For small counts, use combinations
        if count <= 5:
            found = None
            # Use recursive search with pruning for efficiency
            def search(idx, remaining, selected):
                if time.time() > deadline:
                    return None
                if remaining == 0:
                    return selected[:]
                if idx >= len(divisors):
                    return None
                if remaining < 0:
                    return None
                # Pruning: if we can't fill remaining with available divisors
                max_possible = sum(divisors[idx:min(idx + len(divisors) - len(selected), len(divisors))])
                if max_possible < remaining:
                    return None
                
                # Try including divisors[idx]
                if divisors[idx] <= remaining:
                    selected.append(divisors[idx])
                    result = search(idx + 1, remaining - divisors[idx], selected)
                    if result is not None:
                        return result
                    selected.pop()
                
                # Try excluding divisors[idx]
                result = search(idx + 1, remaining, selected)
                if result is not None:
                    return result
                
                return None
            
            found = search(0, target, [])
            if found is not None:
                return found
        
        elif count == 6:
            # For count 6, use a different approach
            # Try combinations of 6 from the larger divisors
            # But this could be too many, so use pruning
            found = None
            
            def search6(idx, remaining, selected, depth):
                if time.time() > deadline:
                    return None
                if remaining == 0:
                    return selected[:]
                if depth == count:
                    return None
                if idx >= len(divisors):
                    return None
                if remaining < 0:
                    return None
                
                # Pruning
                if len(divisors) - idx < count - depth:
                    return None
                max_possible = sum(divisors[idx:min(idx + count - depth, len(divisors))])
                if max_possible < remaining:
                    return None
                
                # Try including
                if divisors[idx] <= remaining:
                    selected.append(divisors[idx])
                    result = search6(idx + 1, remaining - divisors[idx], selected, depth + 1)
                    if result is not None:
                        return result
                    selected.pop()
                
                # Try excluding
                result = search6(idx + 1, remaining, selected, depth + 1)
                if result is not None:
                    return result
                
                return None
            
            found = search6(0, target, [], 0)
            if found is not None:
                return found
    
    # Fallback: greedy solution (should always work)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0, f"Greedy failed: remainder={remainder}"
    return chosen

# EVOLVE-BLOCK-END
