"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _find_violations(selected):
    """
    Find all violations in the selected set.
    Returns a list of tuples (a, b, c, d) where a divides b+c+d.
    Also returns a count of how many violations each element is involved in.
    """
    n = len(selected)
    if n < 4:
        return [], {}
    
    # Index map for quick lookup? Not needed for n=49.
    # selected is a list of ints.
    
    violations = []
    # counts[x] = number of violations x is part of
    counts = {x: 0 for x in selected}
    
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            s = t[0] + t[1] + t[2]
            if s % a == 0:
                violations.append((a, t[0], t[1], t[2]))
                counts[a] += 1
                counts[t[0]] += 1
                counts[t[1]] += 1
                counts[t[2]] += 1
                
    return violations, counts

def _greedy_removal(initial_set, time_limit=10.0):
    """
    Start with initial_set and greedily remove elements to satisfy constraints.
    """
    selected = list(initial_set)
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        violations, counts = _find_violations(selected)
        
        if not violations:
            return selected
            
        # Find element with max count
        max_count = max(counts.values())
        # Pick one of the elements with max count.
        # If tie, pick the largest? Or smallest? 
        # Largest numbers are "safer" divisors, so maybe prefer removing smaller ones?
        # Or remove the one that is most "central".
        # Let's pick the one with max count. If tie, pick the largest value (less likely to be a divisor of sums? No, largest is less likely to divide. Smallest is more likely to divide. So removing a small divisor might break many things. Removing a large summand might break many things.)
        # Let's just pick the first one with max count.
        candidates = [x for x in selected if counts[x] == max_count]
        # Heuristic: Remove the smallest one? Or largest?
        # If we remove a divisor 'a', we fix all violations where 'a' is the divisor.
        # If we remove a summand 'b', we fix all violations where 'b' is a summand.
        # Usually divisors are the "bottleneck".
        # Let's try removing the smallest candidate to keep the "strong" large numbers.
        # Actually, the knowledge says "Greedy removal...". It doesn't specify which.
        # I will try removing the one with the highest count. If tie, smallest value.
        to_remove = min(candidates)
        selected.remove(to_remove)
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Strategy: Greedy removal from [144, 192]
    # This range has 49 elements.
    initial_pool = list(range(144, 193))
    
    best = _greedy_removal(initial_pool, time_limit=5.0)
    
    # If we didn't get 48, try other pools or local search
    if len(best) < 48 and time.time() < deadline:
        # Try slightly wider pool
        for start_val in [143, 142, 141, 140]:
            if time.time() >= deadline:
                break
            pool = list(range(start_val, 193))
            # We can limit the removal process to keep it fast
            candidate = _greedy_removal(pool, time_limit=5.0)
            if len(candidate) > len(best):
                best = candidate
                
    # Final check: is best valid?
    # The greedy removal ensures validity at the end.
    # But let's double check.
    violations, _ = _find_violations(best)
    if violations:
        # Should not happen if _greedy_removal worked correctly
        # Fallback to a simple valid set
        # Just return a small valid set to avoid 0 score
        best = [192, 191, 190, 189] # Check if valid?
        # 192 | 191+190+189 = 570? 570/192 no.
        # 191 | 570? 570/191 no.
        # 190 | 570? 570/190 = 2.94 no.
        # 189 | 570? 570/189 no.
        # So [189,190,191,192] is valid.
        
    return best

# EVOLVE-BLOCK-END
