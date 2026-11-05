"""Rewritten program: Repair+Backfill with O(N^2) residue checks and deadline guard."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _is_valid_set(S):
    """Check if set S satisfies the condition: no element divides sum of any 3 distinct others."""
    S_list = sorted(S)
    n = len(S_list)
    if n < 4:
        return True
    for i, a in enumerate(S_list):
        # Check if a divides sum of any 3 distinct OTHER elements
        # For efficiency, we can break early
        others = [x for x in S_list if x != a]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _violation_count(S):
    """Count total violations in set S."""
    S_list = sorted(S)
    n = len(S_list)
    if n < 4:
        return 0
    count = 0
    for i, a in enumerate(S_list):
        others = [x for x in S_list if x != a]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                count += 1
    return count

def _can_add(S_set, x):
    """Check if adding x to S_set maintains validity. S_set is a set of ints."""
    S_list = sorted(S_set | {x})
    n = len(S_list)
    if n < 4:
        return True
    
    # Check new element x against all triples of others
    others_for_x = [v for v in S_list if v != x]
    if len(others_for_x) >= 3:
        for t in combinations(others_for_x, 3):
            if sum(t) % x == 0:
                return False
    
    # Check existing elements against triples that include x
    # For each a in S_set, check if a divides sum of any 3 distinct others in S_list
    # The triples that could cause NEW violations are those that include x
    # For a given a (a != x), a triple containing x and two others b, c (b,c != a, b,c != x)
    # sum = x + b + c. Check if (x+b+c) % a == 0
    # Also triples not containing x were already valid (since S_set was valid)
    for a in S_set:
        if a == x:
            continue
        # Find pairs (b, c) from S_set \ {a, x} such that (x + b + c) % a == 0
        # Also need to consider triples with x and two others where one of the others might be... 
        # Actually, the others in the triple are from S_list \ {a}. 
        # The new triples are those that include x. So the triple is {x, b, c} where b,c in S_set \ {a, x}
        # and b != c.
        # Also, we need to check if a divides x + b + c for any distinct b, c in S_set \ {a, x}
        candidates = [v for v in S_set if v != a and v != x]
        if len(candidates) < 2:
            continue
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                b = candidates[i]
                c = candidates[j]
                if (x + b + c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Strategy: Start with a dense set from the upper half, then repair and backfill.
    # The key insight is that larger numbers are less likely to divide sums of smaller numbers
    # in a problematic way, or rather, the sums of three numbers are bounded, so large divisors
    # are less likely to divide those sums.
    
    # Phase 1: Greedy construction from top down, but with faster check
    # We'll use a set for O(1) membership and sorted list for iteration.
    
    # Let's try a different approach: Local Search / Hill Climbing
    # Start with a random valid set, then try to add elements or swap elements to increase size.
    
    # First, let's get a baseline valid set using greedy from top down with efficient check.
    # But the original greedy is slow. Let's use the _can_add function which is O(n^2) per addition.
    
    selected = set()
    # Try to add elements from 192 down to 1
    for x in range(192, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        if _can_add(selected, x):
            selected.add(x)
    
    # Now we have a valid set. Let's try to improve it by swapping or adding more.
    # But first, let's see how many we have.
    # The greedy from top down might get stuck. Let's try a randomized restart approach.
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Try multiple random orders to find a better set
    import random
    random.seed(42)
    
    for iteration in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random order of candidates
        candidates = list(range(192, 0, -1))
        random.shuffle(candidates)
        
        current_set = set()
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if _can_add(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Local search: try to add elements that were skipped
    # For each element not in best_set, try adding it and removing some elements to maintain validity
    # This is complex. Let's try a simpler repair: 
    # Start with a large set (e.g., all numbers > 96), find violations, remove elements to fix.
    
    # Phase 2: Dense start and repair
    # Start with all numbers in [97, 192]
    dense_set = set(range(97, 193))
    
    # Check if valid, if not, remove elements that cause violations
    # We'll iterate: find an element a that divides sum of 3 others, remove a or one of the others
    # To maximize size, we should remove the element that is involved in the most violations
    # or the smallest element (since small elements are more likely to divide sums)
    
    # Let's implement a repair loop
    repaired_set = set(dense_set)
    
    for _ in range(1000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Find a violation
        S_list = sorted(repaired_set)
        n = len(S_list)
        if n < 4:
            break
            
        violation_found = False
        for i, a in enumerate(S_list):
            others = [x for x in S_list if x != a]
            if len(others) < 3:
                continue
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    # Violation found: a divides sum(t)
                    # Remove a (or one of t). Removing a is simpler.
                    # But maybe removing a smaller element from t is better?
                    # Let's remove a for now, as it's the divisor.
                    repaired_set.remove(a)
                    violation_found = True
                    break
            if violation_found:
                break
        
        if not violation_found:
            break
    
    if len(repaired_set) > best_size:
        best_size = len(repaired_set)
        best_set = set(repaired_set)
    
    # Phase 3: Try to add back elements that were removed
    # Try to add each element from 1..192 not in best_set
    for x in range(1, 193):
        if time.time() - start_time > DEADLINE:
            break
        if x not in best_set:
            if _can_add(best_set, x):
                best_set.add(x)
                best_size = len(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
