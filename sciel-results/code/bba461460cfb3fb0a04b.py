"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _check_triple(a, b, c):
    """Check if 1/a = 1/b + 1/c for distinct a, b, c."""
    if a == b or a == c or b == c:
        return False
    # 1/a = 1/b + 1/c = (b+c)/(bc)
    # So bc = a(b+c) => bc - ab - ac = 0 => bc = a(b+c)
    # Check: 1/a - 1/b - 1/c = 0
    # bc - ac - ab = 0
    return b * c == a * (b + c)

def _find_violations(S):
    """Find all triples (a, b, c) in S that violate the condition."""
    S_set = set(S)
    violations = []
    for a in S:
        for b in S:
            if b == a:
                continue
            # c = a*b / (a - b) if a != b
            # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab)
            # c = ab / (b-a)
            denom = b - a
            if denom == 0:
                continue
            num = a * b
            if num % denom != 0:
                continue
            c = num // denom
            if c > 0 and c != a and c != b and c in S_set:
                # Ensure we don't count same triple multiple times
                violations.append((a, b, c))
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s
    
    N = 180
    # Start with the upper half which is known to be valid
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Greedy addition: try to add elements from lower half
    # Sort candidates by value (smaller elements are more likely to form triples)
    candidates = list(range(1, 91))
    random.Random(42).shuffle(candidates)
    
    # For each candidate, try to add it if no new violation is created
    for c in candidates:
        if time.time() > deadline - 10:
            break
        if c in best_set:
            continue
        # Check if adding c creates any violation
        new_set = best_set | {c}
        # Check triples involving c
        valid = True
        for a in best_set:
            if a == c:
                continue
            for b in best_set:
                if b == c or b == a:
                    continue
                if _check_triple(a, b, c):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            best_set = new_set
            best_size = len(best_set)
    
    # Local search: try swapping elements
    # Try to replace a high element with a lower one if it reduces conflicts
    # or maintain size while enabling more additions
    
    # Try adding more elements with different random orderings
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 91))
        rng.shuffle(candidates)
        
        current_set = set(best_set)
        
        for c in candidates:
            if time.time() > deadline - 5:
                break
            if c in current_set:
                continue
            # Check if adding c is valid
            valid = True
            for a in current_set:
                if a == c:
                    continue
                for b in current_set:
                    if b == c or b == a:
                        continue
                    if _check_triple(a, b, c):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_set = current_set | {c}
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
