"""Combined program: fast integer LCM-based dissociation check + backtracking search."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _lcm_list(xs):
    L = 1
    for x in xs:
        L = math.lcm(L, x)
    return L

def _is_dissociated_int(S):
    """Fast check: all subset sums of {1/x for x in S} distinct, via integer LCM."""
    if not S:
        return True
    L = _lcm_list(S)
    vals = [L // x for x in S]
    sums = {0}
    for v in vals:
        # new sums = sums + v
        new = {s + v for s in sums}
        # Check if any new sum collides with an existing sum
        if new & sums:
            return False
        sums |= new
    return True

def _can_add(S, x):
    """Check if S ∪ {x} is dissociated, incrementally (fast path)."""
    if x in S:
        return False
    S2 = list(S) + [x]
    L = _lcm_list(S2)
    vals = [L // s for s in S2]
    sums = {0}
    for v in vals:
        new = {s + v for s in sums}
        if new & sums:
            return False
        sums |= new
    return True

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    universe = list(range(1, 23))
    
    # Start with an empty best set to ensure validity
    # We will find a valid set via search.
    # Fallback valid set: [1] is always valid.
    best = []
    
    # --- Phase 1: Backtracking search for maximum cardinality ---
    # We want the largest dissociated subset of [1,22].
    # Use backtracking: try to include/exclude each element.
    # Order elements to find large sets quickly.
    
    # Strategy: search over subsets, maximizing size.
    # Since 2^22 is too big, use a smart backtracking with pruning.
    
    # Heuristic ordering: try to include elements that are "easy" to add.
    # Large numbers have small reciprocals, so they're less likely to cause collisions.
    # But small numbers (1, 2, 3) are powerful and often needed.
    
    # Let's do a systematic backtracking:
    # Sort candidates. At each step, try adding the next candidate.
    # Prune: if remaining candidates + current size <= best size, stop.
    
    # Precompute: for each element, its reciprocal as a fraction numerator/denominator
    # Actually, let's just use the integer LCM approach.
    
    # Backtracking search
    def backtrack(current, remaining, current_size):
        nonlocal best
        if time.time() > deadline - 2:
            return
        
        # Prune: even if we add all remaining, can we beat best?
        if current_size + len(remaining) <= len(best):
            return
        
        if not remaining:
            if current_size > len(best):
                best = list(current)
            return
        
        # Try to add remaining[0]
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include x
        if _can_add(current, x):
            current.append(x)
            backtrack(current, rest, current_size + 1)
            current.pop()
        
        # Option 2: exclude x
        backtrack(current, rest, current_size)
    
    # Start with empty set, try all elements
    # To find large sets fast, order elements so that "easy" ones come first
    # Heuristic: large numbers first (smaller reciprocals, fewer collisions)
    # But we also want small numbers for coverage...
    # Let's try multiple orderings.
    
    # Ordering 1: ascending (small first)
    order1 = list(range(1, 23))
    backtrack([], order1, 0)
    
    if time.time() > deadline - 10:
        # Not enough time, return best
        if _is_dissociated_int(best):
            return sorted(best)
        return sorted(base)
    
    # Ordering 2: descending (large first)
    order2 = list(range(22, 0, -1))
    backtrack([], order2, 0)
    
    if time.time() > deadline - 10:
        if _is_dissociated_int(best):
            return sorted(best)
        return sorted(base)
    
    # Ordering 3: primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    primes = [x for x in range(1, 23) if is_prime(x)]
    composites = [x for x in range(1, 23) if not is_prime(x)]
    order3 = primes + composites
    backtrack([], order3, 0)
    
    if time.time() > deadline - 10:
        if _is_dissociated_int(best):
            return sorted(best)
        return sorted(base)
    
    # --- Phase 2: Randomized greedy with local search ---
    # Try random orderings with greedy construction
    random.seed(42)
    
    def greedy_construct(order):
        current = []
        for x in order:
            if time.time() > deadline - 2:
                break
            if _can_add(current, x):
                current.append(x)
        return current
    
    # Try several random orderings
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        result = greedy_construct(order)
        if len(result) > len(best):
            best = result
    
    # Local search: try to swap elements to increase size
    for _ in range(200):
        if time.time() > deadline - 3:
            break
        improved = False
        # Try adding any element not in best
        for x in range(1, 23):
            if x in best:
                continue
            if _can_add(best, x):
                best.append(x)
                improved = True
                break
        if not improved:
            # Try swapping: remove one, add one
            for i in range(len(best)):
                if time.time() > deadline - 3:
                    break
                for x in range(1, 23):
                    if x in best:
                        continue
                    candidate = [v for v in best if v != best[i]] + [x]
                    if _can_add(candidate, x) or _is_dissociated_int(candidate):
                        # Actually _can_add checks if we can add x to a set,
                        # but here we're replacing, so use _is_dissociated_int
                        if _is_dissociated_int(candidate):
                            best = candidate
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    # Final verification
    # If best is empty or invalid, return a known valid non-empty set
    if not best or not _is_dissociated_int(best):
        # [1, 2] is a valid dissociated set
        # Let's verify [1, 2] quickly in code to be safe, or just use [1]
        # [1] is definitely valid.
        return [1]
    
    return sorted(best)
# EVOLVE-BLOCK-END
