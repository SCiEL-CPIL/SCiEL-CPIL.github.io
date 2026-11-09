"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    
    # Build conflict sets: conflict[a] = set of b != a in [1,N] such that (a+b) | a*b
    # Note: (a+b) | a*b  <=>  (a+b) | a*(a+b-a) = a*(a+b) - a*a  <=>  (a+b) | a^2
    # So b = d - a where d | a^2 and d > a (since b >= 1)
    
    conflict = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find all divisors d of a_sq where d > a
        # b = d - a, need 1 <= b <= N, b != a
        for d in range(a + 1, min(a_sq, a + N) + 1):
            if a_sq % d == 0:
                b = d - a
                if 1 <= b <= N and b != a:
                    conflict[a].add(b)
                    conflict[b].add(a)
    
    # All odd numbers never conflict with each other
    # (a+b) is even for odd a,b; a*b is odd; even | odd is false
    # So all odds form an independent set of size 128
    
    odds = list(range(1, N + 1, 2))
    
    # For evens, we need to find which can be added to the odds set
    # An even e conflicts with an odd o if (e+o) | e*o
    # Since e+o is odd, and e*o... let's just use the precomputed conflict sets
    
    # Strategy: start with all odds, then greedily add evens that don't conflict with any selected element
    # Multiple random greedy trials + local search
    
    best_set = set(odds)
    best_size = len(best_set)
    
    # Precompute for each even which odds it conflicts with
    even_candidates = list(range(2, N + 1, 2))
    
    def get_compatible_evens(selected):
        """Return evens that don't conflict with any element in selected."""
        compatible = []
        for e in even_candidates:
            ok = True
            for s in selected:
                if e in conflict[s]:
                    ok = False
                    break
            if ok:
                compatible.append(e)
        return compatible
    
    # Greedy: add evens one by one, each time choosing the one with fewest conflicts with remaining candidates
    def greedy_add_evens(initial_set, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        selected = set(initial_set)
        remaining_evens = list(even_candidates)
        
        # Shuffle for randomness
        rng.shuffle(remaining_evens)
        
        for e in remaining_evens:
            # Check if e is compatible with all currently selected
            ok = True
            for s in selected:
                if e in conflict[s]:
                    ok = False
                    break
            if ok:
                selected.add(e)
        
        return selected
    
    # Local search: try to swap out an element for a better one
    def local_search(initial_set, max_iters=500, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        selected = set(initial_set)
        size = len(selected)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Try to add a random unselected element
            unselected = [x for x in range(1, N + 1) if x not in selected]
            if not unselected:
                break
            
            # Pick a random unselected element
            e = rng.choice(unselected)
            
            # Check if e can be added
            conflicts_with_selected = [s for s in selected if e in conflict[s]]
            
            if not conflicts_with_selected:
                selected.add(e)
                size += 1
            else:
                # Try to remove one conflicting element and add e
                # Only beneficial if we can add e (size stays same or increases)
                # Since we're removing 1 and adding 1, size stays same
                # This is only useful if it enables future additions
                # For simplicity, skip swaps for now, just try additions
                pass
        
        return selected
    
    # Multi-random greedy
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        current = greedy_add_evens(odds, random_seed=trial)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Local search on best found
    if time.time() < deadline - 10:
        improved = local_search(best_set, max_iters=2000, random_seed=42)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_size)
    
    # Try more local search iterations with different seeds
    if time.time() < deadline - 5:
        for seed in range(10):
            if time.time() > deadline - 2:
                break
            improved = local_search(best_set, max_iters=500, random_seed=100 + seed)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
    
    return list(best_set)

# EVOLVE-BLOCK-END
