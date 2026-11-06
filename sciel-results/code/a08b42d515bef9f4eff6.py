"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 28.0  # stay safely under 30 s
    
    def is_sidon(S):
        seen = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    def get_addable(S):
        Sset = set(S)
        sums = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            sums.add(2 * a)
            for b in L[i + 1:]:
                sums.add(a + b)
        
        res = []
        for x in range(1, N + 1):
            if x in Sset:
                continue
            # Check if x can be added
            # 1. 2x not in sums
            if 2 * x in sums:
                continue
            # 2. x+s not in sums for any s in S
            # 3. x+s not equal to 2x (impossible if s != x)
            # Actually, just check if adding x creates a duplicate sum.
            # New sums: 2x, and x+s for s in S.
            # These must not be in `sums` and must be distinct from each other.
            new_sums = {2 * x}
            for s in S:
                new_sums.add(x + s)
            
            # Check intersection with existing sums
            if new_sums & sums:
                continue
            
            # Check internal distinctness of new sums
            # 2x == x+s => x=s (impossible as x not in S)
            # x+s1 == x+s2 => s1=s2 (impossible)
            # So new_sums are always distinct internally.
            # We just need to ensure they don't collide with existing sums.
            
            res.append(x)
        return res

    def make_maximal(S):
        """Greedily add elements to S until it is maximal."""
        S = list(S)
        while True:
            addable = get_addable(S)
            if not addable:
                break
            # Pick the element that blocks the most? 
            # For now, just pick the first or random to vary.
            # To minimize size, we should pick the one that "covers" the most potential additions?
            # Actually, any choice leads to a maximal set. 
            # The size depends on the path. 
            # Let's pick the one with the smallest value? Or random?
            # Random is better for local search diversity.
            x = random.choice(addable)
            S.append(x)
        return S

    def generate_random_sidon():
        """Generate a random maximal Sidon set."""
        S = []
        # Randomly shuffle elements
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        for x in elems:
            # Try to add x
            if is_sidon(S + [x]):
                S.append(x)
        # Now make it maximal
        S = make_maximal(S)
        return S

    best_set = generate_random_sidon()
    best_size = len(best_set)
    
    # Local Search / Hill Climbing
    while time.time() < deadline:
        # Pick a random current solution or the best one?
        # Let's try to improve the best one.
        current = best_set[:]
        
        # Try to remove a random element
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        s_to_remove = current[idx]
        candidate = current[:idx] + current[idx+1:]
        
        # Check if candidate is still Sidon
        if not is_sidon(candidate):
            continue
            
        # Now candidate is Sidon but not necessarily maximal.
        # We need to make it maximal.
        # If it's already maximal, we improved!
        addable = get_addable(candidate)
        if not addable:
            # We removed an element and it's still maximal. 
            # This is a strict improvement in size.
            best_set = candidate
            best_size = len(best_set)
            continue
        
        # It's not maximal. We must add elements.
        # We want to add as few as possible.
        # This is a set cover problem: Cover all addable elements with the fewest additions.
        # Greedy: Pick the element that blocks the most other addable elements.
        
        def get_blocking_power(x, addable_set):
            """How many elements in addable_set are blocked by adding x?"""
            # We simulate adding x to the current set (which is 'candidate')
            # and see which other addable elements become non-addable.
            # This is expensive to compute for all x.
            # Approximation: Just pick the one with smallest index?
            # Or random?
            # Let's try a simple heuristic: pick the one that minimizes the remaining addable set size.
            # Since we don't know the cost, let's just pick one randomly or the first one.
            # To be smarter, we can estimate.
            return 0 # Placeholder
            
        # For efficiency, let's just pick a random addable element and add it,
        # then repeat. This is a "Greedy Repair".
        # To improve chances, we can try a few different repair strategies.
        
        repaired = candidate[:]
        while True:
            addable = get_addable(repaired)
            if not addable:
                break
            # Heuristic: Choose the element that seems "most constrained" or "blocks most"?
            # Let's just pick the smallest one for determinism in this step, 
            # or random. Random is better for exploration.
            x = random.choice(addable)
            repaired.append(x)
            
        if len(repaired) < best_size:
            best_set = repaired
            best_size = len(best_set)
            
        # If we didn't improve, we can still keep 'repaired' as the current state for next iteration?
        # Or stick to best? Sticking to best is safer (Hill Climbing).
        # But sometimes we need to move sideways.
        # Let's stick to best for now to ensure monotonic improvement in size.

    # Final verification
    if not is_sidon(best_set):
        # Fallback
        S_fallback = []
        for x in range(1, N + 1):
            if is_sidon(S_fallback + [x]):
                S_fallback.append(x)
        best_set = make_maximal(S_fallback)
        
    return best_set
# EVOLVE-BLOCK-END
