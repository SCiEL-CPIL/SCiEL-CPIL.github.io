"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    import random
    
    def is_sidon(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                s = selected[i] + selected[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(selected):
        if not selected:
            return False
        sums = set()
        sel_set = set(selected)
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                sums.add(selected[i] + selected[j])
        
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # Check if x can be added to selected
            can_add = True
            # New sums will be x+a for a in selected, and 2*x
            # We need to check if any of these are in 'sums'
            # Or if any of the new sums collide with each other
            # Since x is not in selected, x+a1 != x+a2 if a1 != a2.
            # So we just need to check intersection with existing sums.
            # Also check if 2*x collides with existing sums.
            # And check if x+a collides with 2*x? No, x+a = 2x => a=x, but x not in selected.
            # So we just check:
            # 1. 2*x in sums
            # 2. Any (x+a) in sums
            
            if 2 * x in sums:
                continue # Collision, cannot add
            
            conflict = False
            for a in selected:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict:
                return False # Can add x, so not maximal
        
        return True

    def generate_greedy(seed):
        random.seed(seed)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        selected = []
        sums = set()
        for x in candidates:
            added = [x + a for a in selected] + [2 * x]
            # Check for internal collision in added
            # x+a1 == x+a2 implies a1==a2, so no internal collision among x+a's
            # 2*x == x+a implies a=x, but x not in selected yet.
            # So added elements are distinct from each other.
            # Just check intersection with existing sums
            if not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    def prune(selected):
        # Try to remove elements while maintaining maximality
        changed = True
        while changed:
            changed = False
            # Try removing each element
            # Randomize order to avoid bias? Or just iterate.
            # We need to be careful: removing one might make another removable.
            for i in range(len(selected)):
                x = selected[i]
                candidate = selected[:i] + selected[i+1:]
                if is_maximal(candidate):
                    selected = candidate
                    changed = True
                    break # Restart loop with new set
        return selected

    best = None
    best_size = float('inf')
    
    # Try multiple seeds
    for seed in range(200):
        s = generate_greedy(seed)
        # Ensure it's maximal (greedy usually produces maximal if we check properly, 
        # but the greedy above only checks Sidon property. 
        # Wait, the greedy adds x if it doesn't conflict. 
        # Does it guarantee maximality?
        # If we add all possible x in a random order, the final set is maximal 
        # with respect to that order? No.
        # If we skip x because it conflicts, it might not conflict later if we add other elements? 
        # No, adding more elements only adds more sums, making it harder to add x.
        # So if x couldn't be added at step k, it can't be added later.
        # Thus, the set of all elements that *could* be added is exactly the set we built.
        # Any element not in the set was rejected because it conflicted with the set at that time.
        # Since the set only grows, the conflicts persist.
        # So the resulting set IS maximal.
        
        s = prune(s)
        
        if len(s) < best_size:
            best_size = len(s)
            best = s
            # Optional: break if we reach a theoretical lower bound?
            # Lower bound for maximal Sidon set size is roughly sqrt(N)?
            # For N=256, sqrt(256)=16. But small maximal sets can be smaller?
            # Actually, maximal Sidon sets must be "dense" enough to block all others.
            # A known result: maximal Sidon sets in [1,N] have size at least c * sqrt(N).
            # For N=256, we expect sizes around 10-15.
            if best_size <= 12: # Heuristic cutoff
                break

    if best is None:
        # Fallback
        best = generate_greedy(0)
        
    return best
# EVOLVE-BLOCK-END
