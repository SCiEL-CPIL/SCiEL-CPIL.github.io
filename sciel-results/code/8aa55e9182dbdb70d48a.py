import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        
        # Check conflicts with existing sums
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict:
            if (2 * x) in sums:
                conflict = True
        
        if not conflict:
            # x can be added, so the set is not maximal.
            return False
        
    return True

def _greedy_construct(order, rng=None):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    """
    if order is None:
        order = list(range(1, N + 1))
        if rng:
            rng.shuffle(order)
        else:
            random.shuffle(order)
            
    selected = []
    sums = set()
    
    for x in order:
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected

def _low_degree_greedy(rng, deadline):
    """
    Structural change: Formulate as a minimum cardinality maximal independent set problem
    on a "Sidon compatibility graph". 
    Vertices are elements 1..N. 
    Two vertices are compatible if they can coexist in a Sidon set.
    We want a maximal independent set (Sidon set) that is minimal in size.
    
    Strategy:
    At each step, among all remaining candidates that are compatible with the current set,
    pick the one that "blocks" the most other candidates (i.e., has the lowest degree in the
    remaining compatibility subgraph). This aggressively prunes the search space and forces
    early maximality, often leading to smaller sets than standard greedy which picks
    arbitrary or high-degree nodes.
    """
    selected = []
    used_sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() > deadline:
            break
            
        # Find valid candidates: those that can be added to 'selected'
        valid_candidates = []
        for x in remaining:
            # Check if x is compatible with selected
            # Sums: 2x, x+s for s in selected
            valid = True
            if 2 * x in used_sums:
                valid = False
            else:
                for s in selected:
                    if (x + s) in used_sums:
                        valid = False
                        break
            if valid:
                valid_candidates.append(x)
                
        if not valid_candidates:
            # No valid candidates, but we need maximality.
            # If we are stuck, the set might not be maximal if there are elements in remaining
            # that are incompatible with each other but compatible with selected?
            # Actually, if no candidate is compatible with selected, then selected is maximal
            # with respect to 'remaining'. But we must ensure no element in remaining can be added.
            # The loop condition is that we pick from remaining. If valid_candidates is empty,
            # it means no element in remaining can be added. Thus, the set is maximal.
            break
            
        # For each valid candidate, calculate how many other valid candidates it blocks.
        # "Blocking" y means that if we add x, y becomes incompatible.
        # y is blocked if:
        # 1. 2y is in added_sums(x) -> 2y == 2x (impossible if x!=y) or 2y == x+s
        # 2. y+s is in added_sums(x) -> y+s == 2x or y+s == x+s'
        # 3. y+x is in used_sums (this would make y invalid even without x? No, y is valid now.
        #    Wait, if y+x is in used_sums, then adding x makes y invalid because y+x is a new sum
        #    that conflicts with an existing sum. But y is currently valid, meaning y's sums
        #    (2y, y+s) don't conflict with used_sums. It does NOT check y+x because x is not in selected.
        #    So if y+x is in used_sums, then adding x creates a conflict for y. Thus y is blocked.
        
        best_candidate = None
        min_blocks = float('inf')
        
        # Optimization: Precompute some things? 
        # For small N=256, O(V^2) per step is ~ 256^2 * 256 steps = 16M ops worst case.
        # With 30s limit, Python might be slow. Let's optimize the inner loop.
        
        # We'll pick the candidate that minimizes the number of valid_candidates it blocks.
        # This is the "Low Degree" heuristic in the context of the remaining graph.
        
        for x in valid_candidates:
            added_sums = set()
            added_sums.add(2 * x)
            for s in selected:
                added_sums.add(x + s)
                
            blocks = 0
            for y in valid_candidates:
                if y == x:
                    continue
                    
                # Check if y is blocked by adding x
                # y's new sums relative to selected+{x}: 2y, y+s (s in selected), y+x
                # Conflicts with added_sums:
                # 1. 2y in added_sums?
                # 2. y+s in added_sums?
                # 3. y+x in added_sums? (y+x == 2x => y=x, no. y+x == x+s => y=s, no.)
                #    So y+x is not in added_sums.
                # Conflicts with used_sums:
                # y is currently valid, so 2y and y+s are not in used_sums.
                # But y+x might be in used_sums.
                
                blocked = False
                if 2 * y in added_sums:
                    blocked = True
                else:
                    for s in selected:
                        if (y + s) in added_sums:
                            blocked = True
                            break
                    if not blocked:
                        # Check if y+x is in used_sums
                        if (y + x) in used_sums:
                            blocked = True
                            
                if blocked:
                    blocks += 1
                    
            if blocks < min_blocks:
                min_blocks = blocks
                best_candidate = x
                # Early termination if we find a candidate that blocks 0 others?
                # That would be ideal, as it preserves options.
                if min_blocks == 0:
                    break
                    
        if best_candidate is None:
            # Fallback if logic failed (shouldn't happen)
            best_candidate = valid_candidates[0]
            
        # Add best_candidate to selected
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        
        # Update used_sums
        for s in selected:
            used_sums.add(best_candidate + s)
        used_sums.add(2 * best_candidate)
        
    return selected

def construct():
    start_time = time.time()
    # Safety margin: 60s limit, return by 55s
    deadline = start_time + 55.0
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # 1. Try deterministic low-degree greedy
    # The low-degree heuristic depends on the state, so it's deterministic if we start with full set.
    # But "low degree" might have ties. We break ties arbitrarily (first found).
    # Let's try a few variations.
    
    # Variation A: Standard Low Degree (deterministic tie-break)
    s = _low_degree_greedy(rng, deadline)
    if s and len(s) < best_size:
        best_size = len(s)
        best_set = s[:]
        
    if time.time() > deadline:
        return best_set
        
    # Variation B: Randomized Low Degree (random tie-break among min-block candidates)
    # We can modify _low_degree_greedy to accept a random tie-break, or just run it multiple times
    # with different random shuffles of the initial 'remaining' set? 
    # Actually, the algorithm picks the min-block candidate. If there are ties, we pick the first one.
    # To randomize, we can shuffle the order of valid_candidates before checking.
    
    # Let's create a wrapper that shuffles the valid candidates list before selection.
    def _low_degree_greedy_randomized(rng, deadline):
        selected = []
        used_sums = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            if time.time() > deadline:
                break
                
            valid_candidates = []
            for x in remaining:
                valid = True
                if 2 * x in used_sums:
                    valid = False
                else:
                    for s in selected:
                        if (x + s) in used_sums:
                            valid = False
                            break
                if valid:
                    valid_candidates.append(x)
                    
            if not valid_candidates:
                break
                
            # Shuffle to randomize tie-breaking
            rng.shuffle(valid_candidates)
            
            best_candidate = None
            min_blocks = float('inf')
            
            for x in valid_candidates:
                added_sums = set()
                added_sums.add(2 * x)
                for s in selected:
                    added_sums.add(x + s)
                    
                blocks = 0
                for y in valid_candidates:
                    if y == x:
                        continue
                    blocked = False
                    if 2 * y in added_sums:
                        blocked = True
                    else:
                        for s in selected:
                            if (y + s) in added_sums:
                                blocked = True
                                break
                        if not blocked:
                            if (y + x) in used_sums:
                                blocked = True
                    if blocked:
                        blocks += 1
                        
                if blocks < min_blocks:
                    min_blocks = blocks
                    best_candidate = x
                    if min_blocks == 0:
                        break
                        
            if best_candidate is None:
                best_candidate = valid_candidates[0]
                
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            for s in selected:
                used_sums.add(best_candidate + s)
            used_sums.add(2 * best_candidate)
            
        return selected

    # Run randomized low-degree greedy multiple times
    for i in range(20):
        if time.time() > deadline:
            break
        s = _low_degree_greedy_randomized(rng, deadline)
        if s and len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            if best_size <= 13:
                break
                
    # 2. Also try standard greedy with different orders as a fallback/comparison
    # The current program did this. It might find different structures.
    for order in [list(range(N, 0, -1)), list(range(1, N + 1))]:
        if time.time() > deadline:
            break
        s = _greedy_construct(order)
        if s and len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Randomized standard greedy
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = _greedy_construct(order, rng)
        if s and len(s) < best_size:
            best_size = len(s)
            best_set = s[:]
            if best_size <= 13:
                break

    if not best_set or not _is_valid_solution(best_set):
        # Fallback to a known valid construction or simple greedy
        fallback = _greedy_construct(list(range(1, N+1)))
        if _is_valid_solution(fallback):
            return fallback
        # Last resort: hardcoded small set if found in previous runs? 
        # Or just return the greedy result even if not maximal? No, must be valid.
        # The greedy result is always a Sidon set. Is it maximal?
        # Greedy until no more can be added ensures maximality.
        # _greedy_construct stops when no more can be added? 
        # No, it iterates through `order`. If it skips elements, it doesn't retry them.
        # So the result of _greedy_construct is NOT necessarily maximal!
        # Wait, _greedy_construct in the seed:
        # for x in order: ... if conflict: continue ...
        # It does not retry skipped x. So if x is skipped because of conflict with current set,
        # and later the set changes? No, the set only grows. So if x conflicts with current set,
        # it will conflict with any superset. So x can never be added later.
        # Thus, at the end, no element in `order` that was skipped can be added.
        # What about elements not in `order`? `order` is a permutation of 1..N.
        # So all elements are considered. If skipped, they can't be added.
        # So _greedy_construct DOES produce a maximal Sidon set.
        
        return fallback

    return best_set

# EVOLVE-BLOCK-END
