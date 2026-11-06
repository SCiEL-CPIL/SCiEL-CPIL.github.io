"""Optimized solver for large sets avoiding reciprocal Schur triples."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all valid triples (a, b, c) of DISTINCT elements in [1, N]
    such that 1/a = 1/b + 1/c.
    
    From the equation: 1/a = (b+c)/(bc) => bc = a(b+c) => bc - ab - ac = 0.
    Multiply by a: abc - a^2b - a^2c = 0.
    Alternatively, solve for c: 1/c = 1/a - 1/b = (b-a)/(ab) => c = ab/(b-a).
    
    For c to be a positive integer, we need b > a and (b-a) | ab.
    Let d = b - a. Then b = a + d.
    c = a(a+d)/d = a^2/d + a.
    So d must divide a^2.
    
    We iterate over a, then over divisors d of a^2 such that b = a+d <= N.
    Then c = a(a+d)/d. We check if c is an integer (it will be by construction)
    and 1 <= c <= N and c != a and c != b.
    
    Note: The equation is symmetric in b and c. So for a given a, the pairs (b,c) 
    are unordered. However, the condition "three distinct elements" just means 
    a, b, c are all different.
    
    We store the set of forbidden triples as a list of frozensets or just tuples 
    for quick lookup? No, for the hitting set approach, we need to know which 
    elements form a triple.
    
    Let's store the triples as a list of (a, b, c) where a < b < c is NOT necessarily true.
    Actually, it's better to store them as sets of 3 elements for easy checking.
    But for the removal algorithm, we need to know for each element, which triples it is part of.
    
    Let's build:
    1. A list of all valid triples (as tuples of 3 distinct ints).
    2. A mapping from each element x to the list of triple indices that contain x.
    """
    triples = []
    # We need distinct a, b, c.
    # From c = ab/(b-a), with b > a.
    # Also we can have the case where the "a" in the equation is not the smallest.
    # The equation is 1/a = 1/b + 1/c. This implies a < b and a < c.
    # So 'a' is the smallest element in the triple?
    # Let's check: 1/a = 1/b + 1/c. Since b,c >= 1, 1/b + 1/c <= 2. So a >= 1/2, a>=1.
    # Also 1/a > 1/b => a < b. And 1/a > 1/c => a < c.
    # So yes, 'a' is strictly smaller than 'b' and 'c'.
    # Thus, in any valid triple (x, y, z) satisfying the equation, one element is the smallest.
    # Let the smallest be 'a'. Then the other two are 'b' and 'c' (order doesn't matter for the set).
    
    for a in range(1, N + 1):
        # Find all divisors d of a^2 such that b = a + d <= N
        # d = b - a. c = a(a+d)/d.
        # We need d | a^2.
        # Iterate through possible d.
        # Since b <= N, d <= N - a.
        # Also d must be a divisor of a^2.
        
        # To be efficient, iterate through divisors of a^2.
        # Or just iterate d from 1 to N-a and check if (a*a) % d == 0?
        # N=180 is small. a^2 is at most 32400.
        # Number of divisors is small.
        # Let's just iterate d from 1 to N-a.
        
        for d in range(1, N - a + 1):
            if (a * a) % d == 0:
                b = a + d
                c = (a * b) // d
                # Check if c is integer (it is by construction)
                # Check bounds and distinctness
                if 1 <= c <= N:
                    if c != a and c != b:
                        # We have a valid triple {a, b, c}
                        # Note: b and c might be equal? 
                        # If b = c, then 1/a = 2/b => b = 2a.
                        # Then d = a. a^2 % a == 0. 
                        # c = a(2a)/a = 2a = b.
                        # So b=c is possible. But the task requires THREE DISTINCT elements.
                        # So we must exclude cases where b == c.
                        if b != c:
                            triples.append((a, b, c))
    
    # Remove duplicates?
    # Since we iterate a, and for each a find b,c.
    # Could the same set {a,b,c} be generated twice?
    # No, because 'a' is uniquely the smallest element in the set that satisfies the equation structure?
    # Wait. If {x,y,z} is a set, and 1/x = 1/y + 1/z.
    # x must be the smallest.
    # So for a given set, only the smallest element 'a' will generate it in our loop.
    # So no duplicates.
    
    return triples

def _build_index(N=180, triples=None):
    if triples is None:
        triples = _precompute_triples(N)
    
    # Map element -> list of triple indices
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
    
    return triples, elem_to_triples

def _remove_random_greedy(N=180, triples=None, elem_to_triples=None, seed=None):
    """
    Randomized greedy hitting set removal.
    Start with full set [1..N].
    While there is a triple fully contained in the set:
        Pick a triple at random (or weighted by frequency?).
        Remove one of its elements at random (or the one with highest degree?).
    """
    if seed is not None:
        random.seed(seed)
        
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    # State: boolean array of size N+1, True if element is present
    present = [True] * (N + 1)
    
    # Active triples: those where all 3 elements are present
    # We can maintain a count of how many elements of each triple are present.
    # Or simpler: just check on the fly? N=180, triples ~ few hundred.
    # Let's maintain a list of active triple indices.
    
    # Initialize active triples
    active_triples = set(range(len(triples)))
    
    # For each triple, count how many of its elements are present.
    # Initially all 3.
    # When we remove an element, we decrement the count for all triples containing it.
    # If count drops to 2, it's no longer active.
    
    triple_counts = [3] * len(triples)
    
    # To pick a random active triple efficiently, we can maintain a list of active indices.
    # But removing from a list is O(n). 
    # Given small N, a simple loop to find a random active triple is fine.
    
    # Better: maintain a list `active_list` of indices.
    # When a triple becomes inactive, remove it from active_list.
    # Removal from list is O(n).
    
    active_list = list(range(len(triples)))
    
    while active_list:
        # Pick a random index from active_list
        pos = random.randint(0, len(active_list) - 1)
        t_idx = active_list[pos]
        a, b, c = triples[t_idx]
        
        # Decide which element to remove.
        # Heuristic: remove the one that appears in the most other active triples?
        # Or just random?
        # Let's try random first, maybe weighted by degree.
        
        # Calculate "cost" of removing each element: number of active triples it is in.
        # If we remove x, all active triples containing x become inactive.
        
        # Get active triples for a, b, c
        # Note: elem_to_triples gives ALL triples. We need to check if they are active.
        
        def count_active_triples_for_elem(x):
            cnt = 0
            for t_idx in elem_to_triples[x]:
                if t_idx in active_list: # This is O(n) check
                    cnt += 1
            return cnt
            
        # This is slow. Let's just pick random element from the triple.
        # Or pick the one with highest global degree (static)?
        
        # Let's try: pick the element that is in the most triples (static degree).
        # Precompute static degrees.
        
        # For now, random choice among a, b, c.
        x_to_remove = random.choice([a, b, c])
        
        # Remove x_to_remove
        present[x_to_remove] = False
        
        # Update active_list and triple_counts
        # For each triple containing x_to_remove:
        for t_idx in elem_to_triples[x_to_remove]:
            if triple_counts[t_idx] > 0:
                triple_counts[t_idx] -= 1
                if triple_counts[t_idx] == 2:
                    # This triple is now inactive. Remove from active_list.
                    # Find t_idx in active_list and remove it.
                    # Swap with last element and pop for O(1) removal?
                    # But we need to keep track of positions if we use random index.
                    # Simple approach: mark as removed?
                    # Since N is small, let's just rebuild active_list periodically or use a set?
                    # Using a set for active_list makes random selection harder (need list(set)).
                    
                    # Let's just use a list and remove by value. O(n).
                    # N_triples is small (~100-200).
                    try:
                        active_list.remove(t_idx)
                    except ValueError:
                        pass # Already removed? Shouldn't happen if logic is correct.
                        
    result = [i for i in range(1, N + 1) if present[i]]
    return result

def _local_search(initial_set, N=180, triples=None, elem_to_triples=None, max_iters=1000, deadline=None):
    """
    Try to add elements back to the set if they don't create a violation.
    """
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    present = set(initial_set)
    candidates = set(range(1, N + 1)) - present
    
    # To check if adding x is valid, we need to ensure no triple {x, y, z} with y,z in present exists.
    # This is equivalent to checking if x is part of any triple where the other two are in present.
    
    # Precompute for each x, the list of triples it's in.
    # For a candidate x, check all triples (x, y, z). If y in present and z in present, invalid.
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        # Shuffle candidates for randomness
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for x in cand_list:
            if deadline and time.time() > deadline:
                break
            if x in present:
                continue
                
            valid = True
            for t_idx in elem_to_triples[x]:
                a, b, c = triples[t_idx]
                # Find the other two elements
                others = [y for y in (a, b, c) if y != x]
                if others[0] in present and others[1] in present:
                    valid = False
                    break
                    
            if valid:
                present.add(x)
                candidates.remove(x)
                improved = True
                
    return list(present)

def _greedy_add(N=180, triples=None, elem_to_triples=None, seed=None):
    """
    Greedy addition: start with empty set, add elements one by one if valid.
    Order: random or by degree?
    """
    if seed is not None:
        random.seed(seed)
        
    if triples is None or elem_to_triples is None:
        triples, elem_to_triples = _build_index(N)
        
    present = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        valid = True
        for t_idx in elem_to_triples[x]:
            a, b, c = triples[t_idx]
            others = [y for y in (a, b, c) if y != x]
            if others[0] in present and others[1] in present:
                valid = False
                break
        if valid:
            present.add(x)
            
    return list(present)

def construct():
    N = 180
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Precompute
    triples, elem_to_triples = _build_index(N)
    
    best_set = set(range(91, 181)) # Initial seed result (size 90)
    best_size = len(best_set)
    
    # Strategy 1: Randomized Greedy Removal
    # Run multiple times with different seeds
    num_removal_runs = 20
    for i in range(num_removal_runs):
        if time.time() > deadline:
            break
        seed = i
        s = _remove_random_greedy(N, triples, elem_to_triples, seed=seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
    # Strategy 2: Local Search on the best set found so far
    if time.time() < deadline:
        s = _local_search(list(best_set), N, triples, elem_to_triples, max_iters=500, deadline=deadline)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
    # Strategy 3: Greedy Add from scratch, multiple times
    num_add_runs = 20
    for i in range(num_add_runs):
        if time.time() > deadline:
            break
        seed = 100 + i
        s = _greedy_add(N, triples, elem_to_triples, seed=seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
        # Try local search on the greedy add result
        if time.time() < deadline:
            s2 = _local_search(s, N, triples, elem_to_triples, max_iters=200, deadline=deadline)
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = set(s2)
                
    # Final local search on the global best
    if time.time() < deadline:
        s_final = _local_search(list(best_set), N, triples, elem_to_triples, max_iters=1000, deadline=deadline)
        if len(s_final) > best_size:
            best_set = set(s_final)
            
    return list(best_set)

# EVOLVE-BLOCK-END
