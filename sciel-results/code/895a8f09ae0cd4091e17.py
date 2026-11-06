import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples (a, b, c) with distinct elements in [1, N]
    # Equation: 1/a = 1/b + 1/c  => c = ab / (b-a)
    # We assume a < b. Then c is determined.
    # We must ensure a, b, c are distinct.
    # Also 1/a > 1/b implies a < b. 1/a > 1/c implies a < c.
    # So a is always the smallest element in the triple?
    # If a < b, then 1/b < 1/a. 1/c = 1/a - 1/b > 0.
    # c = ab/(b-a).
    # If b > a, b-a > 0.
    # Is c > a? ab/(b-a) > a <=> ab > ab - a^2 <=> 0 > -a^2 (True). So c > a.
    # Is c > b? ab/(b-a) > b <=> ab > b^2 - ab <=> 2ab > b^2 <=> 2a > b.
    # So if b < 2a, then c > b. If b > 2a, then c < b.
    # In either case, a is the smallest.
    
    triples = []
    # Map from element to list of indices of triples it belongs to
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # We have a valid triple (a, b, c)
                    # Sort them for consistency? Not strictly needed for logic, but good for uniqueness if we were using sets.
                    # Here we just store the tuple.
                    t = (a, b, c)
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)

    # 2. Greedy Removal Algorithm
    # State: active_elements (set), active_triples (set of indices)
    active_elements = set(elements)
    active_triples = set(range(len(triples)))
    
    # Conflict count for each element: number of active triples it is in
    conflict_count = {i: 0 for i in range(1, N + 1)}
    for idx in active_triples:
        for x in triples[idx]:
            conflict_count[x] += 1
    
    # We need to efficiently find the element with max conflict count among those with count > 0.
    # Since N is small (180), we can just scan all elements each time.
    # Number of removals is at most 180. Scanning 180 elements is cheap.
    
    while time.time() < deadline:
        # Find element with max conflict count
        max_conflict = -1
        victim = -1
        for x in active_elements:
            if conflict_count[x] > max_conflict:
                max_conflict = conflict_count[x]
                victim = x
        
        if max_conflict == 0:
            break # No more conflicts
        
        # Remove victim
        active_elements.remove(victim)
        
        # Update active_triples and conflict_counts
        # All triples containing victim are now inactive
        for idx in elem_to_triples[victim]:
            if idx in active_triples:
                active_triples.remove(idx)
                for x in triples[idx]:
                    if x != victim and x in active_elements:
                        conflict_count[x] -= 1
                        # If conflict_count becomes 0, it's fine, it just won't be a victim unless it gets higher later (which it won't, counts only decrease)
        
        # Early exit check
        if len(active_elements) == 0:
            break
            
    # Fallback: if something went wrong or time ran out badly, return at least something valid
    # The greedy process ensures validity by construction (we remove until no active triples remain)
    # So active_elements is always a valid set.
    
    return list(active_elements)

# EVOLVE-BLOCK-END
