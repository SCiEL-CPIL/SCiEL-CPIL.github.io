import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all distinct triples (c, a, b) with c < a < b
    such that 1/c = 1/a + 1/b and a, b, c <= N.
    Returns a list of tuples (c, a, b).
    """
    triples = []
    # c is the smallest element
    for c in range(1, N + 1):
        # (a-c)(b-c) = c^2
        # Let u = a-c, v = b-c. u*v = c^2.
        # a = c+u, b = c+v.
        # We need a, b <= N.
        # So c+u <= N and c+v <= N.
        # Also u, v >= 1.
        # u <= v WLOG (so a <= b).
        # u <= c.
        c_sq = c * c
        # Iterate u from 1 to c
        for u in range(1, c + 1):
            if c_sq % u != 0:
                continue
            v = c_sq // u
            a = c + u
            b = c + v
            if a <= N and b <= N:
                # Ensure distinct
                if a != b and a != c and b != c:
                    triples.append((c, a, b))
    return triples

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    N = 180
    
    triples = _precompute_triples(N)
    
    # Map element to list of triples it belongs to
    # To speed up updates, we can use sets of triple indices
    elem_to_triples = {i: set() for i in range(1, N + 1)}
    for i, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].add(i)
            
    # Active triples: set of indices
    active_triples = set(range(len(triples)))
    
    # Removed elements
    removed = set()
    
    # Degrees: number of active triples each element is in
    degrees = {i: len(elem_to_triples[i]) for i in range(1, N + 1)}
    
    # Greedy Vertex Cover
    while active_triples:
        if time.time() >= deadline:
            break
            
        # Pick a random active triple to start, or one with max "conflict"
        # Picking a triple with elements of high degree is good.
        # Let's just pick the first one, or a random one to avoid bias.
        # Random might be better for diversity? Or deterministic max degree?
        # Let's try: pick a triple, then pick the vertex in it with max degree.
        
        # To be efficient, we can just iterate through active_triples
        # But we need to pick one.
        # Let's pick a triple that has the highest sum of degrees?
        # Or just pick the first one.
        # Actually, picking the triple with the "most constrained" elements is good.
        
        # Optimization: Instead of checking all active triples to find the best one,
        # we can just pick one arbitrarily, then pick the best vertex in it.
        # This is O(1) to pick triple, O(3) to pick vertex.
        
        # However, if we always pick the first active triple, we might be biased.
        # Let's shuffle the list of active triples occasionally?
        # Or just pick a random one.
        
        # Let's pick a random active triple.
        t_idx = random.choice(list(active_triples))
        t = triples[t_idx]
        
        # Find the element in t with max degree
        best_elem = max(t, key=lambda x: degrees[x])
        
        # Remove best_elem
        removed.add(best_elem)
        
        # Update active_triples and degrees
        # All triples containing best_elem are now broken
        to_remove_triples = list(elem_to_triples[best_elem])
        for tri_idx in to_remove_triples:
            if tri_idx in active_triples:
                active_triples.remove(tri_idx)
                # Update degrees for other elements in this triple
                for x in triples[tri_idx]:
                    if x != best_elem:
                        degrees[x] -= 1
        
        # Reset degree of removed element? Not necessary, it's removed.
        # But we should mark it so we don't consider it for future removals?
        # It's in 'removed', so we don't pick it again.
        
    # Current set is {1..N} - removed
    current_set = set(range(1, N + 1)) - removed
    
    # Local Search: Try to add back removed elements
    # If adding x back doesn't create any new triple, add it.
    # A new triple is created if there exist a, b in current_set such that (x, a, b) is a triple.
    # Or (a, b, x) etc.
    # Since triples are precomputed, we can just check if any triple containing x is fully in current_set U {x}.
    
    # Precompute for each element x, the list of triples containing x.
    # We already have elem_to_triples.
    
    # We need to check if adding x back completes any triple.
    # A triple (c, a, b) is completed if c, a, b are all in the set.
    # If we add x, we only need to check triples containing x.
    # For each such triple, check if the other two elements are in current_set.
    
    # To make this fast, we can maintain a count of "active" elements in each triple?
    # Or just check on the fly.
    
    # Let's try a simple loop:
    # Iterate through removed elements.
    # Try to add each one back.
    
    # To improve, we can do multiple passes.
    
    # Also, we can try swaps: remove y from current_set, add x from removed.
    # If this increases size (it doesn't, size is same) but allows adding more later?
    # Or if it resolves conflicts?
    
    # Actually, the greedy might have removed too many.
    # Local search "add back" is key.
    
    # Let's refine the "add back" check.
    # For x in removed:
    #   if no triple containing x has both other elements in current_set:
    #       add x to current_set, remove from removed
    
    # We can do this in a loop until no more can be added.
    
    def can_add(x, S):
        # Check if adding x to S creates a triple
        for tri_idx in elem_to_triples[x]:
            t = triples[tri_idx]
            # Check if the other two are in S
            # t has 3 elements. One is x.
            # Find the other two.
            others = [y for y in t if y != x]
            if others[0] in S and others[1] in S:
                return False
        return True

    # Pass 1: Add back
    changed = True
    while changed:
        if time.time() >= deadline:
            break
        changed = False
        # Iterate through a copy of removed
        for x in list(removed):
            if can_add(x, current_set):
                current_set.add(x)
                removed.discard(x)
                changed = True
        
    # Pass 2: Swap search
    # Try to swap an element y in current_set with x in removed.
    # If we remove y, can we add x?
    # And does removing y allow adding other elements?
    # Simple swap:
    # For x in removed:
    #   For y in current_set:
    #      If (current_set - {y}) U {x} is valid (no triples), and len is same, 
    #      maybe it helps?
    # Actually, we want to maximize size. Swap keeps size same.
    # But it might allow adding MORE elements later.
    
    # Let's try a more aggressive local search:
    # Randomly remove an element from current_set, then greedily add back as many as possible from removed + that element.
    
    # Or, simpler:
    # Try to add x back. If it fails, it's because of some triples.
    # Maybe we can remove one element from each blocking triple?
    
    # Given time constraints, let's stick to the add-back and simple swaps.
    
    # Let's do a few random restarts of the local search?
    # Or just one good pass.
    
    # Let's try to improve the greedy by using a better selection strategy.
    # The current greedy picks a random triple, then max degree vertex.
    # We can pick the triple with the max sum of degrees of its vertices.
    
    # Let's re-run the greedy with a better strategy if time permits.
    # But the current result is likely good.
    
    # Final verification
    # Check if current_set has any triples.
    # This is O(|S|^2) or using precomputed triples O(|S| * degree).
    
    # If invalid, fallback to a safe set?
    # The greedy should produce a valid set.
    
    return list(current_set)

# EVOLVE-BLOCK-END
