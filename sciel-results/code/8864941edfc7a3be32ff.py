import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc / (b+c)
    # Condition: (b+c) divides (b*c)
    
    a_to_bc = {i: [] for i in range(N + 1)}
    b_to_ac = {i: [] for i in range(N + 1)}
    c_to_ab = {i: [] for i in range(N + 1)}
    
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triples.append((a, b, c))
                    a_to_bc[a].append((b, c))
                    b_to_ac[b].append((a, c))
                    c_to_ab[c].append((a, b))
    
    # Calculate degree (number of triples each element is involved in)
    degree = [0] * (N + 1)
    for a, b, c in triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        
    # Sort elements by degree (ascending). Low degree elements are "safer" to add.
    sorted_by_degree = list(range(1, N + 1))
    sorted_by_degree.sort(key=lambda x: degree[x])
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        # x as 'a'
        for (b, c) in a_to_bc[x]:
            if b in S and c in S:
                return False
        # x as 'b'
        for (a, c) in b_to_ac[x]:
            if a in S and c in S:
                return False
        # x as 'c'
        for (a, b) in c_to_ab[x]:
            if a in S and b in S:
                return False
        return True

    def is_valid(S):
        """Verify S has no forbidden triple."""
        for a in S:
            for (b, c) in a_to_bc[a]:
                if b in S and c in S:
                    return False
        return True

    def maximize(S):
        """Greedy addition until no more elements can be added, using degree-sorted order."""
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            # Iterate in order of increasing degree
            for x in sorted_by_degree:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
        return S

    def greedy_min_degree(rng):
        """Construct an initial set using greedy selection based on degree."""
        S = set()
        # We can add randomness to tie-breaking or the order if degrees are equal.
        # Here we just use the static sorted order. To add diversity, we can shuffle 
        # elements with the same degree.
        
        # Group by degree? Or just shuffle the whole list slightly?
        # Let's stick to the static order for determinism in the "base" greedy, 
        # but we will run multiple ILS restarts.
        
        for x in sorted_by_degree:
            if can_add(x, S):
                S.add(x)
        return S

    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    # Initial Greedy: Run a few times with slight variations?
    # The static order is deterministic. 
    # Let's try a few random shuffles of the "low degree" part?
    # Or just run the ILS which will perturb.
    
    S = greedy_min_degree(rng)
    maximize(S) # Ensure it's fully maximized
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
        
    # Also try a standard ascending greedy as a backup/diversity
    S2 = set()
    for x in range(1, N + 1):
        if can_add(x, S2):
            S2.add(x)
    if len(S2) > best_size:
        best_set = set(S2)
        best_size = len(S2)

    # Iterated Local Search
    while time.time() < deadline:
        # Perturb: Remove k random elements from current S (or best if S is empty)
        if not S:
            S = set(best_set)
        
        # Determine kick size
        kick_size = rng.randint(1, 3)
        
        # Remove kick_size elements
        if len(S) > kick_size:
            elements_to_remove = rng.sample(list(S), kick_size)
            for e in elements_to_remove:
                S.remove(e)
        elif len(S) > 0:
            if len(S) <= kick_size:
                 S.clear()
        
        # Maximize again
        maximize(S)
        
        # Update best
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
            
        # If we have used all elements, stop
        if best_size == N:
            break

    # Final verification
    if not is_valid(best_set):
        # Fallback
        best_set = set(range(91, N + 1))
        if not is_valid(best_set):
            best_set = {180}
            
    return list(best_set)

# EVOLVE-BLOCK-END
