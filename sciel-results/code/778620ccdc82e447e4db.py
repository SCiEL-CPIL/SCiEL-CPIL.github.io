"""Rewritten program: maximize subset of [1,180] with no distinct triple a,b,c s.t. 1/a=1/b+1/c."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict

def construct():
    N = 180
    # Find all forbidden triples (a,b,c) distinct in [1,N] with 1/a = 1/b + 1/c
    # c = a*b/(b-a), need b > a, c integer in [1,N], c distinct from a,b
    forbidden = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    forbidden.add(frozenset((a, b, c)))
    
    # Precompute adjacency: for each element, list of triples it participates in
    triples_of = defaultdict(list)
    for tri in forbidden:
        for x in tri:
            triples_of[x].append(tri)

    # Helper to check if x can be added to included
    def can_add(x, included):
        for tri in triples_of[x]:
            if (tri - {x}).issubset(included):
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Run multiple random restarts
    for restart in range(100):
        included = set()
        excluded = set(range(1, N + 1))
        
        # Initial greedy fill with random order
        items = list(range(1, N + 1))
        random.shuffle(items)
        for x in items:
            if can_add(x, included):
                included.add(x)
                excluded.discard(x)
        
        if len(included) > best_size:
            best_size = len(included)
            best_set = set(included)
        
        # Local Search
        # We will try to improve the set.
        # Stagnation counter to break out of loops if no improvement
        stagnation = 0
        max_stagnation = 50
        
        while stagnation < max_stagnation:
            # 1. Try to add any excluded element
            added_any = False
            exc_list = list(excluded)
            random.shuffle(exc_list)
            for x in exc_list:
                if can_add(x, included):
                    included.add(x)
                    excluded.discard(x)
                    added_any = True
                    if len(included) > best_size:
                        best_size = len(included)
                        best_set = set(included)
                    break # Restart the "try add" phase to catch new opportunities
            
            if not added_any:
                # 2. Try a "kick": remove a random element, add all possible
                if not included:
                    break
                
                # Save state
                old_included = set(included)
                old_excluded = set(excluded)
                old_size = len(included)
                
                # Pick a random element to remove
                v = random.choice(list(included))
                included.discard(v)
                excluded.add(v)
                
                # Try to add all possible from excluded
                # We do this greedily
                exc_list = list(excluded)
                random.shuffle(exc_list)
                for y in exc_list:
                    if can_add(y, included):
                        included.add(y)
                        excluded.discard(y)
                
                # Check if we improved
                if len(included) > old_size:
                    stagnation = 0 # Reset stagnation
                    if len(included) > best_size:
                        best_size = len(included)
                        best_set = set(included)
                else:
                    # Revert
                    included = old_included
                    excluded = old_excluded
                    stagnation += 1
        
        # End of local search for this restart

    return sorted(best_set)
# EVOLVE-BLOCK-END
