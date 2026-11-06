"""Rewritten program: maximize subset of [1,180] with no distinct triple a,b,c s.t. 1/a=1/b+1/c."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict

def construct():
    N = 180
    # Find all forbidden triples (a,b,c) distinct in [1,N] with 1/a = 1/b + 1/c
    # a is the smallest. c = a*b/(b-a). b > a.
    forbidden = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    forbidden.add(frozenset((a, b, c)))
    
    # Build adjacency: for each element, list of triples it participates in
    triples_of = defaultdict(list)
    for tri in forbidden:
        for x in tri:
            triples_of[x].append(tri)
    
    # Precompute list of forbidden triples for fast access
    forbidden_list = list(forbidden)
    
    def greedy_construct(order):
        """
        Greedily build a set by iterating through 'order'.
        Returns the set of included elements.
        """
        current_set = set()
        # trip_count[t] = number of elements of t currently in current_set
        trip_count = {t: 0 for t in forbidden_list}
        
        for x in order:
            # Check if adding x creates any triple
            can_add = True
            for t in triples_of[x]:
                if trip_count[t] == 2:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(x)
                for t in triples_of[x]:
                    trip_count[t] += 1
        
        return current_set

    def local_search(initial_set):
        """
        Improve the set using local search (additions and swaps).
        """
        current_set = set(initial_set)
        # Rebuild trip_count for current_set
        trip_count = {t: 0 for t in forbidden_list}
        for x in current_set:
            for t in triples_of[x]:
                trip_count[t] += 1
        
        excluded = set(range(1, N + 1)) - current_set
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any excluded element
            added_any = False
            for x in list(excluded):
                can_add = True
                for t in triples_of[x]:
                    if trip_count[t] == 2:
                        can_add = False
                        break
                if can_add:
                    current_set.add(x)
                    excluded.discard(x)
                    for t in triples_of[x]:
                        trip_count[t] += 1
                    improved = True
                    added_any = True
                    # Restart addition loop? Or just continue? 
                    # Restarting is safer to catch new opportunities immediately.
                    # But breaking and restarting the while loop is more standard.
                    # Let's just break out of the for loop and let the while loop handle it.
                    break
            
            if added_any:
                continue # Restart the while loop to try more additions
            
            # 2. If no additions, try 1-for-1 swaps
            # We try to remove x and add y if it allows the set to remain valid 
            # and potentially allows more additions later.
            # To keep it simple and effective, we'll just try swaps that result in a valid set 
            # and then restart the addition phase.
            swap_made = False
            for x in list(current_set):
                # Temporarily remove x
                current_set.discard(x)
                # Save state of trip_counts affected by x
                # Instead of saving full state, we can just try to add y 
                # and see if it's valid. But we need to know if the set *after* 
                # removing x is valid (it is, as we removed an element).
                # Then we try to add y.
                
                # We need to check if adding y is valid given current_set (which now excludes x).
                # But trip_count still reflects the state WITH x.
                # We must update trip_count for removal of x first.
                for t in triples_of[x]:
                    trip_count[t] -= 1
                
                # Now try to add some y from excluded
                for y in list(excluded):
                    can_add = True
                    for t in triples_of[y]:
                        if trip_count[t] == 2:
                            can_add = False
                            break
                    if can_add:
                        # Perform swap: add y
                        current_set.add(y)
                        excluded.discard(y)
                        for t in triples_of[y]:
                            trip_count[t] += 1
                        
                        # The set size is the same, but the composition changed.
                        # This might allow further additions.
                        swap_made = True
                        improved = True
                        break # Break inner for y loop
                
                if swap_made:
                    break # Break outer for x loop
                
                # If no y could be added, restore x
                if not swap_made:
                    current_set.add(x)
                    for t in triples_of[x]:
                        trip_count[t] += 1
                    # Note: excluded was not modified in this branch, so no need to restore it.
            
            if swap_made:
                continue # Restart while loop to try additions again
        
        return current_set

    # Run multiple randomized greedy constructions
    best_set = set()
    best_size = 0
    num_iterations = 200 # Adjust based on time limit
    
    for i in range(num_iterations):
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Apply local search to the best set found
    final_set = local_search(best_set)
    
    return sorted(final_set)
# EVOLVE-BLOCK-END
