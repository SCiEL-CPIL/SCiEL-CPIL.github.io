import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0
    N = 4096
    
    # Step 1: Construct a maximal Sidon set in [1, N]
    # A Sidon set has all pair sums a+b (a<=b) distinct.
    def construct_sidon(n):
        sidon = []
        pair_sums = set()
        
        for x in range(1, n + 1):
            valid = True
            # Check 2x
            if 2 * x in pair_sums:
                valid = False
            else:
                # Check x + a for all a in sidon
                for a in sidon:
                    s = x + a
                    if s in pair_sums:
                        valid = False
                        break
                if valid:
                    # If we got here, all pair sums involving x are new
                    # We should add them to a temp set to check for internal conflicts?
                    # No, we check against existing pair_sums.
                    # Do new sums conflict with each other?
                    # 2x vs x+a? x+a = 2x => a=x (impossible, x not in sidon yet)
                    # x+a vs x+b? a=b (impossible, distinct elements in loop? No, a,b from sidon)
                    # If a!=b, x+a != x+b.
                    # So only need to check against existing pair_sums.
                    pass
            
            if valid:
                # Add x to sidon
                sidon.append(x)
                # Update pair_sums
                pair_sums.add(2 * x)
                for a in sidon[:-1]: # All previously added
                    pair_sums.add(x + a)
                    
        return sidon

    # Step 2: Filter the Sidon set to be a B_3 set
    # B_3: All unordered triple sums distinct.
    def is_b3_and_get_conflict(S):
        sums = {}
        triples = list(combinations_with_replacement(S, 3))
        for t in triples:
            s = sum(t)
            if s in sums:
                return False, sums[s], t
            sums[s] = t
        return True, None, None

    # Get initial Sidon set
    sidon_set = construct_sidon(N)
    
    # Now prune to B_3
    current_set = sidon_set
    
    # To speed up, we can try to remove elements that are part of conflicts.
    # We will iterate until no conflicts or time limit.
    
    while time.time() - start_time < time_limit:
        is_valid, conf1, conf2 = is_b3_and_get_conflict(current_set)
        if is_valid:
            break
        
        # Conflict found: conf1 and conf2 are two different triples with same sum.
        # We need to remove at least one element from the union of conf1 and conf2.
        # Heuristic: Remove the element that appears in the most other triples?
        # Or just remove a random element from the union to explore different branches?
        # Since we want to maximize size, we should remove an element that is "least central".
        # Let's just remove a random element from the union of the two conflicting triples.
        # This is a random walk on the space of subsets.
        
        candidates_to_remove = set(conf1).union(set(conf2))
        if not candidates_to_remove:
            break
            
        # Pick a random one to remove
        # To be more deterministic and potentially better, pick the one with highest value?
        # Or lowest?
        # Let's try removing the largest element first, as smaller elements are more "dense" in sum space?
        # Actually, removing a large element might free up high sum space.
        # Let's just pick one randomly to allow variety, or use a simple heuristic.
        # Given the time limit, random is fine.
        elem_to_remove = max(candidates_to_remove) # Try removing largest first?
        # Let's stick to random for now to avoid getting stuck in a specific local minimum.
        # import random
        # elem_to_remove = random.choice(list(candidates_to_remove))
        
        current_set.remove(elem_to_remove)
        
    return current_set
