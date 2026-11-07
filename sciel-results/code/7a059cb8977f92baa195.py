import time

def construct():
    start_time = time.time()
    deadline = 60.0
    
    # reachable[c] is the set of sums achievable with exactly c elements
    reachable = [{0}]
    # sum_to_card maps a sum to its cardinality
    # Since sums are disjoint across cardinalities, this is a valid mapping
    sum_to_card = {0: 0}
    
    selected = []
    
    # Iterate from largest to smallest to minimize collisions
    for x in range(256, 0, -1):
        if time.time() > deadline - 1.0:
            break
            
        valid = True
        new_sums = []
        
        # Check if adding x is valid
        # We iterate through existing cardinalities
        for c in range(len(reachable) - 1, -1, -1):
            for s in reachable[c]:
                ns = s + x
                # Check if ns collides with an existing sum of a DIFFERENT cardinality
                if ns in sum_to_card:
                    if sum_to_card[ns] != c + 1:
                        valid = False
                        break
                else:
                    # ns is new, so no collision with old sums
                    # We don't need to check against other new sums because
                    # new sums from different c are distinct (proved above)
                    # and new sums from same c are distinct (s are distinct)
                    pass
                
            if not valid:
                break
        
        if valid:
            # Update reachable and sum_to_card
            # We need to add the new sums to reachable[c+1]
            # And update sum_to_card
            
            # To avoid modifying reachable while iterating, we already finished iteration.
            # But we need to collect the new sums.
            # Actually, we can just re-iterate or store them.
            # Since we broke out of the check loop, we need to do the update.
            # It's cleaner to just re-iterate for the update if valid.
            
            for c in range(len(reachable) - 1, -1, -1):
                for s in reachable[c]:
                    ns = s + x
                    # Add to reachable[c+1]
                    if c + 1 >= len(reachable):
                        reachable.append(set())
                    reachable[c+1].add(ns)
                    
                    # Update sum_to_card
                    # If ns was already in sum_to_card, it must have been with card c+1
                    # (because valid check passed)
                    # If it wasn't, we add it.
                    if ns not in sum_to_card:
                        sum_to_card[ns] = c + 1
                    # If it was, it's already c+1, so no change needed.
                    
            selected.append(x)

    return selected
