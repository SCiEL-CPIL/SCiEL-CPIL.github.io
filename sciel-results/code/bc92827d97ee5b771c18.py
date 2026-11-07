import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # sums_by_card[k] = set of achievable sums using exactly k elements
    # Start with empty set: cardinality 0 has sum 0
    sums_by_card = {0: {0}}
    chosen = []
    
    # Try adding numbers from 256 down to 1
    for x in range(256, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Check if adding x maintains disjointness
        # New sums: for each k, new_sums[k+1] includes old_sums[k] + x
        # We need: for all k1 != k2, new_sums[k1] ∩ new_sums[k2] = empty
        
        # Compute the new sums_by_card tentatively
        max_k = len(sums_by_card) - 1
        new_sums = {}
        for k in range(max_k + 1):
            new_sums[k] = set(sums_by_card[k])
        
        # Add new sums from including x
        valid = True
        for k in range(max_k + 1):
            new_card = k + 1
            if new_card not in new_sums:
                new_sums[new_card] = set()
            for s in sums_by_card[k]:
                ns = s + x
                new_sums[new_card].add(ns)
        
        # Check disjointness across all cardinalities
        # For each pair of cardinalities, check no overlap
        cards = sorted(new_sums.keys())
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                if new_sums[cards[i]] & new_sums[cards[j]]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            sums_by_card = new_sums
            chosen.append(x)
    
    # Return as list of Python ints
    return [int(x) for x in chosen]
