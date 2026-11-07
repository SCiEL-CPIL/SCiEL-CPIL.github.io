import random

def construct():
    n = 12
    num_sets = 40
    max_mask = 1 << n
    
    def get_comparable_count(mask, chosen_list):
        count = 0
        for c in chosen_list:
            if (mask & c) == mask or (mask & c) == c:
                count += 1
        return count

    best_score = -1
    best_chosen = []
    
    # Run multiple trials
    for _ in range(50):
        # Initialize with a random chain
        # A chain is a sequence of sets where each is a subset of the next.
        # We can generate a random chain by picking a random order of elements
        # and adding them one by one.
        order = list(range(n))
        random.shuffle(order)
        
        current = 0
        chain = [0]
        for i in order:
            current |= (1 << i)
            chain.append(current)
            
        # We have a chain of length 13.
        # We need 40 sets.
        # Start with this chain? Or a subset of it?
        # If we start with the whole chain, we have 13 sets.
        # We need 27 more.
        
        chosen = set(chain)
        # To speed up, we can remove some from the chain if we want to optimize, 
        # but starting with a full chain is a strong core.
        # However, maybe a shorter core allows more "branching"?
        # Let's start with the full chain.
        
        # Greedy expansion
        # We need to add 40 - len(chosen) sets.
        # Candidates are all masks not in chosen.
        # To make it faster, we can just iterate over all masks and pick the best.
        
        for _ in range(num_sets - len(chosen)):
            best_next = -1
            best_mask = -1
            for m in range(max_mask):
                if m in chosen:
                    continue
                score = get_comparable_count(m, list(chosen))
                if score > best_next:
                    best_next = score
                    best_mask = m
            
            chosen.add(best_mask)
            
        # Evaluate final set
        # Count total comparable pairs
        chosen_list = list(chosen)
        total_pairs = 0
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                a = chosen_list[i]
                b = chosen_list[j]
                if (a & b) == a or (a & b) == b:
                    total_pairs += 1
        
        if total_pairs > best_score:
            best_score = total_pairs
            best_chosen = chosen_list

    # Convert best_chosen to list of lists
    result = []
    for mask in best_chosen:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
