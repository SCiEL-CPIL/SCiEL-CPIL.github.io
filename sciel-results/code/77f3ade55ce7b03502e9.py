import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    # Generate all k-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    best_family = []
    start_time = time.time()
    max_time = 50.0 # Leave buffer for overhead
    time_func = time.time
    
    # Precompute intersection counts for heuristic ordering
    # score[s] = number of other sets that intersect with s
    scores = [0] * len(all_sets)
    for i in range(len(all_sets)):
        s1 = all_sets[i]
        for j in range(i + 1, len(all_sets)):
            s2 = all_sets[j]
            if s1 & s2:
                scores[i] += 1
                scores[j] += 1
    
    # Create a list of indices sorted by score (descending)
    # We will shuffle within groups of similar scores? Or just sort?
    # If we sort strictly, we lose randomness.
    # We can sort by (score + random_noise)
    
    while time_func() - start_time < max_time:
        # Create a shuffled list of indices, then sort by score with some randomness
        indices = list(range(len(all_sets)))
        random.shuffle(indices)
        # Add noise to score to break ties and introduce randomness
        noisy_scores = [scores[i] + random.random() for i in indices]
        # Sort indices by noisy_scores descending
        # We want to pick high score (central) sets first?
        # Let's try descending.
        indices.sort(key=lambda i: noisy_scores[i], reverse=True)
        ordered_sets = [all_sets[i] for i in indices]
        
        family = []
        pair_intersections = [] # List of (a_mask, b_mask, inter_mask)
        
        for s in all_sets:
            valid = True
            # Check against all existing pairs
            for a, b, i in pair_intersections:
                if (a & s == i) and (b & s == i):
                    valid = False
                    break
            
            if valid:
                family.append(s)
                # Update pair_intersections with new pairs involving s
                for a in family[:-1]: # All previous elements
                    i = a & s
                    pair_intersections.append((a, s, i))
        
        # Local Search: Try to swap one set for another to allow more additions
        # This is expensive, so we do it only if we have time
        if time_func() - start_time < max_time - 2.0:
            F = family[:]
            # Get sets not in F
            F_set = set(F)
            rejected = [s for s in all_sets if s not in F_set]
            
            # Try to remove one set from F and add a rejected set
            # We try to remove the set that "blocks" the most rejected sets?
            # Simplified: Randomly try a few swaps
            for _ in range(10):
                if time_func() - start_time > max_time - 1.0:
                    break
                if not F:
                    break
                a_idx = random.randint(0, len(F) - 1)
                a = F[a_idx]
                # Find a rejected set b that is compatible with F \ {a}
                # We need to check if b forms a sunflower with any pair in F \ {a}
                # And also if b forms a sunflower with a? No, a is removed.
                # But we need to check pairs involving b and other elements in F.
                
                # Rebuild pairs for F \ {a} is expensive.
                # Instead, check b against all pairs in F that do NOT involve a.
                # And check if b conflicts with a? No, a is gone.
                
                # We need to construct the new family F' = F - {a} + {b}
                # And check if it's valid.
                # Then try to add more sets to F'.
                
                # This is getting complex. Let's skip complex local search.
                pass
        
        if len(family) > len(best_family):
            best_family = family
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
