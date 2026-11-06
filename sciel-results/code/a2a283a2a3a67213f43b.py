import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    candidates = list(range(2, 129))
    
    # State: (values, products_set)
    # We use a beam of states.
    # Each state is (values_list, products_set)
    
    beam = [([], set())]
    beam_width = 50  # Adjust if needed
    
    for x in candidates:
        new_beam = []
        for values, products in beam:
            # Option 1: Skip x
            new_beam.append((values, products))
            
            # Option 2: Add x
            # Calculate tails
            tails = []
            prod = x
            tails.append(prod)
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            
            # Check collision
            if not products.intersection(tails):
                new_values = values + [x]
                new_products = products.union(tails)
                new_beam.append((new_values, new_products))
        
        # Prune the beam
        if len(new_beam) > beam_width:
            # We want to keep the best states.
            # Metric: Length of values is primary.
            # Secondary: Maybe sum of values? Or max value?
            # Smaller values might be better for future?
            # Let's sort by length (desc), then by sum of values (asc) to keep "lighter" sequences?
            # Or just keep the first beam_width?
            # To diversify, we should keep states with different properties.
            # Let's sort by length desc. If tie, by the set of products size? (should be same for same length).
            # Maybe by the maximum product? Smaller max product is better?
            
            # Let's try sorting by length desc, then by the last element (smaller is better?)
            # Actually, let's just keep the top beam_width by length, and if there are ties, keep a diverse subset.
            
            # Simple pruning: keep the longest. If multiple have same max length, keep a few of them.
            max_len = max(len(v) for v, p in new_beam)
            candidates_for_beam = [s for s in new_beam if len(s[0]) == max_len]
            
            # If we have more candidates than beam_width, we need to select.
            # We can select based on the "tightness" of the product set.
            # Or just random selection to allow exploration.
            
            if len(candidates_for_beam) > beam_width:
                # Use random sampling to maintain diversity
                random.shuffle(candidates_for_beam)
                new_beam = candidates_for_beam[:beam_width]
            else:
                new_beam = candidates_for_beam
                
        beam = new_beam
        
        # Check time
        if time.time() > deadline:
            break
            
    # Find the best state in the final beam
    best_state = max(beam, key=lambda s: len(s[0]))
    return best_state[0]

# Let's test this logic mentally.
# The beam search might lose good solutions if a shorter sequence leads to a much longer one later.
# But since we are adding elements one by one, a shorter sequence at step i has fewer constraints, so it *should* be able to extend at least as well as a longer one?
# Not necessarily, because the specific values matter.
# However, the greedy approach (which is effectively beam width 1 with "include if possible" bias) works well.
# Beam search with width > 1 should be better.

# Let's refine the pruning.
# If we keep only max length, we might discard a state that has slightly shorter length but "better" numbers.
# But since we are iterating through fixed candidates, "better" is hard to define.
# Let's try keeping a mix of lengths?
# No, maximizing length is the goal. A state with length L can only become length L+k.
# A state with length L-1 can become L-1+k.
# It is unlikely that L-1+k > L+k_final if the search is consistent.
# So keeping max length is a safe bet for maximizing final length.

# One issue: `products.union(tails)` creates a new set. This is expensive.
# We can optimize by checking intersection first, which we do.
# The union is only done if valid.

# Let's increase beam_width to 100 or 200 to be safe.
# And add a time check inside the loop if it's slow.

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    candidates = list(range(2, 129))
    
    # State: (values, products_set)
    beam = [([], set())]
    beam_width = 200
    
    for x in candidates:
        new_beam = []
        for values, products in beam:
            # Option 1: Skip x
            # We can optimize by not creating new tuples if we don't need to, 
            # but here we need to keep the state.
            new_beam.append((values, products))
            
            # Option 2: Add x
            tails = []
            prod = x
            tails.append(prod)
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            
            # Check collision
            if not products.intersection(tails):
                new_values = values + [x]
                new_products = products.union(tails)
                new_beam.append((new_values, new_products))
        
        # Prune the beam
        if len(new_beam) > beam_width:
            # Sort by length descending
            new_beam.sort(key=lambda s: len(s[0]), reverse=True)
            
            # Keep top beam_width
            # To ensure diversity, we can take the top half by length, and random from the rest?
            # Or just take the top beam_width.
            # Since we sorted by length, the first beam_width are the longest.
            # If there are ties in length, we might want to keep diverse ones.
            # Let's take the first beam_width.
            new_beam = new_beam[:beam_width]
            
        beam = new_beam
        
        # Check time
        if time.time() > deadline:
            break
            
    # Find the best state in the final beam
    if not beam:
        return []
    best_state = max(beam, key=lambda s: len(s[0]))
    return best_state[0]
