"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Greedy construction: try each candidate, accept if all suffix products are new
    values, products = [], set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        # Compute all suffix products ending at x
        tails = []
        prod = x
        for a in reversed(values):
            tails.append(prod)
            prod *= a
        tails.append(prod)  # the full sequence product
        
        # Check for collisions
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    if time.time() > deadline:
        return values
    
    # Try to improve by removing one element and re-inserting others
    # This is a local search: for each position, try removing the element
    # and then greedily re-adding the skipped elements in a different order
    
    best_values = values
    best_len = len(values)
    
    # Strategy: try removing each element and see if we can add back more
    for remove_idx in range(len(values)):
        if time.time() > deadline:
            break
        
        # Remove element at remove_idx
        trial = values[:remove_idx] + values[remove_idx+1:]
        
        # Recompute products for the trial sequence
        trial_products = set()
        # Compute all block products for trial
        for i in range(len(trial)):
            prod = 1
            for j in range(i, len(trial)):
                prod *= trial[j]
                trial_products.add(prod)
        
        # Now try to add back the removed element and any other skipped elements
        removed = values[remove_idx]
        skipped = [x for x in range(2, 129) if x not in trial and x != removed]
        
        # Try inserting removed at various positions
        improved = False
        for insert_pos in range(len(trial) + 1):
            if time.time() > deadline:
                break
            
            candidate = trial[:insert_pos] + [removed] + trial[insert_pos:]
            
            # Check validity
            valid = True
            seen_products = set()
            for i in range(len(candidate)):
                prod = 1
                for j in range(i, len(candidate)):
                    prod *= candidate[j]
                    if prod in seen_products:
                        valid = False
                        break
                    seen_products.add(prod)
                if not valid:
                    break
            
            if valid and len(candidate) > best_len:
                best_values = candidate
                best_len = len(candidate)
                improved = True
        
        # Also try adding back skipped elements greedily
        if not improved:
            trial2 = trial[:]
            trial2_products = set(trial_products)
            
            for x in sorted(skipped):
                if time.time() > deadline:
                    break
                
                # Find best position to insert x
                for insert_pos in range(len(trial2) + 1):
                    candidate = trial2[:insert_pos] + [x] + trial2[insert_pos:]
                    
                    # Quick check: compute new suffix products
                    new_prods = set()
                    prod = x
                    for a in reversed(candidate[:insert_pos]):
                        prod *= a
                        new_prods.add(prod)
                    new_prods.add(x)
                    
                    # Also need to check products that start before insert_pos and end after
                    # This is more complex, so let's do a full check if it passes the quick check
                    if not trial2_products.intersection(new_prods):
                        # Full validation
                        valid = True
                        seen_products = set()
                        for i in range(len(candidate)):
                            prod = 1
                            for j in range(i, len(candidate)):
                                prod *= candidate[j]
                                if prod in seen_products:
                                    valid = False
                                    break
                                seen_products.add(prod)
                            if not valid:
                                break
                        
                        if valid:
                            trial2 = candidate
                            trial2_products = seen_products
                            break
    
    return best_values
# EVOLVE-BLOCK-END
