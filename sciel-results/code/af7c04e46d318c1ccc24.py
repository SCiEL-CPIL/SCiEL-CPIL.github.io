import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    MASK = (1 << 201) - 1
    
    def is_valid(bits):
        return not (bits & (1 << 200))
        
    def add_to_bits(bits, x):
        return (bits | (bits << x)) & MASK

    def greedy_construct(order):
        bits = 1  # bit 0 is set
        selected = []
        for x in order:
            new_bits = add_to_bits(bits, x)
            if is_valid(new_bits):
                bits = new_bits
                selected.append(x)
        return selected, bits

    def local_search_add(selected, bits):
        """Try to add back excluded elements greedily."""
        selected_set = set(selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in excluded[:]:
                if time.time() > deadline:
                    break
                new_bits = add_to_bits(bits, x)
                if is_valid(new_bits):
                    selected.append(x)
                    selected_set.add(x)
                    bits = new_bits
                    excluded.remove(x)
                    improved = True
        return selected, bits

    def local_search_swap(selected, bits):
        """Try to swap elements to increase cardinality."""
        selected_set = set(selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each selected element
            for i, x in enumerate(selected[:]):
                if time.time() > deadline:
                    break
                # Remove x
                # Recalculate bits without x? 
                # It's easier to just check if adding a new element y to (bits without x) is valid.
                # But recalculating bits without x is O(N). 
                # Alternative: Pre-compute prefix/suffix? Or just re-run greedy on the rest?
                # Given N=100, re-running is fast enough if done sparsely.
                
                # Let's try a simpler 1-for-1 swap first
                # If we remove x, can we add y > x? No, y is just any excluded.
                # Actually, 1-for-1 swap doesn't change cardinality. We want 1-for-2 or more.
                
                # Let's stick to 1-for-2 logic:
                # Remove x.
                # Try to add as many excluded elements as possible.
                # If count_added > 1, it's a win.
                
                # To do this efficiently, we need bits without x.
                # We can approximate by just trying to add elements to the current set minus x.
                # Let's create a temporary set of selected without x.
                temp_selected = [y for j, y in enumerate(selected) if j != i]
                temp_bits = 1
                for y in temp_selected:
                    temp_bits = add_to_bits(temp_bits, y)
                
                # Now try to add excluded elements to temp_bits
                added_count = 0
                new_selected = list(temp_selected)
                new_bits = temp_bits
                for y in excluded:
                    if time.time() > deadline:
                        break
                    if y in new_selected:
                        continue
                    new_temp_bits = add_to_bits(new_bits, y)
                    if is_valid(new_temp_bits):
                        new_selected.append(y)
                        new_bits = new_temp_bits
                        added_count += 1
                
                if added_count > 1:
                    # We removed 1, added > 1. Net gain.
                    selected = new_selected
                    bits = new_bits
                    selected_set = set(selected)
                    excluded = [x for x in range(1, 101) if x not in selected_set]
                    improved = True
                    # Break out to restart the while loop with new state
                    break
            
            # Also try simple 1-for-1 if we want to maximize flexibility? 
            # No, cardinality is the metric. 1-for-1 is neutral.
            # But 1-for-1 might enable future 1-for-2? 
            # The "improved" flag ensures we loop if we made a 1-for-2.
            
            # If no 1-for-2 found, try to add any single missing element (greedy add)
            # This is covered by local_search_add, but let's integrate it here or call it after.
            
        return selected, bits

    # 1. Generate initial candidates
    best_selected = []
    best_bits = 1
    best_size = 0
    
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
    ]
    
    rng = random.Random(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        orders.append(order)
        
    for order in orders:
        if time.time() > deadline:
            break
        sel, bits = greedy_construct(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_bits = bits
            
    # 2. Local Search on the best candidate
    if best_selected:
        # First pass: Greedy Add
        best_selected, best_bits = local_search_add(best_selected, best_bits)
        
        # Second pass: Swaps (1-for-2)
        best_selected, best_bits = local_search_swap(best_selected, best_bits)
        
        # Third pass: Greedy Add again (in case swaps opened up space)
        best_selected, best_bits = local_search_add(best_selected, best_bits)

    # Fallback
    if not best_selected:
        best_selected = [100]
        
    return sorted(best_selected)
