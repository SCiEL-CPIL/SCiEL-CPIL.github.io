import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 60s limit
    
    N = 128
    FULL_MASK = (1 << (N + 1)) - 1  # Bits 1..128 set. Bit 0 unused.

    # Precompute constraints for fast validity checking
    # For each pair (a, b) with a < b and a+b <= N, 
    # compute the bitmask of 'c' values such that c > b, c+a <= N, c+b <= N.
    # If a, b, a+b are in the set, then none of these c can be in the set.
    # We store this as a list of tuples (a, b, s_ab, c_mask)
    constraints = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break
            # c must be > b
            # c + a <= N => c <= N - a
            # c + b <= N => c <= N - b
            max_c = min(N - a, N - b)
            if max_c > b:
                # Create mask for c in range (b+1, max_c]
                # Bit i represents integer i
                # We want bits b+1 to max_c set
                # Mask = (1 << (max_c + 1)) - 1 - ((1 << (b + 1)) - 1)
                c_mask = ((1 << (max_c + 1)) - 1) ^ ((1 << (b + 1)) - 1)
                if c_mask != 0:
                    constraints.append((a, b, s_ab, c_mask))

    def is_valid(mask):
        """
        Check if the set represented by mask is valid.
        A set is invalid if there exist distinct a, b, c in the set
        such that a+b, a+c, b+c are also in the set.
        """
        for a, b, s_ab, c_mask in constraints:
            # Check if a, b, and a+b are in the mask
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << s_ab)):
                # If so, check if any c in c_mask is also in the mask
                if mask & c_mask:
                    return False
        return True

    def get_elements(mask):
        """Convert mask to a list of integers."""
        return [i for i in range(1, N + 1) if (mask >> i) & 1]

    # Initial candidate: upper half
    # The set {65, ..., 128} is valid because for any a,b,c > 64, 
    # a+b > 128, so a+b is not in the set.
    best_mask = 0
    for i in range(N // 2 + 1, N + 1):
        best_mask |= (1 << i)
    
    best_size = bin(best_mask).count('1')
    
    random.seed(42)

    def local_search(init_mask, max_steps=5000):
        current_mask = init_mask
        current_size = bin(current_mask).count('1')
        steps = 0
        no_improve_count = 0
        
        while steps < max_steps and time.time() < deadline - 2:
            steps += 1
            improved = False
            
            # 1. Try adding a single element
            # Candidates are elements not in current_mask
            # We can generate a random candidate or iterate through all
            # To be efficient, we pick random candidates from the complement
            comp_mask = FULL_MASK & ~current_mask
            if comp_mask:
                # Pick a random bit from comp_mask
                # Method: shift right until we find a set bit? Or use bit_length?
                # Efficient way: 
                # 1. Get list of available bits? Too slow if done every step.
                # 2. Randomly select an index and check?
                # Let's try adding elements in a randomized order or random selection.
                # Since N=128, we can just iterate through 1..128 and check if not in mask.
                # But we want to try "promising" ones. 
                # Let's just shuffle the available bits once per "stuck" phase?
                # Actually, just picking random candidates is fine.
                
                # To pick a random element from comp_mask:
                # We can use a loop to pick a random position and check bit.
                # Or, we can precompute the list of available elements if the set is small?
                # No, just random trial.
                
                # Try a few random additions
                tried = 0
                max_tries = 20
                while tried < max_tries:
                    tried += 1
                    # Pick random x in 1..N
                    x = random.randint(1, N)
                    if (current_mask >> x) & 1:
                        continue
                    trial_mask = current_mask | (1 << x)
                    if is_valid(trial_mask):
                        current_mask = trial_mask
                        current_size += 1
                        improved = True
                        no_improve_count = 0
                        break
            
            if improved:
                continue

            no_improve_count += 1
            
            # 2. If no single addition works, try swaps
            # Strategy: Remove one element, add one or two others.
            # We want to increase size or maintain size while changing structure.
            
            # Get list of current elements
            current_elements = get_elements(current_mask)
            if not current_elements:
                break

            # Pick a random element to remove
            to_remove = random.choice(current_elements)
            base_mask = current_mask & ~(1 << to_remove)
            
            # Try to add elements to base_mask
            # We try to add 1 element (swap) or 2 elements (2-for-1)
            
            # Generate candidates for addition (elements not in base_mask)
            # We can just iterate through 1..N
            
            # First try: Add 1 element
            swap_found = False
            # Shuffle indices for random order?
            # Just try all candidates? 128 checks is fast.
            # But we want to be smart. Let's try all candidates for 1-add.
            for x in range(1, N + 1):
                if x == to_remove:
                    continue
                if (base_mask >> x) & 1:
                    continue
                trial_mask = base_mask | (1 << x)
                if is_valid(trial_mask):
                    current_mask = trial_mask
                    # Size stays same (removed 1, added 1)
                    # This helps escape local minima
                    swap_found = True
                    break
            
            if swap_found:
                # Reset no_improve slightly? No, size didn't increase.
                # But we moved in the search space.
                # Let's not reset no_improve_count fully, but allow more steps.
                continue

            # Second try: Add 2 elements (2-for-1 swap)
            # This increases size by 1.
            # We need to find x, y such that base_mask U {x, y} is valid.
            # This is O(N^2) which is 128^2 = 16384 checks. 
            # is_valid is fast, so this is feasible.
            
            # To make it faster, we can iterate x, then y.
            # We can prune: if x > y, skip? Order doesn't matter for set.
            # Let's just do nested loops.
            
            found_2_for_1 = False
            # Optimization: Only try this if we are really stuck?
            # Or always try if no 1-for-1 swap worked.
            
            # We can limit the number of pairs to check to save time
            # But 16k checks * fast is_valid is likely < 1ms.
            
            for x in range(1, N + 1):
                if x == to_remove:
                    continue
                if (base_mask >> x) & 1:
                    continue
                mask_with_x = base_mask | (1 << x)
                
                # Check if mask_with_x is "almost" valid? 
                # No, just check validity of mask_with_x | (1<<y)
                
                for y in range(x + 1, N + 1): # y > x to avoid duplicates
                    if y == to_remove:
                        continue
                    if (base_mask >> y) & 1:
                        continue
                    
                    trial_mask = mask_with_x | (1 << y)
                    if is_valid(trial_mask):
                        current_mask = trial_mask
                        current_size += 1
                        found_2_for_1 = True
                        no_improve_count = 0
                        break
                if found_2_for_1:
                    break
            
            if found_2_for_1:
                continue

            # If no improvement after trying additions and swaps
            if no_improve_count > 100:
                # Break out of this local search to try a new restart
                break

        return current_mask, current_size

    # Multiple restarts
    # We will run local search from various starting points.
    
    # Restart 0: Upper half
    m, s = local_search(best_mask)
    if s > best_size:
        best_mask = m
        best_size = s

    # Restart 1: Upper half + some lower elements (greedy fill)
    init_mask = 0
    for i in range(N // 2 + 1, N + 1):
        init_mask |= (1 << i)
    
    # Try to add lower elements greedily
    for i in range(1, N // 2 + 1):
        trial = init_mask | (1 << i)
        if is_valid(trial):
            init_mask = trial
    
    m, s = local_search(init_mask)
    if s > best_size:
        best_mask = m
        best_size = s

    # Restart 2..N: Random subsets
    # We generate random masks and clean them up to be valid, then search.
    for restart in range(2, 100):
        if time.time() > deadline - 5:
            break
        
        # Generate a random mask
        # Probability of including each element
        # Higher probability for larger numbers?
        # Let's just use 0.5
        rand_mask = 0
        for i in range(1, N + 1):
            if random.random() < 0.5:
                rand_mask |= (1 << i)
        
        # Clean up: remove elements that cause violations
        # We can do this by checking validity and removing random violators
        # Or simpler: start with empty set and add random elements if valid?
        # That's just another random search.
        
        # Better: Take rand_mask, and if invalid, remove elements until valid.
        # Which elements to remove? Those involved in conflicts.
        # This is complex. 
        # Alternative: Use the best_mask found so far as a seed, and perturb it.
        # Perturbation: Remove k random elements, add k random elements.
        
        if best_mask == 0:
            continue
            
        perturbed_mask = best_mask
        # Remove 2 random elements
        elems = get_elements(perturbed_mask)
        if len(elems) >= 2:
            to_remove = random.sample(elems, 2)
            for e in to_remove:
                perturbed_mask &= ~(1 << e)
        
        # Add 2 random elements
        comp = FULL_MASK & ~perturbed_mask
        if comp:
            # Pick 2 random elements from comp
            # We can just pick random integers and check if in comp
            added = 0
            while added < 2:
                x = random.randint(1, N)
                if (perturbed_mask >> x) & 1:
                    continue
                if (comp >> x) & 1:
                    perturbed_mask |= (1 << x)
                    added += 1
        
        # Ensure the perturbed mask is valid before starting local search
        # If not, clean it up
        if not is_valid(perturbed_mask):
            # Remove random elements until valid
            # This might be slow, so limit it
            clean_steps = 0
            while not is_valid(perturbed_mask) and clean_steps < 50:
                clean_steps += 1
                elems = get_elements(perturbed_mask)
                if not elems:
                    break
                e = random.choice(elems)
                perturbed_mask &= ~(1 << e)
            
            if not is_valid(perturbed_mask):
                # Failed to clean, skip this restart
                continue

        m, s = local_search(perturbed_mask)
        if s > best_size:
            best_mask = m
            best_size = s

    # Final polish: Try to add any remaining element to best_mask
    if time.time() < deadline - 1:
        for x in range(1, N + 1):
            if time.time() > deadline - 0.5:
                break
            if (best_mask >> x) & 1:
                continue
            trial = best_mask | (1 << x)
            if is_valid(trial):
                best_mask = trial
                best_size += 1

    # Convert best_mask to list of ints
    result = get_elements(best_mask)
    return result

# Test locally if needed
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Sample: {res[:10]}")
