import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    
    # Bitset representation: bit i is 1 if i is in the set
    # We use 1..128, so we need bits 1..128.
    
    def is_valid_bitset(mask):
        """Check if no three distinct a,b,c in mask have all pair sums in mask."""
        # Iterate over all pairs a < b
        # If a+b is in mask, check if there exists c > b such that a+c and b+c are in mask.
        # To be efficient, we can precompute or just check on the fly.
        
        # For each a (1..N)
        for a in range(1, N + 1):
            if not (mask >> a) & 1:
                continue
            # For each b > a
            for b in range(a + 1, N + 1):
                if not (mask >> b) & 1:
                    continue
                s_ab = a + b
                if s_ab > N:
                    break
                if not (mask >> s_ab) & 1:
                    continue
                
                # If a+b is in set, we have a "potential" triple.
                # We need to find c > b such that a+c and b+c are in set.
                # Let's iterate c > b
                for c in range(b + 1, N + 1):
                    if not (mask >> c) & 1:
                        continue
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac > N or s_bc > N:
                        break
                    if (mask >> s_ac) & 1 and (mask >> s_bc) & 1:
                        return False
        return True

    def is_valid_set(s_set):
        """Wrapper for set input."""
        mask = 0
        for x in s_set:
            mask |= (1 << x)
        return is_valid_bitset(mask)

    # Initial candidate: upper half
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    random.seed(42)
    
    # Local search function using bitsets for speed
    def local_search(current_set, max_steps=5000):
        current_mask = 0
        for x in current_set:
            current_mask |= (1 << x)
        
        steps = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            improved = False
            
            # Try adding elements
            # Generate candidates not in current mask
            # To make it faster, we can just iterate 1..N and check bit
            candidates = []
            for x in range(1, N + 1):
                if not (current_mask >> x) & 1:
                    candidates.append(x)
            
            random.shuffle(candidates)
            
            for elem in candidates:
                trial_mask = current_mask | (1 << elem)
                if is_valid_bitset(trial_mask):
                    current_mask = trial_mask
                    improved = True
                    break
            
            if improved:
                continue
                
            # If no single addition works, try swaps (remove 1, add 1)
            # This helps escape local optima where adding is blocked but swapping frees space
            if current_mask == 0:
                break
            
            # Pick a random element to remove
            # We need to pick a random set bit
            # Method: generate random number, find next set bit? Or just list them?
            # Since N=128, listing is fast enough.
            current_elements = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
            if not current_elements:
                break
            to_remove = random.choice(current_elements)
            
            base_mask = current_mask & ~(1 << to_remove)
            
            # Try to add 1 element to base_mask (swap)
            candidates = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
            random.shuffle(candidates)
            
            swap_found = False
            for elem in candidates:
                trial_mask = base_mask | (1 << elem)
                if is_valid_bitset(trial_mask):
                    # Only accept if it improves size? 
                    # In a swap, size stays same. 
                    # However, it might allow future additions.
                    # To avoid infinite loops of non-improving swaps, we should be careful.
                    # Let's only accept swaps if they don't decrease size (they don't) 
                    # and maybe we should track if we've been swapping without adding.
                    # For now, accept swap to change landscape.
                    current_mask = trial_mask
                    swap_found = True
                    break
            
            if not swap_found:
                # Try 2-for-1 swap? Remove 1, add 2.
                # This increases size by 1.
                candidates = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
                random.shuffle(candidates)
                
                new_mask = base_mask
                added = 0
                for elem in candidates:
                    if added == 2:
                        break
                    trial_mask = new_mask | (1 << elem)
                    if is_valid_bitset(trial_mask):
                        new_mask = trial_mask
                        added += 1
                
                if added == 2:
                    current_mask = new_mask
                    improved = True # Size increased

        # Convert back to set
        result_set = set()
        for x in range(1, N + 1):
            if (current_mask >> x) & 1:
                result_set.add(x)
        return result_set

    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Different initial sets
        if restart == 0:
            init = set(range(N // 2 + 1, N + 1))
        elif restart == 1:
            # Try a mix: upper half plus some lower
            init = set(range(N // 2 + 1, N + 1))
            # Add random lower elements if valid
            lower_candidates = list(range(1, N // 2 + 1))
            random.shuffle(lower_candidates)
            for x in lower_candidates:
                trial = init | {x}
                if is_valid_set(trial):
                    init = trial
        else:
            # Random subset of upper half + some lower
            init = set()
            for x in range(N // 2 + 1, N + 1):
                if random.random() < 0.9: # Higher probability for upper
                    init.add(x)
            for x in range(1, N // 2 + 1):
                if random.random() < 0.4:
                    init.add(x)
            
            # Ensure validity by removing random elements
            while not is_valid_set(init):
                if len(init) == 0:
                    break
                to_remove = random.choice(list(init))
                init.remove(to_remove)
        
        # Run local search
        improved_set = local_search(init)
        
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final polish: try to add any remaining element
    remaining = set(range(1, N + 1)) - best_set
    for elem in remaining:
        if time.time() > deadline - 2:
            break
        trial = best_set | {elem}
        if is_valid_set(trial):
            best_set = trial
    
    # Ensure return type is correct
    result = list(best_set)
    result = [int(x) for x in result]
    return result
