import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 256
    MAX_SUM = 256 * 10 # Safe upper bound for sums
    
    # Precompute multiples for fast divisibility check?
    # Or just use modulo.
    
    def get_sums(S):
        """Return set of all non-empty subset sums."""
        sums = set()
        n = len(S)
        # Use iterative addition
        # Start with {0}
        current_sums = {0}
        for x in S:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            current_sums.update(new_sums)
        current_sums.discard(0)
        return current_sums

    def is_valid(S):
        """Check if set S has no two distinct subset sums dividing each other."""
        sums = get_sums(S)
        if not sums:
            return True
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        # For each sum, check if it divides any larger sum
        # Optimization: if a sum is small, it has many multiples.
        # If a sum is large, it has few.
        # Iterate smaller sums first.
        for i in range(n):
            s = sorted_sums[i]
            # Check multiples of s in sorted_sums[i+1:]
            # Multiples: 2s, 3s, ...
            # We can just check if any element in sorted_sums[i+1:] is divisible by s.
            # To speed up, we can break early if s * 2 > max_sum.
            max_s = sorted_sums[-1]
            if s * 2 > max_s:
                continue
            # Check divisibility
            # Instead of iterating all j, we can iterate multiples of s
            # and check if they are in the set.
            # Since sorted_sums is sorted, we can binary search or just scan.
            # Scanning is O(N). N is small (~256).
            for j in range(i + 1, n):
                if sorted_sums[j] % s == 0:
                    return False
        return True

    def try_add(S, x):
        """Try to add x to S. Return new set if valid, else None."""
        if x in S:
            return None
        new_S = S + [x]
        if is_valid(new_S):
            return new_S
        return None

    # Strategy:
    # 1. Find a valid set of size 7 quickly.
    # 2. Use local search to try to reach size 8.
    
    best_set = []
    best_size = 0
    
    # Phase 1: Greedy construction to get a baseline
    # Sort candidates in descending order to pick large numbers first
    # Large numbers are less likely to divide smaller sums?
    # Actually, large numbers create large sums.
    # Divisibility is more likely between small numbers.
    # So picking large numbers might help avoid small sum divisibility.
    
    candidates = list(range(1, MAX_VAL + 1))
    
    # Try to build a set of size 8 directly with greedy + randomization
    for restart in range(500):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(candidates)
        current_set = []
        
        # Greedy add
        for x in candidates:
            if len(current_set) >= 8:
                break
            new_set = try_add(current_set, x)
            if new_set:
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        if best_size >= 8:
            # We found a set of size 8. Try to improve to 9?
            # Or just return it.
            # Let's try to push for 9 in remaining time.
            pass

    # Phase 2: Local Search to improve best_set
    if best_size > 0:
        current_best = best_set.copy()
        current_best_size = best_size
        
        # We will try to swap elements to allow adding more.
        # Or just try adding if possible.
        
        used = set(current_best)
        
        # Iterative improvement
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            
            # Try to add any unused element
            added = False
            # Shuffle remaining to randomize
            remaining = [x for x in range(1, MAX_VAL + 1) if x not in used]
            random.shuffle(remaining)
            
            for x in remaining:
                new_set = try_add(current_best, x)
                if new_set:
                    current_best = new_set
                    current_best_size += 1
                    used.add(x)
                    added = True
                    if current_best_size > best_size:
                        best_size = current_best_size
                        best_set = current_best.copy()
                    break # Restart loop with new set
            
            if not added:
                # Try a swap: remove one, add two?
                # Or remove one, add one different, then add another?
                # Simpler: Pick a random element to remove, and try to fill the gap
                # with a different element, then try to add a new one.
                
                if len(current_best) == 0:
                    break
                    
                # Pick random index to remove
                idx = random.randint(0, len(current_best) - 1)
                removed_elem = current_best[idx]
                temp_set = current_best[:idx] + current_best[idx+1:]
                
                # Try to add a replacement
                # We want to end up with size len(current_best) + 1
                # So we need to add 2 elements to temp_set?
                # Or add 1 to get back to size, then add 1 more.
                
                # Let's try to find any element that can be added to temp_set
                # such that we can then add another.
                
                # This is getting complex. Let's stick to simple swaps:
                # Replace one element with another, then try to add a new one.
                
                # Find a replacement for removed_elem
                # We want the new set to be valid.
                # Then try to add a new element.
                
                # Pick a random candidate for replacement
                # Exclude current best elements
                repl_candidates = [x for x in range(1, MAX_VAL + 1) if x not in used]
                if not repl_candidates:
                    break
                random.shuffle(repl_candidates)
                
                swapped = False
                for repl in repl_candidates:
                    # Add repl to temp_set
                    new_set_1 = try_add(temp_set, repl)
                    if not new_set_1:
                        continue
                    
                    # Now try to add a new element to new_set_1
                    # This new element must be different from all in new_set_1
                    used_in_new = set(new_set_1)
                    # Note: removed_elem is not in used_in_new
                    # But we might want to re-add it? No, we are trying to improve size.
                    # So we need an element not in new_set_1.
                    
                    # Try to add any element not in new_set_1
                    # To be fast, just pick a few random ones
                    add_candidates = [x for x in range(1, MAX_VAL + 1) if x not in used_in_new]
                    random.shuffle(add_candidates)
                    
                    # Limit checks to save time
                    for add_x in add_candidates[:20]:
                        final_set = try_add(new_set_1, add_x)
                        if final_set:
                            current_best = final_set
                            current_best_size += 1
                            used = set(current_best)
                            swapped = True
                            if current_best_size > best_size:
                                best_size = current_best_size
                                best_set = current_best.copy()
                            break
                    
                    if swapped:
                        break
                
                if not swapped:
                    # If no swap worked, maybe just restart with a different random set?
                    # Or break to avoid infinite loop of failures.
                    # Let's do a few more tries before giving up on this configuration.
                    # Actually, if we can't add or swap, we are stuck.
                    # Let's try to randomize the current set slightly?
                    # For now, just break to save time.
                    break

    return [int(x) for x in best_set]
