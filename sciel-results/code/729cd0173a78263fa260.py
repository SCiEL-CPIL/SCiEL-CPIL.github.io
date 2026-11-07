import random
import time

def construct():
    """
    Finds a large set of distinct integers in 1..256 such that:
    1. All non-empty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    
    Returns a list of integers.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s to leave buffer for 60s limit
    
    max_val = 256
    best_set = []
    best_size = 0
    
    # Precompute all numbers
    numbers = list(range(1, max_val + 1))
    
    def is_valid_set(S):
        """
        Checks if set S satisfies the conditions.
        1. All subset sums are distinct.
        2. No subset sum divides another.
        """
        if not S:
            return True
            
        # Generate all subset sums
        # Use iterative doubling
        sums = [0]
        for x in S:
            new_sums = [s + x for s in sums]
            sums.extend(new_sums)
            
        # Remove 0 (empty set sum)
        sums = [s for s in sums if s > 0]
        
        # Check for duplicates
        if len(sums) != len(set(sums)):
            return False
            
        # Check divisibility
        # Max possible sum is 30 * 256 = 7680, but realistically smaller.
        # Let's find the max sum to size the boolean array
        max_sum = max(sums)
        is_sum = [False] * (max_sum + 1)
        
        for s in sums:
            is_sum[s] = True
            
        for s in sums:
            # Check multiples of s
            multiple = s * 2
            while multiple <= max_sum:
                if is_sum[multiple]:
                    return False
                multiple += s
                
        return True

    def try_add(S, candidate):
        """
        Tries to add candidate to S. Returns new set if valid, else None.
        """
        if candidate in S:
            return None
        new_S = list(S) + [candidate]
        if is_valid_set(new_S):
            return new_S
        return None

    # Main search loop
    # We use a randomized local search with restarts
    # Since the space is small, we can afford many restarts.
    
    while time.time() < deadline:
        # Start with a random small set to bootstrap
        # Randomly pick 3-5 numbers
        initial_size = random.randint(3, 5)
        current_set = random.sample(numbers, initial_size)
        
        # If initial set is invalid, try to fix it or restart
        if not is_valid_set(current_set):
            # Simple repair: remove elements until valid
            while current_set and not is_valid_set(current_set):
                # Remove random element
                idx = random.randint(0, len(current_set) - 1)
                current_set.pop(idx)
            if not current_set:
                continue
                
        # Greedy expansion
        while time.time() < deadline:
            # Try to add a random element
            # To be more efficient, we can shuffle the remaining numbers
            remaining = [n for n in numbers if n not in current_set]
            random.shuffle(remaining)
            
            added = False
            for cand in remaining:
                new_set = try_add(current_set, cand)
                if new_set is not None:
                    current_set = new_set
                    added = True
                    break
            
            if not added:
                # Local optimum reached, try a swap or remove
                # If we can't add anything, we might be stuck.
                # Try removing one element and then adding another
                if len(current_set) > 1:
                    remove_idx = random.randint(0, len(current_set) - 1)
                    removed_elem = current_set.pop(remove_idx)
                    
                    # Now try to add something different
                    remaining_after_remove = [n for n in numbers if n not in current_set]
                    random.shuffle(remaining_after_remove)
                    
                    swapped = False
                    for cand in remaining_after_remove:
                        if cand == removed_elem:
                            continue
                        new_set = try_add(current_set, cand)
                        if new_set is not None:
                            current_set = new_set
                            swapped = True
                            break
                    
                    if not swapped:
                        # Put it back if we didn't find a better swap
                        current_set.append(removed_elem)
                        # If we are stuck, break to restart with a different seed
                        break
                else:
                    break
            
            # Update best
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                
        # If we found a new best, we might want to continue searching around it
        # But for simplicity, we restart with a fresh random seed or refine current best
        # Let's refine the current best set in the next iteration by starting from it
        # Actually, starting from scratch with random restarts is often better for escaping local optima
        # But we can also try to improve best_set directly
        
        # Alternative: Start next iteration from best_set and try to grow it
        # This might be more efficient if best_set is good.
        # Let's do a mix: 50% restart random, 50% grow from best
        
        if best_size > 0 and random.random() < 0.5:
            current_set = list(best_set)
            # Try to grow it again
            while time.time() < deadline:
                remaining = [n for n in numbers if n not in current_set]
                random.shuffle(remaining)
                added = False
                for cand in remaining:
                    new_set = try_add(current_set, cand)
                    if new_set is not None:
                        current_set = new_set
                        added = True
                        break
                if not added:
                    break
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)

    # Ensure return type is list of ints
    return [int(x) for x in best_set]

# To test locally
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
