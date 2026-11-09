import time
import random
from itertools import combinations

def construct():
    """
    Finds a subset of [1, 192] of maximum cardinality (up to 48) such that
    no element divides the sum of any 3 distinct other elements in the subset.
    Uses a randomized local search (swap-based hill climbing) with a time limit.
    """
    deadline = time.time() + 85  # 85s limit to be safe within 90s
    
    # Helper to check if adding 'x' to 'current_set' is valid
    def is_valid_addition(x, current_set):
        # Check if x divides sum of any 3 other elements in current_set
        # Also check if any element in current_set divides sum of any 3 other elements involving x?
        # No, the condition is: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
        # So if we add x, we must check:
        # 1. Does x divide sum(a,b,c) for any a,b,c in current_set?
        # 2. Does any a in current_set divide sum(x, b, c) for any b,c in current_set?
        
        # 1. x divides sum of 3 others from current_set
        for triple in combinations(current_set, 3):
            if sum(triple) % x == 0:
                return False
        
        # 2. Any a in current_set divides sum(x, b, c)
        # Note: b and c are distinct from a and x, and b != c
        # We can iterate over a in current_set
        for a in current_set:
            # We need to check if there exist distinct b, c in current_set \ {a} such that (x + b + c) % a == 0
            # This is equivalent to checking if there exist b, c in current_set \ {a} such that b + c ≡ -x (mod a)
            # To be safe and simple, we can just iterate combinations of the rest
            # But that might be slow. Let's optimize slightly.
            # Actually, for small sets (<=48), iterating combinations is O(N^3) per check, which is ~ 48^3/6 ~ 18k operations.
            # With 192 candidates, this is feasible.
            
            # Let's just do the direct check for correctness first
            remaining = [v for v in current_set if v != a]
            # We need to check if any pair (b,c) in remaining sums to -x mod a
            # Or just check all pairs
            for b, c in combinations(remaining, 2):
                if (x + b + c) % a == 0:
                    return False
        return True

    # Initialize with a greedy approach to get a good starting point
    # Greedy: add largest numbers first
    current_set = []
    for x in range(192, 0, -1):
        if is_valid_addition(x, current_set):
            current_set.append(x)
            if len(current_set) >= 48:
                break
    
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Local Search Loop
    # We try to improve the size. If we can't add, we try to swap.
    # A swap: remove y from current_set, add x.
    # If len(current_set) < 48, we can just try to add x.
    
    while time.time() < deadline:
        if len(current_set) >= 48:
            break
            
        # Try to add a random element not in current_set
        remaining = [x for x in range(1, 193) if x not in current_set]
        if not remaining:
            break
            
        # Shuffle remaining to randomize
        random.shuffle(remaining)
        
        found_addition = False
        for x in remaining:
            if is_valid_addition(x, current_set):
                current_set.append(x)
                found_addition = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                if len(current_set) >= 48:
                    break
                # Reset remaining to check if we can add more
                remaining = [v for v in range(1, 193) if v not in current_set]
                random.shuffle(remaining)
                if not remaining:
                    break
                # Continue loop to try adding another
        
        if not found_addition:
            # Try swaps: remove one element y, try to add x
            # To make this efficient, we can pick a random y to remove and a random x to add
            # Or iterate through a few candidates
            
            # Strategy: Pick a random x from remaining. Try to find a y in current_set 
            # such that (current_set - {y}) + {x} is valid.
            
            # To avoid O(N^2) full validity checks, we can just try a few random swaps
            num_swap_attempts = 10
            for _ in range(num_swap_attempts):
                if time.time() > deadline:
                    break
                if not remaining:
                    break
                    
                x = random.choice(remaining)
                # Try removing a random y
                y_idx = random.randint(0, len(current_set) - 1)
                y = current_set[y_idx]
                
                # Create candidate set
                new_set = current_set.copy()
                new_set.remove(y)
                
                if is_valid_addition(x, new_set):
                    new_set.append(x)
                    # If size increased (it won't, since we swapped 1 for 1), 
                    # but maybe this set allows further additions later?
                    # Actually, if we just swap, size stays same. 
                    # The goal is to increase size. 
                    # So a swap is only useful if it enables a subsequent addition.
                    # This is complex. 
                    
                    # Alternative: If we can't add, maybe we should just accept the current best.
                    # But sometimes a "bad" element blocks many "good" ones.
                    # Let's try a different heuristic: 
                    # If we can't add any element, try to remove the element that "blocks" the most others?
                    # That's expensive.
                    
                    # Let's stick to: If we can't add, try to swap a random element with a random other 
                    # to see if it leads to a state where we CAN add.
                    # This is a random walk in the space of fixed-size sets.
                    
                    # If the swap is valid, accept it and restart the addition loop.
                    current_set = new_set
                    # Restart addition loop immediately
                    found_addition = False # We didn't add, we swapped. 
                    # Let's just break out of swap loop and go to next main loop iteration
                    # to try adding again
                    break
            
            # If we swapped, we loop back to try adding.
            # If we didn't swap, we are stuck. 
            # We can try a more aggressive reset: 
            # Restart greedy from a different order?
            # Or just return best_set found so far.
            
            # To ensure we don't infinite loop without progress:
            # If we didn't find an addition and didn't find a valid swap, 
            # we might be in a local optimum.
            # Let's try to restart the greedy phase with a different random seed/order 
            # if we have time.
            
            # For simplicity and robustness within time, let's just try to 
            # add elements in a different random order.
            random.shuffle(remaining)
            # If we still can't add, we might be stuck.
            # Let's break out if we haven't improved in a while?
            # No, just let the time limit handle it.
            # But to avoid wasting time on futile swaps that don't lead to additions,
            # we can limit the number of consecutive failed attempts.
            
            # Simple heuristic: If we can't add, try to remove the smallest element 
            # and see if we can add a larger one? 
            # Actually, the constraint is about divisibility. 
            # Larger numbers are "safer" as divisors because they are less likely to divide smaller sums.
            # But as summands, they create larger sums.
            # The greedy descending order is actually quite good.
            
            # If we are stuck, let's just return the best set found.
            # But maybe we can try one more thing: 
            # Check if there's any element x in [1,192] not in current_set 
            # that can be added if we remove ONE specific element y.
            # This is an O(N^2) check. 192^2 ~ 36k. 
            # For each x, for each y, check validity. 
            # Validity check is O(N^3). 36k * 18k ~ 600M operations. Too slow in Python.
            
            # So we rely on random swaps.
            # If we fail to add after shuffling, we just loop.
            # To prevent infinite looping on the same state, we can keep a history of recent states?
            # Or just accept that we might loop. The time limit will stop it.
            
            # To make the loop more productive, let's try to add elements 
            # in descending order again, but skipping those that failed.
            # Actually, the `remaining` list is already shuffled.
            
            # If we are stuck, let's just break and return best_set.
            # But maybe we can try a different starting point?
            # No, we have a deadline.
            
            # Let's add a counter for failed attempts. If > 100, break.
            # But we don't have a counter in the main loop.
            # Let's just rely on the time limit.
            
            # One improvement: If we can't add, try to remove the element 
            # that is most likely to be causing conflicts.
            # Heuristic: Remove the smallest element? 
            # Or the element that divides the most sums?
            # Let's just remove a random element and try to add a random one.
            # We already did that.
            
            # Let's break if we haven't improved in a long time?
            # No, just let it run until deadline.
            
            # To avoid pure random wandering, let's try to add elements 
            # in descending order from the remaining list.
            remaining_sorted = sorted(remaining, reverse=True)
            for x in remaining_sorted:
                if is_valid_addition(x, current_set):
                    current_set.append(x)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    if len(current_set) >= 48:
                        break
                    remaining = [v for v in range(1, 193) if v not in current_set]
                    if not remaining:
                        break
                    break # Restart the outer while loop to reshuffle or try again

    return best_set

# EVOLVE-BLOCK-END
