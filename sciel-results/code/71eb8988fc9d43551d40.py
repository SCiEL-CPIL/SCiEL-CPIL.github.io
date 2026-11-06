import time
import random

N = 512

def construct():
    """
    Construct a large subset of [1, N] with no 4-term arithmetic progression.
    Uses bitset-based conflict checking for speed, enabling many randomized restarts.
    """
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    best_size = 0
    best_set = []
    
    # Precompute masks for faster bit operations if needed, 
    # but direct bit shifting is usually fast enough for N=512.
    
    def _can_add_bitset(bits, x):
        """
        Check if adding x (1-indexed) to the set represented by bits (integer)
        creates a 4-term AP.
        bits has bit i set if (i+1) is in the set.
        x is 1-indexed.
        """
        # x is the 4th term: a, a+d, a+2d, x => a=x-3d
        # d can be 1 to (x-1)//3
        # We need to check if (x-3d, x-2d, x-d) are all in the set.
        # Let's iterate d.
        # Optimization: We can check ranges.
        
        # 1. x is 4th term: a, a+d, a+2d, x
        # a = x - 3d
        # Indices in bits: a-1, a+d-1, a+2d-1
        # a-1 = x - 3d - 1
        # a+d-1 = x - 2d - 1
        # a+2d-1 = x - d - 1
        # We need bits & (1 << (x-3d-1)) & (1 << (x-2d-1)) & (1 << (x-d-1))
        # This is equivalent to checking if the bits at those positions are set.
        
        # Instead of individual checks, we can potentially use bit tricks,
        # but for N=512, simple loops with bit checks are very fast in Python 
        # if we avoid list overhead.
        
        # Case 1: x is 4th term
        for d in range(1, (x - 1) // 3 + 1):
            if (bits >> (x - 3 * d - 1)) & 1 and \
               (bits >> (x - 2 * d - 1)) & 1 and \
               (bits >> (x - d - 1)) & 1:
                return False

        # Case 2: x is 3rd term: a, a+d, x, x+d
        # a = x - 2d
        # Indices: a-1, a+d-1, x+d-1
        # a-1 = x - 2d - 1
        # a+d-1 = x - d - 1
        # x+d-1
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (bits >> (x - 2 * d - 1)) & 1 and \
               (bits >> (x - d - 1)) & 1 and \
               (bits >> (x + d - 1)) & 1:
                return False

        # Case 3: x is 2nd term: a, x, x+d, x+2d
        # a = x - d
        # Indices: a-1, x-1, x+d-1, x+2d-1
        # a-1 = x - d - 1
        # x-1
        # x+d-1
        # x+2d-1
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (bits >> (x - d - 1)) & 1 and \
               (bits >> (x - 1)) & 1 and \
               (bits >> (x + d - 1)) & 1 and \
               (bits >> (x + 2 * d - 1)) & 1:
                return False

        # Case 4: x is 1st term: x, x+d, x+2d, x+3d
        # Indices: x-1, x+d-1, x+2d-1, x+3d-1
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (bits >> (x - 1)) & 1 and \
               (bits >> (x + d - 1)) & 1 and \
               (bits >> (x + 2 * d - 1)) & 1 and \
               (bits >> (x + 3 * d - 1)) & 1:
                return False

        return True

    # Main Loop: Randomized Greedy with Restarts
    # We will run multiple iterations. Each iteration does a greedy pass.
    # We use different random orderings to find different local optima.
    
    while time.time() < deadline:
        bits = 0
        current_size = 0
        
        # Generate a random order for greedy selection
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy Phase
        for x in order:
            if _can_add_bitset(bits, x):
                bits |= (1 << (x - 1))
                current_size += 1
        
        # Simple Local Search: Try to add any remaining elements
        # Since we did a full pass, no single element can be added alone?
        # Wait, the greedy pass adds elements if they don't conflict with *current* set.
        # If we skip an element because it conflicts, adding other elements later 
        # won't remove the conflict. So the set is maximal w.r.t single additions.
        
        # However, we might be able to swap elements to increase size?
        # Swapping 1 for 1 doesn't increase size.
        # We want to maximize size.
        # If the set is maximal, we can't add any element.
        # To get a larger set, we need to remove some elements to allow adding more.
        # This is a local search problem.
        
        # Given the time constraint and the fact that greedy is fast,
        # the primary gain comes from finding a "better" greedy order.
        # Let's do a few simple local search steps if time permits.
        
        improved = True
        # Limit local search iterations to avoid getting stuck in one bad seed
        while improved and time.time() < deadline:
            improved = False
            
            # Try to find a swap that increases size?
            # Actually, standard local search for max independent set on this graph:
            # Try to remove a random element y and see if we can add multiple elements?
            # Or just try to add elements that were previously blocked.
            
            # Let's try a simple kick: remove a random element, then try to greedily add back.
            # This is a form of Simulated Annealing / Random Restart Local Search.
            
            # Pick a random element to remove
            if current_size == 0:
                break
                
            # Get list of set bits
            # Efficient way to get indices of set bits in a small integer
            # Or just iterate 1..N
            elements = [i for i in range(1, N + 1) if (bits >> (i - 1)) & 1]
            if not elements:
                break
                
            y = random.choice(elements)
            
            # Remove y
            bits ^= (1 << (y - 1))
            current_size -= 1
            
            # Try to add y back or other elements?
            # Let's try to add y back. If it fails, we've lost y.
            # Better: Try to add ANY element that is not in the set, in random order.
            # This might add y or something else.
            
            # To be efficient, we just do a quick pass over a subset of candidates
            # or all candidates.
            # Since N=512, checking all candidates is fast.
            
            added_any = False
            # We want to see if we can grow the set after removing y.
            # We'll try to add elements in a random order.
            # If we add more than 1 element, we improved.
            # If we add 1 element (y), we are back to same size (waste of time).
            # If we add 0, we are worse.
            
            # Actually, a better local search:
            # Try to swap y with some x not in set.
            # If x can be added after removing y, and x != y, size stays same.
            # This doesn't help maximize size directly unless it enables further growth.
            
            # Let's stick to the "Kick and Refill" strategy.
            # Remove y. Then try to add elements from the complement in random order.
            # If we end up with size > current_size (which is old_size - 1), we improved.
            
            # We track the best bits/size during this process.
            temp_bits = bits
            temp_size = current_size
            
            # Try to add elements
            candidates = [i for i in range(1, N + 1) if not (temp_bits >> (i - 1)) & 1]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                if _can_add_bitset(temp_bits, x):
                    temp_bits |= (1 << (x - 1))
                    temp_size += 1
            
            # If we grew the set, accept the change
            if temp_size > current_size:
                bits = temp_bits
                current_size = temp_size
                improved = True
            else:
                # Revert (we already modified bits to remove y, 
                # but if temp_size <= current_size, we should revert to state before removal?
                # No, we modified bits to remove y.
                # If we don't accept the growth, we should restore y?
                # If we restore y, we are back to original state.
                # If we don't restore y, we have a smaller set.
                # Let's just revert to the state before the kick if no improvement.
                bits |= (1 << (y - 1))
                current_size += 1
                # And break local search for this seed to avoid wasting time?
                # Or just continue with different kicks.
                # Let's break to save time for new restarts.
                break
                
        if current_size > best_size:
            best_size = current_size
            best_set = [i for i in range(1, N + 1) if (bits >> (i - 1)) & 1]
            
    return sorted(best_set)

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Size: {len(s)}")
    # Verify no 4-AP
    set_s = set(s)
    valid = True
    for a in set_s:
        for d in range(1, (N - a) // 3 + 1):
            if a+d in set_s and a+2*d in set_s and a+3*d in set_s:
                valid = False
                print(f"Found 4-AP: {a}, {a+d}, {a+2*d}, {a+3*d}")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
