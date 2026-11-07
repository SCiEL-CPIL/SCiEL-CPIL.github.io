import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s margin
    
    N = 128
    # Initial set: Upper half
    initial_set = set(range(N // 2 + 1, N + 1))
    
    def to_bitset(s):
        b = 0
        for x in s:
            b |= (1 << x)
        return b

    def from_bitset(b):
        return set(i for i in range(1, N + 1) if (b >> i) & 1)

    def is_valid_bit(b):
        # Optimized check:
        # For any distinct a, b, c in S, not (a+b in S and a+c in S and b+c in S)
        # Equivalent: For any a < b in S with a+b in S, there is no c in S (c!=a,b) 
        # such that a+c in S and b+c in S.
        
        elems = [i for i in range(1, N + 1) if (b >> i) & 1]
        n = len(elems)
        # Precompute sums for pairs? 
        # Just iterate pairs
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                c_val = elems[j]
                s_ab = a + c_val
                if s_ab > N:
                    continue
                if (b >> s_ab) & 1:
                    # a+c_val is in S. Check if there exists k such that a+k in S and c_val+k in S
                    # k must be distinct from a, c_val
                    # Note: k can be less than a, between a and c_val, or greater than c_val
                    # We just need to check all other elements
                    for k_idx in range(n):
                        if k_idx == i or k_idx == j:
                            continue
                        k = elems[k_idx]
                        s_ak = a + k
                        s_ck = c_val + k
                        if s_ak > N or s_ck > N:
                            continue
                        if (b >> s_ak) & 1 and (b >> s_ck) & 1:
                            return False
        return True

    best_size = len(initial_set)
    best_set = initial_set
    best_bit = to_bitset(initial_set)
    
    # Verify initial set is valid (it should be, sums > 128)
    if not is_valid_bit(best_bit):
        # Fallback to empty or single element if something goes wrong
        best_set = {1}
        best_bit = to_bitset(best_set)
        best_size = 1

    random.seed(42)
    
    while time.time() < deadline:
        current_set = set(best_set)
        current_bit = best_bit
        current_size = best_size
        
        # Local Search Phase
        # Try to improve current_set
        stuck_count = 0
        while time.time() < deadline and stuck_count < 100:
            improved = False
            # Try Add
            # Pick random element not in current_set
            while True:
                x = random.randint(1, N)
                if x not in current_set:
                    break
            
            # Check if adding x is valid
            new_bit = current_bit | (1 << x)
            if is_valid_bit(new_bit):
                current_set.add(x)
                current_bit = new_bit
                current_size += 1
                improved = True
                stuck_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    best_bit = current_bit
            else:
                # Try Swap
                # Pick random y in current_set
                if current_set:
                    y = random.choice(list(current_set))
                    # Remove y, add x
                    new_bit = (current_bit & ~(1 << y)) | (1 << x)
                    if is_valid_bit(new_bit):
                        current_set.remove(y)
                        current_set.add(x)
                        current_bit = new_bit
                        # Size unchanged
                        improved = True
                        stuck_count = 0
                        # Update best if size is same? No, we want max size.
                        # But maybe this swap leads to a better configuration for future adds.
                        # We don't update best_set here unless size increases, 
                        # but we should continue from this state.
                        # To allow exploration, we keep current_set as the working state.
                        # If we want to track the best valid state of same size, we could, 
                        # but it's not necessary for finding max size.
                else:
                    stuck_count += 1
            
            if not improved:
                stuck_count += 1
        
        # If we improved, we continue with the new best.
        # If we didn't improve in the last loop, we might be stuck.
        # We can restart with a different random perturbation.
        
        # Perturb best_set
        if best_size < N:
            # Remove 2 random elements from best_set to allow different structure
            perturbed = set(best_set)
            if len(perturbed) > 2:
                to_remove = random.sample(list(perturbed), 2)
                for r in to_remove:
                    perturbed.remove(r)
                current_set = perturbed
                current_bit = to_bitset(current_set)
                current_size = len(current_set)
            else:
                # If best is small, just random restart
                current_set = set(random.sample(range(1, N+1), random.randint(10, 40)))
                current_bit = to_bitset(current_set)
                if not is_valid_bit(current_bit):
                    # Repair: remove elements causing violations
                    # Simple repair: just start fresh with upper half
                    current_set = set(range(N // 2 + 1, N + 1))
                    current_bit = to_bitset(current_set)
                    current_size = len(current_set)
                else:
                    # Continue search from this perturbed state
                    pass
        
        # Note: The loop above updates best_set if an add succeeds.
        # The perturbation is applied to the *best* set found so far.
        # We should probably run the local search on the perturbed set.
        
        # Let's restructure the loop to be cleaner:
        # 1. Get a starting set (best or perturbed)
        # 2. Run local search on it
        # 3. Update best if improved
        
        # The logic above was a bit mixed. Let's refine.
        
        pass # The while loop above was a draft. Let's rewrite the main loop properly below.
        
    # The above while loop was just a placeholder for thinking.
    # Let's implement the actual loop correctly in the final code.
    
    # Re-implementing the loop properly for the final output
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_size = 0
    best_set = set()
    
    # Initialize with upper half
    init_set = set(range(N // 2 + 1, N + 1))
    init_bit = to_bitset(init_set)
    if is_valid_bit(init_bit):
        best_set = init_set
        best_size = len(init_set)
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Start from best set or a perturbed version
        if restart == 0:
            current_set = set(best_set)
        else:
            # Perturb best set
            current_set = set(best_set)
            # Remove k random elements
            k = random.randint(2, 5)
            if len(current_set) > k:
                to_remove = random.sample(list(current_set), k)
                for r in to_remove:
                    current_set.remove(r)
            else:
                # Random start
                current_set = set(random.sample(range(1, N+1), random.randint(20, 50)))
                # Ensure valid
                if not is_valid_bit(to_bitset(current_set)):
                    current_set = set(range(N // 2 + 1, N + 1))
        
        current_bit = to_bitset(current_set)
        current_size = len(current_set)
        
        # Local Search
        while time.time() < deadline:
            improved = False
            # Try Add
            # Sample a few candidates
            for _ in range(20):
                x = random.randint(1, N)
                if x in current_set:
                    continue
                new_bit = current_bit | (1 << x)
                if is_valid_bit(new_bit):
                    current_set.add(x)
                    current_bit = new_bit
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
                        best_bit = current_bit
                    break
            
            if improved:
                continue
            
            # Try Swap
            for _ in range(20):
                x = random.randint(1, N)
                if x in current_set:
                    continue
                y = random.choice(list(current_set))
                new_bit = (current_bit & ~(1 << y)) | (1 << x)
                if is_valid_bit(new_bit):
                    current_set.remove(y)
                    current_set.add(x)
                    current_bit = new_bit
                    # Size unchanged
                    break
            else:
                # No improvement in adds or swaps
                # Break local search and restart
                break
        
    return list(best_set)

# Wait, the code above has a `pass` and re-initialization. I will clean it up in the final block.
# The `is_valid_bit` function is the bottleneck.
# Optimization: In `is_valid_bit`, we can break early.
# Also, using `elems` list creation is $O(N)$.
# We can pass the list of elements to `is_valid_bit` to avoid recreation?
# But the set changes.
# Given N=128, `elems` creation is fast.

# One more optimization:
# In `is_valid_bit`, the inner loop checks all k.
# We can precompute `sums_in_set`? No.

# Let's refine the `is_valid_bit` to be as fast as possible.

def construct():
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 128
    
    def to_bitset(s):
        b = 0
        for x in s:
            b |= (1 << x)
        return b

    def is_valid_bit(b):
        elems = [i for i in range(1, N + 1) if (b >> i) & 1]
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                c_val = elems[j]
                s_ab = a + c_val
                if s_ab > N:
                    continue
                if (b >> s_ab) & 1:
                    # Check if there exists k distinct from a, c_val such that a+k in S and c_val+k in S
                    # k can be any element in elems
                    for k_idx in range(n):
                        if k_idx == i or k_idx == j:
                            continue
                        k = elems[k_idx]
                        s_ak = a + k
                        if s_ak > N:
                            continue
                        s_ck = c_val + k
                        if s_ck > N:
                            continue
                        if (b >> s_ak) & 1 and (b >> s_ck) & 1:
                            return False
        return True

    # Initial best
    init_set = set(range(N // 2 + 1, N + 1))
    init_bit = to_bitset(init_set)
    best_set = init_set
    best_bit = init_bit
    best_size = len(init_set)
    
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_set = set(best_set)
        else:
            # Perturb
            current_set = set(best_set)
            k = random.randint(3, 8)
            if len(current_set) > k:
                to_remove = random.sample(list(current_set), k)
                for r in to_remove:
                    current_set.remove(r)
            else:
                current_set = set(random.sample(range(1, N+1), random.randint(20, 60)))
                if not is_valid_bit(to_bitset(current_set)):
                    current_set = set(range(N // 2 + 1, N + 1))
        
        current_bit = to_bitset(current_set)
        current_size = len(current_set)
        
        # Local Search
        no_improve_count = 0
        while time.time() < deadline and no_improve_count < 50:
            improved = False
            
            # Try Add
            # Try a few random additions
            for _ in range(10):
                x = random.randint(1, N)
                if x in current_set:
                    continue
                new_bit = current_bit | (1 << x)
                if is_valid_bit(new_bit):
                    current_set.add(x)
                    current_bit = new_bit
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
                        best_bit = current_bit
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Try Swap
            for _ in range(10):
                x = random.randint(1, N)
                if x in current_set:
                    continue
                if not current_set:
                    break
                y = random.choice(list(current_set))
                new_bit = (current_bit & ~(1 << y)) | (1 << x)
                if is_valid_bit(new_bit):
                    current_set.remove(y)
                    current_set.add(x)
                    current_bit = new_bit
                    # Size unchanged, but might enable future adds
                    break
            else:
                no_improve_count += 1
        
    return list(best_set)
