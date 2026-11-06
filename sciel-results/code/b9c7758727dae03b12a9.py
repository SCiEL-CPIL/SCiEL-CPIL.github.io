import time
import random
import numpy as np

def _can_add_bitset(bits, x):
    """
    Check if adding element x (1-indexed) to the set represented by bits 
    creates a 4-term arithmetic progression.
    bits is a numpy array of uint8, where bits[i] is 1 if (i+1) is in the set.
    x is 1-indexed (1..512).
    """
    idx = x - 1  # 0-indexed index for x
    
    # A 4-AP is a, a+d, a+2d, a+3d.
    # If we add x, x can be the 1st, 2nd, 3rd, or 4th term.
    
    # Case 1: x is the 1st term (a = x)
    # We need x+d, x+2d, x+3d in the set.
    # d >= 1 such that x+3d <= 512 => d <= (512-x)//3
    max_d1 = (512 - x) // 3
    if max_d1 > 0:
        # d ranges from 1 to max_d1
        # Indices for x+d: idx+1 to idx+max_d1
        # Indices for x+2d: idx+2 to idx+2*max_d1
        # Indices for x+3d: idx+3 to idx+3*max_d1
        # We need to check if there exists d such that bits[idx+d] & bits[idx+2d] & bits[idx+3d] is 1.
        # This is a bit tricky to vectorize directly without loops if max_d1 is large, 
        # but max_d1 is at most 170.
        # A faster way: 
        # For each d, check bits[idx+d] AND bits[idx+2d] AND bits[idx+3d].
        # We can create arrays of indices.
        ds = np.arange(1, max_d1 + 1)
        # Check if any d satisfies the condition
        # bits[idx + ds] & bits[idx + 2*ds] & bits[idx + 3*ds]
        if np.any(bits[idx + ds] & bits[idx + 2*ds] & bits[idx + 3*ds]):
            return False
            
    # Case 2: x is the 2nd term (a+d = x => a = x-d)
    # We need x-d, x, x+d, x+2d in the set.
    # x is added. So we need x-d, x+d, x+2d in the set.
    # d >= 1 such that x-d >= 1 => d <= x-1
    # x+2d <= 512 => d <= (512-x)//2
    max_d2 = min(x - 1, (512 - x) // 2)
    if max_d2 > 0:
        ds = np.arange(1, max_d2 + 1)
        if np.any(bits[idx - ds] & bits[idx + ds] & bits[idx + 2*ds]):
            return False

    # Case 3: x is the 3rd term (a+2d = x => a = x-2d)
    # We need x-2d, x-d, x, x+d in the set.
    # x is added. So we need x-2d, x-d, x+d in the set.
    # d >= 1 such that x-2d >= 1 => d <= (x-1)//2
    # x+d <= 512 => d <= 512-x
    max_d3 = min((x - 1) // 2, 512 - x)
    if max_d3 > 0:
        ds = np.arange(1, max_d3 + 1)
        if np.any(bits[idx - 2*ds] & bits[idx - ds] & bits[idx + ds]):
            return False

    # Case 4: x is the 4th term (a+3d = x => a = x-3d)
    # We need x-3d, x-2d, x-d, x in the set.
    # x is added. So we need x-3d, x-2d, x-d in the set.
    # d >= 1 such that x-3d >= 1 => d <= (x-1)//3
    max_d4 = (x - 1) // 3
    if max_d4 > 0:
        ds = np.arange(1, max_d4 + 1)
        if np.any(bits[idx - ds] & bits[idx - 2*ds] & bits[idx - 3*ds]):
            return False

    return True

def _is_4ap_free_bitset(bits):
    """
    Check if the set represented by bits is 4-AP free.
    This is used for validation, but in the main loop we rely on _can_add_bitset.
    """
    # For efficiency, we don't actually need to call this often if _can_add is correct.
    # But for safety in the final step:
    for x in range(1, 513):
        if bits[x-1]:
            # Temporarily remove x, check if adding it back is safe? No, that's circular.
            # Just check all 4-APs ending at x?
            # Actually, if we built the set using _can_add_bitset, it should be free.
            # We can do a sparse check:
            # Iterate through all x in set.
            # For each x, check if it forms a 4-AP with other elements.
            # This is O(N^2) or O(N^3) worst case. 
            # Given N=512, a full check is:
            # for a in 1..512:
            #   for d in 1..(512-a)//3:
            #     if bits[a-1] & bits[a+d-1] & bits[a+2d-1] & bits[a+3d-1]: return False
            # This is roughly sum(512/3) ~ 85k operations. Very fast in Python if optimized, 
            # but let's use numpy for the final validation just in case.
            pass
    
    # Efficient validation using numpy
    # We want to find any a, d such that bits[a-1], bits[a+d-1], bits[a+2d-1], bits[a+3d-1] are all 1.
    # This is a 4-sum-like problem on a boolean array.
    # Given the small size, a simple nested loop in Python might be faster than complex numpy broadcasting 
    # due to overhead, but let's try a vectorized approach for a few d values?
    # Actually, the main loop ensures validity incrementally. 
    # We will trust the incremental check but do a final sanity check.
    
    # Simple Python loop for final check (fast enough for 512)
    s = set(i+1 for i, b in enumerate(bits) if b)
    for x in s:
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - d) in s and (x - 2*d) in s and (x - 3*d) in s:
                return False
    return True

def construct():
    deadline = time.time() + 55.0  # Safety margin
    N = 512
    bits = np.zeros(N, dtype=np.uint8)
    
    # 1. Greedy Initialization
    # Try to add elements in a random order to get a diverse start
    order = list(range(1, N + 1))
    random.shuffle(order)
    
    for x in order:
        if _can_add_bitset(bits, x):
            bits[x-1] = 1
            
    current_size = np.sum(bits)
    best_bits = bits.copy()
    best_size = current_size
    
    # 2. Simulated Annealing / Local Search
    # Temperature starts high, decays.
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.99
    
    # We will use a mix of:
    # - Add: Try to add a random element not in set. (Size +1)
    # - Swap: Try to remove a random element from set and add a random element not in set. (Size 0 or +1/-1 depending on validity)
    # - Kick: Remove k random elements to escape local minima. (Size -k)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Determine move type
        r = random.random()
        
        if r < 0.7:
            # Try to Add
            candidates = [x for x in range(1, N + 1) if bits[x-1] == 0]
            if candidates:
                x = random.choice(candidates)
                if _can_add_bitset(bits, x):
                    bits[x-1] = 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_bits = bits.copy()
                    # Accept add immediately (it's an improvement)
                else:
                    # Reject
                    pass
        elif r < 0.9:
            # Try to Swap
            # Remove y from set, add x to set.
            # We need to ensure the set remains 4-AP free.
            # Strategy: 
            # 1. Pick y in set.
            # 2. Remove y.
            # 3. Pick x not in set (and x != y).
            # 4. Check if adding x to (set - {y}) is valid.
            # Note: Removing y might "unblock" x.
            
            in_set = [x for x in range(1, N + 1) if bits[x-1] == 1]
            if not in_set:
                continue
                
            y = random.choice(in_set)
            # Temporarily remove y
            bits[y-1] = 0
            temp_size = current_size - 1
            
            candidates = [x for x in range(1, N + 1) if bits[x-1] == 0]
            if candidates:
                x = random.choice(candidates)
                if _can_add_bitset(bits, x):
                    bits[x-1] = 1
                    new_size = temp_size + 1
                    # Always accept swap if size doesn't decrease? 
                    # If new_size == current_size, it's a lateral move. 
                    # In SA, we might accept lateral moves to explore.
                    # But for simplicity and speed, let's only accept if size increases or stays same?
                    # If we accept lateral moves, we might wander. 
                    # Let's accept if new_size >= current_size.
                    if new_size >= current_size:
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_bits = bits.copy()
                    else:
                        # Revert
                        bits[x-1] = 0
                        bits[y-1] = 1
                        current_size = temp_size + 1
                else:
                    # Revert
                    bits[y-1] = 1
                    current_size = temp_size + 1
            else:
                bits[y-1] = 1
                current_size = temp_size + 1
        else:
            # Kick: Remove 2 random elements to escape local minimum
            # This is a "hard" reset for part of the set.
            in_set = [x for x in range(1, N + 1) if bits[x-1] == 1]
            if len(in_set) >= 2:
                # Remove 2 random elements
                y1, y2 = random.sample(in_set, 2)
                bits[y1-1] = 0
                bits[y2-1] = 0
                current_size -= 2
                
                # Try to greedily fill back up? 
                # Or just let the subsequent Add/Swap steps recover.
                # To be more effective, let's try to add 2 random valid elements immediately.
                # This is a "restart" of the local neighborhood.
                
                # Greedy fill for a few steps
                for _ in range(5):
                    candidates = [x for x in range(1, N + 1) if bits[x-1] == 0]
                    if not candidates:
                        break
                    x = random.choice(candidates)
                    if _can_add_bitset(bits, x):
                        bits[x-1] = 1
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_bits = bits.copy()
            
        # Cool down
        T *= cooling_rate
        
    # Final Polish: Try to add any remaining elements
    # This is a deterministic greedy pass to ensure we didn't miss any easy adds.
    for x in range(1, N + 1):
        if bits[x-1] == 0:
            if _can_add_bitset(bits, x):
                bits[x-1] = 1
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_bits = bits.copy()
                    
    # Return the best set found
    # Convert best_bits back to a list of integers
    result = [i + 1 for i, b in enumerate(best_bits) if b]
    
    # Final validation (just in case)
    if not _is_4ap_free_bitset(np.array([1 if i in set(result) else 0 for i in range(N)], dtype=np.uint8)):
        # Fallback to a simple greedy if something went wrong
        s = set()
        for x in range(1, N + 1):
            is_safe = True
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if (x - d) in s and (x - 2*d) in s and (x - 3*d) in s:
                    is_safe = False
                    break
            if is_safe:
                s.add(x)
        return sorted(s)
        
    return sorted(result)
