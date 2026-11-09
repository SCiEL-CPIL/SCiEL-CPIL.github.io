"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Precompute multiples for fast checking
    # divisors[a] contains list of (b, c) pairs? No, that's too big.
    # We will use vectorized checks.

    def check_add(S_mask, x):
        """
        Check if x can be added to set represented by S_mask (numpy bool array of size n+1).
        S_mask[i] is True if i is in the set.
        x is the candidate element.
        """
        if not S_mask.any():
            return True
        
        # Role 1: x is 'a' (smallest)
        # Need: for all b, c in S with x < b <= c, x does NOT divide b+c
        if x > 1:
            # Get elements in S greater than x
            # We can check if there exist b, c in S, b,c > x, such that (b+c) % x == 0
            # This is equivalent to checking if the sumset of S_{>x} intersects {k*x}
            
            # Efficient check:
            # For each k such that 2x < k*x <= 2n, check if there exist b, c in S_{>x} with b+c = k*x.
            # b+c = k*x => c = k*x - b.
            # We can iterate k and check intersection of S_mask and (k*x - S_mask).
            
            max_sum = 2 * n
            # k starts from 2 (since b,c > x => b+c > 2x, so k*x > 2x => k > 2, actually k>=2 is possible if b,c close to x? No, b>x, c>x => b+c > 2x. So k*x > 2x => k > 2. Wait.
            # If x=10, b=11, c=11, b+c=22. 22 % 10 != 0.
            # If x=10, b=15, c=15, b+c=30. 30 % 10 == 0. Here k=3.
            # So k must be at least 3?
            # b > x, c >= b > x => b+c > 2x.
            # So k*x > 2x => k > 2. So k >= 3.
            
            for k in range(3, max_sum // x + 1):
                target_sum = k * x
                if target_sum > 2 * n:
                    break
                
                # We need to find if there exist b, c in S_mask such that:
                # 1. b > x, c > x
                # 2. b <= c
                # 3. b + c = target_sum
                
                # b ranges from x+1 to target_sum // 2
                # c = target_sum - b
                # We can check:
                # Let B be the set of indices in S_mask where index > x.
                # Let C be the set of indices in S_mask where index > x.
                # We need intersection of B and (target_sum - C).
                
                # Vectorized approach:
                # Create a shifted mask?
                # Simpler: iterate b in range(x+1, target_sum//2 + 1)
                # This loop is at most n/2 iterations.
                # We can optimize by checking if any b in that range is in S_mask and (target_sum - b) is in S_mask.
                
                # Using numpy:
                # range_b = np.arange(x+1, target_sum//2 + 1)
                # valid_b = S_mask[range_b]
                # corresponding_c = target_sum - range_b
                # valid_c = S_mask[corresponding_c]
                # if np.any(valid_b & valid_c): return False
                
                # To avoid creating arrays every time, we can do a simple loop if the range is small,
                # but n=256 is small enough that even a Python loop might be okay, 
                # but let's stick to numpy for safety.
                
                # Optimization: Only check if there are at least 2 elements in S > x
                # Actually, we can just do the check.
                
                # b must be <= c, so b <= target_sum / 2.
                # Also c <= n, so target_sum - b <= n => b >= target_sum - n.
                # And b > x.
                
                min_b = max(x + 1, target_sum - n)
                max_b = min(target_sum // 2, n)
                
                if min_b <= max_b:
                    # Check if any b in [min_b, max_b] is in S and (target_sum - b) is in S
                    # We can use a bitset trick or just a loop.
                    # Given n=256, a loop of ~128 iterations is fast.
                    found = False
                    for b in range(min_b, max_b + 1):
                        if S_mask[b] and S_mask[target_sum - b]:
                            found = True
                            break
                    if found:
                        return False
        elif x == 1:
            # If x=1, need 2 elements > 1 in S
            count_greater = np.sum(S_mask[2:])
            if count_greater >= 2:
                return False

        # Role 2: x is 'b' (middle)
        # Need: for all a in S with a < x, and c in S with c >= x, a does NOT divide x+c
        # a | (x+c) => c ≡ -x (mod a)
        
        # Get a's in S < x
        a_indices = np.where(S_mask[:x])[0]
        for a in a_indices:
            if a == 0: continue # 0 is not in [1,n]
            target_res = (-x) % a
            # Check if there exists c in S, c >= x, such that c % a == target_res
            # c ranges from x to n
            # We can check this by iterating c or using a precomputed table?
            # For each a, we can check:
            # c_start = x + ((target_res - x) % a)
            # if c_start < x: c_start += a
            # Actually, find first c >= x with c % a == target_res.
            
            # c = k*a + target_res
            # We want c >= x.
            # k*a >= x - target_res
            # k >= ceil((x - target_res) / a)
            
            if x > target_res:
                k_min = (x - target_res + a - 1) // a
            else:
                k_min = 0
            
            c = k_min * a + target_res
            
            # Check if c <= n and S_mask[c] is True
            # Also need c >= x. The formula ensures c >= x if k_min was calculated correctly?
            # If x=10, a=3, target_res=2. x > target_res.
            # k_min = ceil((10-2)/3) = ceil(8/3) = 3.
            # c = 3*3 + 2 = 11. 11 >= 10. Correct.
            
            while c <= n:
                if S_mask[c]:
                    return False
                c += a

        # Role 3: x is 'c' (largest)
        # Need: for all a, b in S with a < b <= x, a does NOT divide b+x
        # a | (b+x) => b ≡ -x (mod a)
        
        # Get b's in S, b <= x
        b_indices = np.where(S_mask[:x+1])[0]
        # For each b, get a's in S, a < b
        # This is nested loop.
        # We can optimize by iterating a first?
        # For each a in S, a < x:
        #   Check if there exists b in S, a < b <= x, such that b ≡ -x (mod a)
        
        a_indices_all = np.where(S_mask[:x])[0] # a < x
        for a in a_indices_all:
            if a == 0: continue
            target_res = (-x) % a
            # Find b in S, a < b <= x, b % a == target_res
            # b = k*a + target_res
            # b > a => k*a + target_res > a
            # b <= x
            
            # First b > a with b % a == target_res
            # If target_res == 0, b is multiple of a.
            # b_min = a + 1? No.
            
            # Let's find the smallest b > a such that b % a == target_res.
            # b = k*a + target_res
            # k*a > a - target_res
            # k > (a - target_res) / a
            
            if target_res == 0:
                # b is multiple of a.
                # Smallest multiple > a is 2a.
                b_start = 2 * a
            else:
                # b = k*a + target_res
                # We want b > a.
                # If target_res < a (always true for mod),
                # k=1 gives a + target_res.
                # Is a + target_res > a? Yes.
                # So b_start = a + target_res.
                # Wait, if target_res is 0, handled above.
                b_start = a + target_res
            
            # Check if b_start <= x
            if b_start <= x:
                # Iterate b from b_start to x with step a
                for b in range(b_start, x + 1, a):
                    if S_mask[b]:
                        return False

        return True

    def is_valid_set(S_mask):
        """Check if set represented by S_mask satisfies the property."""
        # We can just try to add all elements one by one to an empty set?
        # No, that's not how it works. The property is global.
        # We can check all triples.
        # Since n=256, we can do a triple loop if needed for verification.
        S_list = np.where(S_mask)[0]
        size = len(S_list)
        for i in range(size):
            a = S_list[i]
            for j in range(i + 1, size):
                b = S_list[j]
                for k in range(j, size):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    best_mask = np.zeros(n + 1, dtype=bool)
    best_size = 0

    # Strategy 1: Start with valid tails and greedily add smaller elements
    for start_threshold in range(171, 80, -1):
        if time.time() > deadline - 5:
            break

        current_mask = np.zeros(n + 1, dtype=bool)
        current_mask[start_threshold:n+1] = True
        
        if not current_mask.any():
            continue

        if not is_valid_set(current_mask):
            continue

        if np.sum(current_mask) > best_size:
            best_mask = current_mask.copy()
            best_size = np.sum(current_mask)

        # Try to add smaller elements in descending order
        candidates = list(range(start_threshold - 1, 0, -1))
        for x in candidates:
            if time.time() > deadline - 3:
                break
            if check_add(current_mask, x):
                current_mask[x] = True
                if np.sum(current_mask) > best_size:
                    best_mask = current_mask.copy()
                    best_size = np.sum(current_mask)

    # Strategy 2: Greedy from scratch
    if time.time() < deadline - 10:
        current_mask = np.zeros(n + 1, dtype=bool)
        for x in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if check_add(current_mask, x):
                current_mask[x] = True
        if np.sum(current_mask) > best_size:
            best_mask = current_mask.copy()
            best_size = np.sum(current_mask)

    # Strategy 3: Random restarts
    if time.time() < deadline - 15:
        for restart in range(100): # Increased restarts due to speed
            if time.time() > deadline - 3:
                break

            start_threshold = random.randint(130, 180)
            current_mask = np.zeros(n + 1, dtype=bool)
            current_mask[start_threshold:n+1] = True
            
            if not is_valid_set(current_mask):
                continue

            candidates = list(range(1, start_threshold))
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > deadline - 3:
                    break
                if check_add(current_mask, x):
                    current_mask[x] = True

            if np.sum(current_mask) > best_size:
                best_mask = current_mask.copy()
                best_size = np.sum(current_mask)

    # Strategy 4: Local Search (Swap)
    if time.time() < deadline - 10 and best_size > 0:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_indices = np.where(best_mask)[0]
            for i in range(len(best_indices)):
                if time.time() > deadline - 5:
                    break
                removed = best_indices[i]
                temp_mask = best_mask.copy()
                temp_mask[removed] = False
                
                # Try to add any x not in temp_mask
                # We can iterate x from 1 to n
                for x in range(1, n + 1):
                    if temp_mask[x]:
                        continue
                    if check_add(temp_mask, x):
                        new_mask = temp_mask.copy()
                        new_mask[x] = True
                        if np.sum(new_mask) > best_size:
                            best_mask = new_mask
                            best_size = np.sum(best_mask)
                            improved = True
                            break

    # Strategy 5: Final Polish
    if time.time() < deadline - 5 and best_size > 0:
        for x in range(1, n + 1):
            if best_mask[x]:
                continue
            if check_add(best_mask, x):
                best_mask[x] = True
                best_size = np.sum(best_mask)

    # Final verification
    if best_size > 0 and not is_valid_set(best_mask):
        # Fallback
        fallback_mask = np.zeros(n + 1, dtype=bool)
        fallback_mask[171:n+1] = True
        if is_valid_set(fallback_mask):
            best_mask = fallback_mask
            best_size = np.sum(best_mask)
        else:
            # Find a valid single element
            best_mask = np.zeros(n + 1, dtype=bool)
            best_mask[256] = True
            best_size = 1

    return sorted(np.where(best_mask)[0].tolist())
# EVOLVE-BLOCK-END
