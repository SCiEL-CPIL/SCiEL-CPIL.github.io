"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

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
            max_sum = 2 * n
            # k starts from 3
            for k in range(3, max_sum // x + 1):
                target_sum = k * x
                if target_sum > 2 * n:
                    break
                
                min_b = max(x + 1, target_sum - n)
                max_b = min(target_sum // 2, n)
                
                if min_b <= max_b:
                    found = False
                    for b in range(min_b, max_b + 1):
                        if S_mask[b] and S_mask[target_sum - b]:
                            found = True
                            break
                    if found:
                        return False
        elif x == 1:
            count_greater = np.sum(S_mask[2:])
            if count_greater >= 2:
                return False

        # Role 2: x is 'b' (middle)
        # Need: for all a in S with a < x, and c in S with c >= x, a does NOT divide x+c
        a_indices = np.where(S_mask[:x])[0]
        for a in a_indices:
            if a == 0: continue
            target_res = (-x) % a
            
            if x > target_res:
                k_min = (x - target_res + a - 1) // a
            else:
                k_min = 0
            
            c = k_min * a + target_res
            
            while c <= n:
                if S_mask[c]:
                    return False
                c += a

        # Role 3: x is 'c' (largest)
        # Need: for all a, b in S with a < b <= x, a does NOT divide b+x
        a_indices_all = np.where(S_mask[:x])[0]
        for a in a_indices_all:
            if a == 0: continue
            target_res = (-x) % a
            
            if target_res == 0:
                b_start = 2 * a
            else:
                b_start = a + target_res
            
            if b_start <= x:
                for b in range(b_start, x + 1, a):
                    if S_mask[b]:
                        return False

        return True

    def is_valid_set(S_mask):
        """Check if set represented by S_mask satisfies the property."""
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
    # We try a range of thresholds. 171 is known to be a good tail.
    # We also try slightly higher or lower to see if we can pack more.
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

    # Strategy 2: Greedy from scratch (ascending)
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

    # Strategy 3: Random restarts with shuffled order
    # This helps escape local optima of the greedy approach.
    if time.time() < deadline - 15:
        for restart in range(150): # Increased restarts
            if time.time() > deadline - 3:
                break

            # Randomize the starting tail slightly
            start_threshold = random.randint(165, 180)
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
    # Try to remove an element and add a different one to see if we can gain size.
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
                # We prioritize adding smaller numbers first as they are more constrained?
                # Or just iterate.
                for x in range(1, n + 1):
                    if temp_mask[x]:
                        continue
                    if check_add(temp_mask, x):
                        new_mask = temp_mask.copy()
                        new_mask[x] = True
                        # Since we removed 1 and added 1, size is same.
                        # This swap is only useful if it allows *further* additions later.
                        # So we should accept the swap if it leads to a better state.
                        # For now, let's just do a "polish" pass where we try to add anything.
                        # If we can add something, we improve.
                        # But here we swapped. So size is same.
                        # Let's just look for additions in the temp_mask state.
                        # Actually, the loop above adds x immediately.
                        # So we have a new set of same size.
                        # We should check if this new set allows adding MORE elements.
                        # This is complex. Let's stick to simple addition polish.
                        pass
                
                # Instead of swap, let's just try to add to the original best_mask in Strategy 5.
                # Swap search is tricky because we need to evaluate the *potential* of the new state.
                # Let's skip complex swap and rely on restarts.
                pass

    # Strategy 5: Final Polish
    # Try to add any missing element to the best_mask found so far.
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
            best_size = np.sum(fallback_mask)
        else:
            # Find a valid single element
            best_mask = np.zeros(n + 1, dtype=bool)
            best_mask[256] = True
            best_size = 1

    return sorted(np.where(best_mask)[0].tolist())
# EVOLVE-BLOCK-END
