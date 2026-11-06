"""Rewritten program for B_3[1] set construction."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    n_max = 4096
    max_sum = 3 * n_max  # 12288

    def make_sums_array(selected):
        """Build boolean array of all triple sums."""
        sums = np.zeros(max_sum + 1, dtype=bool)
        k = len(selected)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums[selected[i] + selected[j] + selected[l]] = True
        return sums

    def can_add_fast(x, selected, sums_arr):
        """Check if x can be added using boolean array for O(1) lookups."""
        # 3x
        if sums_arr[3 * x]:
            return False
        # 2x + s
        for s in selected:
            if sums_arr[2 * x + s]:
                return False
        # x + s + t
        k = len(selected)
        for i in range(k):
            s = selected[i]
            for j in range(i, k):
                t = selected[j]
                if sums_arr[x + s + t]:
                    return False
        return True

    def add_x_fast(x, selected, sums_arr):
        """Add x and update sums array."""
        selected.append(x)
        sums_arr[3 * x] = True
        for s in selected[:-1]:
            sums_arr[2 * x + s] = True
        k = len(selected)
        for i in range(k - 1):
            s = selected[i]
            for j in range(i, k - 1):
                t = selected[j]
                sums_arr[x + s + t] = True

    def greedy_fill(selected, order, deadline):
        """Greedy fill in given order."""
        selected = list(selected)
        sums_arr = make_sums_array(selected)
        existing = set(selected)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in existing:
                continue
            if can_add_fast(x, selected, sums_arr):
                add_x_fast(x, selected, sums_arr)
                existing.add(x)
        
        return selected

    def local_search(selected, deadline):
        """Try to improve by swapping elements for smaller ones."""
        if time.time() > deadline - 2:
            return selected
        
        selected = list(selected)
        sums_arr = make_sums_array(selected)
        existing = set(selected)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(len(selected)):
                if time.time() > deadline - 2:
                    break
                orig = selected[idx]
                # Remove orig
                selected_no = selected[:idx] + selected[idx+1:]
                sums_no = make_sums_array(selected_no)
                
                # Find smallest replacement
                best_new = None
                for new_x in range(1, n_max + 1):
                    if new_x in existing and new_x != orig:
                        continue
                    if can_add_fast(new_x, selected_no, sums_no):
                        if best_new is None or new_x < best_new:
                            best_new = new_x
                        if new_x <= orig:
                            break
                
                if best_new is not None and best_new != orig:
                    selected[idx] = best_new
                    existing.discard(orig)
                    existing.add(best_new)
                    sums_arr = make_sums_array(selected)
                    improved = True
                    break
        
        return selected

    best_set = []
    best_len = 0

    # Strategy 1: Powers of 4 skeleton
    base4 = [4**i for i in range(10) if 4**i <= n_max]
    order1 = list(range(1, n_max + 1))
    s1 = greedy_fill(base4, order1, deadline)
    if len(s1) > best_len:
        best_len = len(s1)
        best_set = s1[:]

    if time.time() > deadline - 5:
        return best_set[:80]

    # Strategy 2: Forward greedy from empty
    s2 = greedy_fill([], order1, deadline)
    if len(s2) > best_len:
        best_len = len(s2)
        best_set = s2[:]

    if time.time() > deadline - 5:
        return best_set[:80]

    # Strategy 3: Backward greedy from empty
    order3 = list(range(n_max, 0, -1))
    s3 = greedy_fill([], order3, deadline)
    if len(s3) > best_len:
        best_len = len(s3)
        best_set = s3[:]

    if time.time() > deadline - 5:
        return best_set[:80]

    # Local search on best
    if time.time() < deadline - 3:
        improved = local_search(best_set, deadline)
        if len(improved) > best_len:
            best_len = len(improved)
            best_set = improved

    # Randomized restarts
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 3:
            break
        
        # Random skeleton with geometric spacing
        skeleton_size = random.randint(4, 10)
        # Use geometric progression with random base
        base_val = random.randint(2, 6)
        skeleton = []
        val = 1
        for _ in range(skeleton_size):
            if val > n_max:
                break
            skeleton.append(val)
            val *= base_val
        
        if len(skeleton) < 3:
            continue
        
        # Verify skeleton
        valid = True
        sk_sums = set()
        k = len(skeleton)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s_val = skeleton[i] + skeleton[j] + skeleton[l]
                    if s_val in sk_sums:
                        valid = False
                        break
                    sk_sums.add(s_val)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Random order for filling
        if random.random() < 0.5:
            order = list(range(1, n_max + 1))
        else:
            order = list(range(n_max, 0, -1))
        
        result = greedy_fill(skeleton, order, deadline)
        if len(result) > best_len:
            best_len = len(result)
            best_set = result[:]

    return best_set[:80]
# EVOLVE-BLOCK-END
