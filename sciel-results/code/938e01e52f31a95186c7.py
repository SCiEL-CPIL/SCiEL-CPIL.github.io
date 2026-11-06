import time
import random
import numpy as np

def _count_sums(chosen):
    """Count representations a+b (a<=b) for all sums. Returns array indexed by sum."""
    n = len(chosen)
    if n == 0:
        return np.zeros(2001, dtype=np.int32)
    arr = np.array(chosen, dtype=np.int32)
    # All pairs i <= j
    counts = np.zeros(2001, dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            counts[s] += 1
    return counts

def _check_add(chosen_arr, x, counts):
    """Check if adding x to chosen keeps all sums <= 2. Returns True if valid."""
    n = len(chosen_arr)
    # Check 2*x
    if counts[2 * x] + 1 > 2:
        return False
    # Check x + y for all y in chosen
    sums = chosen_arr + x
    for s in sums:
        if counts[s] + 1 > 2:
            return False
    return True

def _apply_add(chosen_list, x, counts):
    """Add x to chosen and update counts."""
    chosen_list.append(x)
    arr = np.array(chosen_list, dtype=np.int32)
    n = len(arr)
    # Update counts for new sums
    for j in range(n - 1):
        s = arr[-1] + arr[j]
        counts[s] += 1
    counts[2 * x] += 1
    return arr

def _remove_element(chosen_list, x, counts):
    """Remove x from chosen and update counts."""
    chosen_list.remove(x)
    arr = np.array(chosen_list, dtype=np.int32)
    n = len(arr)
    for j in range(n):
        s = x + arr[j]
        counts[s] -= 1
    counts[2 * x] -= 1
    return arr

def _greedy_by_order(order, deadline):
    """Greedy construction in given order."""
    chosen = []
    counts = np.zeros(2001, dtype=np.int32)
    chosen_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        # Check validity
        valid = True
        if counts[2 * x] + 1 > 2:
            valid = False
        else:
            for y in chosen:
                s = x + y
                if counts[s] + 1 > 2:
                    valid = False
                    break
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            for y in chosen:
                s = x + y
                counts[s] += 1
            counts[2 * x] += 1
    return chosen, counts

def _greedy_spread(deadline, seed=42):
    """Greedy that spreads elements to minimize sum collisions."""
    rng = random.Random(seed)
    chosen = []
    counts = np.zeros(2001, dtype=np.int32)
    chosen_set = set()
    all_nums = list(range(1, 1001))
    rng.shuffle(all_nums)
    
    # First pass: try all in shuffled order
    for x in all_nums:
        if time.time() > deadline:
            break
        valid = True
        if counts[2 * x] + 1 > 2:
            valid = False
        else:
            for y in chosen:
                s = x + y
                if counts[s] + 1 > 2:
                    valid = False
                    break
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            for y in chosen:
                s = x + y
                counts[s] += 1
            counts[2 * x] += 1
    return chosen, counts

def _local_search(chosen, counts, deadline, max_iter=50000, seed=42):
    """Local search: try swaps to improve size."""
    rng = random.Random(seed)
    n = len(chosen)
    chosen_set = set(chosen)
    all_nums = set(range(1, 1001))
    available = list(all_nums - chosen_set)
    rng.shuffle(available)
    
    best_size = n
    best_chosen = list(chosen)
    best_counts = counts.copy()
    
    start = time.time()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if not available:
            break
        
        # Try to add a random available element
        x_add = available[rng.randint(0, len(available) - 1)]
        
        if x_add in chosen_set:
            continue
        
        # Check if adding x_add is valid
        valid = True
        if counts[2 * x_add] + 1 > 2:
            valid = False
        else:
            for y in chosen:
                s = x_add + y
                if counts[s] + 1 > 2:
                    valid = False
                    break
        
        if valid:
            # Add it
            chosen.append(x_add)
            chosen_set.add(x_add)
            for y in chosen:
                s = x_add + y
                counts[s] += 1
            counts[2 * x_add] += 1
            if x_add in available:
                available.remove(x_add)
            best_size = len(chosen)
            best_chosen = list(chosen)
            best_counts = counts.copy()
        else:
            # Try a swap: remove one element, add x_add
            # Find an element whose removal would make x_add valid
            removed = False
            for i in range(len(chosen)):
                x_rm = chosen[i]
                # Temporarily remove x_rm
                temp_counts = counts.copy()
                for y in chosen:
                    if y == x_rm:
                        continue
                    s = x_rm + y
                    temp_counts[s] -= 1
                temp_counts[2 * x_rm] -= 1
                
                # Check if x_add is now valid
                ok = True
                if temp_counts[2 * x_add] + 1 > 2:
                    ok = False
                else:
                    for y in chosen:
                        if y == x_rm:
                            continue
                        s = x_add + y
                        if temp_counts[s] + 1 > 2:
                            ok = False
                            break
                
                if ok:
                    # Apply swap
                    chosen.remove(x_rm)
                    chosen_set.remove(x_rm)
                    for y in chosen:
                        s = x_rm + y
                        counts[s] -= 1
                    counts[2 * x_rm] -= 1
                    
                    chosen.append(x_add)
                    chosen_set.add(x_add)
                    for y in chosen:
                        s = x_add + y
                        counts[s] += 1
                    counts[2 * x_add] += 1
                    
                    if x_rm in available:
                        pass
                    else:
                        available.append(x_rm)
                    if x_add in available:
                        available.remove(x_add)
                    
                    best_size = len(chosen)
                    best_chosen = list(chosen)
                    best_counts = counts.copy()
                    removed = True
                    break
                else:
                    # Restore
                    pass
            
            if not removed:
                # Just continue
                pass
    
    return best_chosen, best_counts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s guard, 20s margin
    
    best_size = 0
    best_chosen = []
    best_counts = np.zeros(2001, dtype=np.int32)
    
    # Strategy 1: Natural order greedy
    order1 = list(range(1, 1001))
    ch1, ct1 = _greedy_by_order(order1, deadline)
    if len(ch1) > best_size:
        best_size = len(ch1)
        best_chosen = ch1
        best_counts = ct1
    
    # Strategy 2: Reverse order
    order2 = list(range(1000, 0, -1))
    ch2, ct2 = _greedy_by_order(order2, deadline)
    if len(ch2) > best_size:
        best_size = len(ch2)
        best_chosen = ch2
        best_counts = ct2
    
    # Strategy 3: Multiple random shuffles
    for trial in range(5):
        if time.time() > deadline:
            break
        seed = 100 + trial * 13
        ch, ct = _greedy_spread(deadline, seed=seed)
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct
    
    # Local search on best
    if time.time() < deadline - 10:
        ch_ls, ct_ls = _local_search(best_chosen, best_counts, deadline, max_iter=100000, seed=42)
        if len(ch_ls) > best_size:
            best_size = len(ch_ls)
            best_chosen = ch_ls
            best_counts = ct_ls
    
    # Try a few more cycles if time permits
    for trial in range(3):
        if time.time() > deadline:
            break
        seed = 500 + trial * 7
        ch, ct = _greedy_spread(deadline, seed=seed)
        if len(ch) <= best_size:
            continue
        ch, ct = _local_search(ch, ct, deadline, max_iter=20000, seed=seed)
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct
    
    # Final polish: try to add any remaining valid elements
    if time.time() < deadline - 5:
        chosen_set = set(best_chosen)
        for x in range(1, 1001):
            if time.time() > deadline:
                break
            if x in chosen_set:
                continue
            valid = True
            if best_counts[2 * x] + 1 > 2:
                valid = False
            else:
                for y in best_chosen:
                    s = x + y
                    if best_counts[s] + 1 > 2:
                        valid = False
                        break
            if valid:
                best_chosen.append(x)
                chosen_set.add(x)
                for y in best_chosen:
                    s = x + y
                    best_counts[s] += 1
                best_counts[2 * x] += 1
                best_size = len(best_chosen)
    
    return best_chosen

# EVOLVE-BLOCK-END
