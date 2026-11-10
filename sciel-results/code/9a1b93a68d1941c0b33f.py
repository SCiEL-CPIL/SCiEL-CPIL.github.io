import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute pair targets: for each (a,b) with a<=b, which target does it hit?
    # pair_target[a][b] = s - target_lo if target_lo <= a+b <= target_hi, else -1
    pair_target = np.full((n+1, n+1), -1, dtype=np.int32)
    for a in range(n+1):
        for b in range(a, n+1):
            s = a + b
            if target_lo <= s <= target_hi:
                pair_target[a][b] = s - target_lo
    
    def init_counts_from_set(A):
        """Compute counts from a set A using direct pair enumeration."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                t = pair_target[a][b]
                if t >= 0:
                    counts[t] += 1
        return counts
    
    def is_complete(counts):
        return bool(np.all(counts > 0))
    
    def peak_of(counts):
        return int(np.max(counts))
    
    def remove_element(x, counts, A_set):
        """Remove x from A, update counts. Returns new counts or None if invalid."""
        new_counts = counts.copy()
        # x pairs with all a in A where a <= x
        for a in A_set:
            if a <= x:
                t = pair_target[a][x]
                if t >= 0:
                    new_counts[t] -= 1
        return new_counts
    
    def add_element(x, counts, A_set):
        """Add x to A, update counts. Returns new counts."""
        new_counts = counts.copy()
        # x pairs with all a in A where a <= x, plus x itself
        for a in A_set:
            if a <= x:
                t = pair_target[a][x]
                if t >= 0:
                    new_counts[t] += 1
        # x pairs with itself
        t = pair_target[x][x]
        if t >= 0:
            new_counts[t] += 1
        return new_counts
    
    def swap_elements(x_out, x_in, counts, A_set):
        """Swap x_out for x_in. Returns new counts."""
        new_counts = counts.copy()
        # Remove x_out: for all a in A_set where a <= x_out
        for a in A_set:
            if a <= x_out:
                t = pair_target[a][x_out]
                if t >= 0:
                    new_counts[t] -= 1
        # Add x_in: for all a in A_set (excluding x_out) where a <= x_in, plus x_in itself
        for a in A_set:
            if a == x_out:
                continue
            if a <= x_in:
                t = pair_target[a][x_in]
                if t >= 0:
                    new_counts[t] += 1
        t = pair_target[x_in][x_in]
        if t >= 0:
            new_counts[t] += 1
        return new_counts
    
    # Try multiple structured initializations
    def structured_init_1():
        """Symmetric around 128: include elements that help cover both ends."""
        A = set()
        # Include 0 and 128 to cover extremes
        A.add(0)
        A.add(128)
        # Add elements to cover [128, 256]
        for x in range(64, 129):
            A.add(x)
        return sorted(A)
    
    def structured_init_2():
        """Dense in middle, sparse at ends."""
        A = set()
        # Cover low targets [128, 192] with low elements
        for x in range(0, 65, 2):
            A.add(x)
        # Cover high targets [192, 256] with high elements
        for x in range(96, 129):
            A.add(x)
        return sorted(A)
    
    def structured_init_3():
        """Evenly spaced."""
        A = set()
        for x in range(0, 129, 2):
            A.add(x)
        return sorted(A)
    
    def local_search(A_init, max_iter=200000):
        """Fast local search with incremental updates."""
        A = sorted(set(A_init))
        A_set = set(A)
        counts = init_counts_from_set(A)
        
        if not is_complete(counts):
            return A, peak_of(counts), counts
        
        cur_peak = peak_of(counts)
        A_list = A[:]
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try removals (most likely to improve peak)
            for x in A_list:
                new_counts = remove_element(x, counts, A_set)
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A_set.discard(x)
                        A_list = [a for a in A_list if a != x]
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
            
            if improved:
                continue
            
            if time.time() > deadline - 5:
                break
            
            # Try additions
            for x in range(n + 1):
                if x in A_set:
                    continue
                new_counts = add_element(x, counts, A_set)
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A_set.add(x)
                        A_list.append(x)
                        A_list.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
            
            if improved:
                continue
            
            if time.time() > deadline - 5:
                break
            
            # Try swaps (sampled for speed)
            found_swap = False
            num_swaps_to_try = min(20, len(A_list))
            for _ in range(num_swaps_to_try):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, len(A_list) - 1)
                x_out = A_list[idx]
                x_in = random.randint(0, n)
                if x_in in A_set or x_in == x_out:
                    continue
                
                new_counts = swap_elements(x_out, x_in, counts, A_set)
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A_set.discard(x_out)
                        A_set.add(x_in)
                        A_list[idx] = x_in
                        A_list.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        found_swap = True
                        break
            
            if not found_swap:
                break
        
        return A_list, cur_peak, counts
    
    # Run local search from multiple starts
    best_A = None
    best_peak = float('inf')
    best_counts = None
    
    initializations = [
        structured_init_1(),
        structured_init_2(),
        structured_init_3(),
        list(range(64, 129)),
        list(range(0, 129, 2)),
    ]
    
    # Add some random initializations
    random.seed(42)
    for _ in range(5):
        k = random.randint(50, 80)
        A_rand = sorted(random.sample(range(n + 1), k))
        initializations.append(A_rand)
    
    for init_A in initializations:
        if time.time() > deadline - 10:
            break
        
        A_ls, peak_ls, counts_ls = local_search(init_A)
        
        if is_complete(counts_ls) and peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
            best_counts = counts_ls
    
    # If we haven't found a good solution, try more random restarts
    if best_peak > 3:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            k = random.randint(45, 75)
            A_rand = sorted(random.sample(range(n + 1), k))
            A_ls, peak_ls, counts_ls = local_search(A_rand, max_iter=50000)
            if is_complete(counts_ls) and peak_ls < best_peak:
                best_A = A_ls
                best_peak = peak_ls
                best_counts = counts_ls
    
    # Final validation
    if best_A is None or not is_complete(best_counts):
        # Fallback to known valid solution
        return list(range(64, 129))
    
    return list(best_A)
