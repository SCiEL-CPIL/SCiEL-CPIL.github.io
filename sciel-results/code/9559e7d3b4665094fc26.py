import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer for 60s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute for each target t in [128, 256] the list of pairs (a, b) with a <= b, a+b=t, 0<=a,b<=128
    target_pairs = []
    for i in range(num_targets):
        t = target_lo + i
        pairs = []
        for a in range(t // 2 + 1):  # a <= b means a <= t-a => 2a <= t => a <= t//2
            b = t - a
            if 0 <= a <= n and 0 <= b <= n:
                pairs.append((a, b))
        target_pairs.append(pairs)
    
    def compute_counts_fast(A_set):
        """
        Fast computation of representation counts using direct pair enumeration.
        For each pair (a,b) in A with a<=b and a+b in [128,256], increment count.
        """
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if s < target_lo:
                    continue
                if s > target_hi:
                    break  # since A_list is sorted, further j will only increase s
                idx = s - target_lo
                counts[idx] += 1
        return counts
    
    def evaluate(A_set):
        counts = compute_counts_fast(A_set)
        min_c = int(np.min(counts))
        max_c = int(np.max(counts))
        return min_c, max_c, counts
    
    def try_fix_gaps(A_set, counts, max_fix_attempts=100):
        """
        Try to fix gaps (targets with count 0) by adding elements.
        Returns (A_set, counts, cur_min, cur_peak) or None if failed.
        """
        cur_min = int(np.min(counts))
        cur_peak = int(np.max(counts))
        
        for _ in range(max_fix_attempts):
            if time.time() > deadline - 3:
                return None
            if cur_min >= 1:
                break
            
            gaps = np.where(counts == 0)[0]
            if len(gaps) == 0:
                break
            
            # Pick a random gap
            g_idx = gaps[random.randint(0, len(gaps) - 1)]
            t_val = target_lo + g_idx
            
            # Find all pairs (a, b) with a+b = t_val
            pairs = target_pairs[g_idx]
            if not pairs:
                continue
            
            # Try to add an element from a random pair
            random.shuffle(pairs)
            found = False
            for a, b in pairs:
                # Try adding a
                if a not in A_set:
                    A_set.add(a)
                    new_counts = compute_counts_fast(A_set)
                    new_min = int(np.min(new_counts))
                    if new_min > cur_min:
                        counts = new_counts
                        cur_min = new_min
                        cur_peak = int(np.max(counts))
                        found = True
                        break
                    else:
                        A_set.discard(a)
                
                # Try adding b
                if b not in A_set:
                    A_set.add(b)
                    new_counts = compute_counts_fast(A_set)
                    new_min = int(np.min(new_counts))
                    if new_min > cur_min:
                        counts = new_counts
                        cur_min = new_min
                        cur_peak = int(np.max(counts))
                        found = True
                        break
                    else:
                        A_set.discard(b)
                
                if found:
                    break
            
            if not found:
                # Try a random swap
                if A_set:
                    a_out = random.choice(list(A_set))
                    a_in = random.randint(0, n)
                    if a_in not in A_set:
                        A_set.discard(a_out)
                        A_set.add(a_in)
                        new_counts = compute_counts_fast(A_set)
                        new_min = int(np.min(new_counts))
                        if new_min > cur_min:
                            counts = new_counts
                            cur_min = new_min
                            cur_peak = int(np.max(counts))
                        else:
                            A_set.discard(a_in)
                            A_set.add(a_out)
        
        if cur_min < 1:
            return None
        return A_set, counts, cur_min, cur_peak
    
    def local_search(A_init, max_iters=5000):
        A_set = set(A_init)
        if not A_set:
            return None, float('inf')
        
        min_c, max_c, counts = evaluate(A_set)
        
        # Try to fix gaps if any
        if min_c < 1:
            result = try_fix_gaps(A_set, counts)
            if result is None:
                return None, float('inf')
            A_set, counts, min_c, max_c = result
        
        cur_min = min_c
        cur_peak = max_c
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            if cur_min < 1:
                result = try_fix_gaps(A_set, counts)
                if result is None:
                    break
                A_set, counts, cur_min, cur_peak = result
                continue
            
            # Find target with max count
            max_targets = np.where(counts == cur_peak)[0]
            if len(max_targets) == 0:
                break
            
            t_idx = max_targets[random.randint(0, len(max_targets) - 1)]
            t_val = target_lo + t_idx
            
            # List pairs in A with a+b = t_val
            pairs = []
            for a, b in target_pairs[t_idx]:
                if a in A_set and b in A_set:
                    pairs.append((a, b))
            
            if not pairs:
                continue
            
            # Pick a random pair
            a, b = pairs[random.randint(0, len(pairs) - 1)]
            
            # Try removing a (if a != b)
            if a != b:
                A_set.discard(a)
                new_counts = compute_counts_fast(A_set)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1 and new_peak <= cur_peak:
                    counts = new_counts
                    cur_peak = new_peak
                    cur_min = new_min
                    continue
                else:
                    A_set.add(a)
            
            # Try removing b (if a != b)
            if a != b:
                A_set.discard(b)
                new_counts = compute_counts_fast(A_set)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1 and new_peak <= cur_peak:
                    counts = new_counts
                    cur_peak = new_peak
                    cur_min = new_min
                    continue
                else:
                    A_set.add(b)
            
            # Try swapping a with random x
            x = random.randint(0, n)
            if x not in A_set and x != a:
                A_set.discard(a)
                A_set.add(x)
                new_counts = compute_counts_fast(A_set)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1 and new_peak <= cur_peak:
                    counts = new_counts
                    cur_peak = new_peak
                    cur_min = new_min
                    continue
                else:
                    A_set.discard(x)
                    A_set.add(a)
            
            # Try swapping b with random x
            if a != b:
                x = random.randint(0, n)
                if x not in A_set and x != b:
                    A_set.discard(b)
                    A_set.add(x)
                    new_counts = compute_counts_fast(A_set)
                    new_min = int(np.min(new_counts))
                    new_peak = int(np.max(new_counts))
                    if new_min >= 1 and new_peak <= cur_peak:
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    else:
                        A_set.discard(x)
                        A_set.add(b)
            
            # Try a random move: add or remove a random element
            if random.random() < 0.3:
                if len(A_set) > 1:
                    a_rand = random.choice(list(A_set))
                    A_set.discard(a_rand)
                    new_counts = compute_counts_fast(A_set)
                    new_min = int(np.min(new_counts))
                    new_peak = int(np.max(new_counts))
                    if new_min >= 1 and new_peak <= cur_peak:
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                    else:
                        A_set.add(a_rand)
                else:
                    a_rand = random.randint(0, n)
                    if a_rand not in A_set:
                        A_set.add(a_rand)
                        new_counts = compute_counts_fast(A_set)
                        new_min = int(np.min(new_counts))
                        new_peak = int(np.max(new_counts))
                        if new_min >= cur_min and new_peak <= cur_peak:
                            counts = new_counts
                            cur_peak = new_peak
                            cur_min = new_min
                        else:
                            A_set.discard(a_rand)
        
        return sorted(A_set), cur_peak
    
    best_A = None
    best_peak = float('inf')
    
    random.seed(42)
    
    # Strategy 1: Start with structured sets that are known to work well
    # Arithmetic progressions with various steps
    for step in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 10:
            break
        A_ap = list(range(0, 129, step))
        A_ls, peak_ls = local_search(A_ap, max_iters=3000)
        if A_ls is not None and peak_ls < best_peak:
            freq_check = set(A_ls)
            counts_check = compute_counts_fast(freq_check)
            if int(np.min(counts_check)) >= 1:
                best_A = A_ls
                best_peak = peak_ls
    
    # Strategy 2: Random starts with various sizes
    for k in [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]:
        if time.time() > deadline - 10:
            break
        for trial in range(2):
            if time.time() > deadline - 5:
                break
            A_rand = sorted(random.sample(range(n + 1), k))
            A_ls, peak_ls = local_search(A_rand, max_iters=2000)
            if A_ls is not None and peak_ls < best_peak:
                freq_check = set(A_ls)
                counts_check = compute_counts_fast(freq_check)
                if int(np.min(counts_check)) >= 1:
                    best_A = A_ls
                    best_peak = peak_ls
    
    # If we found a good solution, do additional refinement
    if best_A is not None and time.time() < deadline - 5:
        A_ls, peak_ls = local_search(best_A, max_iters=5000)
        if A_ls is not None and peak_ls < best_peak:
            freq_check = set(A_ls)
            counts_check = compute_counts_fast(freq_check)
            if int(np.min(counts_check)) >= 1:
                best_A = A_ls
                best_peak = peak_ls
    
    # Fallback: use a known valid set
    if best_A is None:
        # All numbers from 64 to 128 should cover [128, 256]
        best_A = list(range(64, 129))
    
    return list(best_A)
