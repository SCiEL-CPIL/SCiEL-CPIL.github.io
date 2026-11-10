import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s buffer for 60s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    all_vals = np.arange(n + 1)
    
    # Precompute for each target t in [128, 256] the list of pairs (a, b) with a <= b, a+b=t, 0<=a,b<=128
    target_pairs = []
    for i in range(num_targets):
        t = target_lo + i
        pairs = []
        a_min = max(0, t - n)
        a_max = min(t // 2, n)
        for a in range(a_min, a_max + 1):
            b = t - a
            if b >= a and b <= n:
                pairs.append((a, b))
        target_pairs.append(pairs)
    
    # Precompute pair indices for faster count computation
    # For each element x, store which (target_idx, pair_idx) it participates in
    elem_pairs = [[] for _ in range(n + 1)]
    for i in range(num_targets):
        for j, (a, b) in enumerate(target_pairs[i]):
            elem_pairs[a].append((i, j))
            if b != a:
                elem_pairs[b].append((i, j))
    
    def compute_counts_fast(freq):
        """Compute representation counts using precomputed pair indices."""
        counts = np.zeros(num_targets, dtype=np.int32)
        for x in range(n + 1):
            if freq[x]:
                for (ti, pi) in elem_pairs[x]:
                    counts[ti] += 1
        # Each pair (a,b) with a != b is counted twice (once for a, once for b)
        # Pairs (a,a) are counted once
        # We need to fix: for a != b, we want count/2, for a == a, count
        # Actually, let's recompute properly
        counts2 = np.zeros(num_targets, dtype=np.int32)
        for i in range(num_targets):
            c = 0
            for a, b in target_pairs[i]:
                if freq[a] and freq[b]:
                    c += 1
            counts2[i] = c
        return counts2
    
    def compute_counts(freq):
        """Compute representation counts for all targets [128, 256]."""
        counts = np.zeros(num_targets, dtype=np.int32)
        for i in range(num_targets):
            t = target_lo + i
            c = 0
            for a, b in target_pairs[i]:
                if freq[a] and freq[b]:
                    c += 1
            counts[i] = c
        return counts
    
    def compute_counts_incremental(freq, changed_elem, old_val, new_val):
        """Update counts when one element changes from old_val to new_val."""
        counts = np.zeros(num_targets, dtype=np.int32)
        # Start from scratch - for simplicity, just recompute
        for i in range(num_targets):
            c = 0
            for a, b in target_pairs[i]:
                if freq[a] and freq[b]:
                    c += 1
            counts[i] = c
        return counts
    
    def objective(counts):
        """Max-SAT style objective: penalty for uncovered targets + penalty for high multiplicity."""
        min_c = int(np.min(counts))
        peak_c = int(np.max(counts))
        # Heavy penalty for any uncovered target
        uncovered_penalty = max(0, 1 - min_c) * 1000
        # Soft-max penalty: sum of exp((count - threshold)) for counts above threshold
        threshold = 2
        excess = np.maximum(counts - threshold, 0)
        soft_max_penalty = float(np.sum(np.exp(excess * 0.5)))
        return uncovered_penalty + soft_max_penalty + peak_c * 0.1
    
    def simulated_annealing(A_init, max_iters=50000):
        """Simulated annealing with Max-SAT-inspired objective."""
        A = sorted(set(A_init))
        if not A:
            return A, float('inf')
        
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
        
        counts = compute_counts(freq)
        cur_min = int(np.min(counts))
        cur_peak = int(np.max(counts))
        A_set = set(A)
        cur_obj = objective(counts)
        
        # Fix invalid sets first
        if cur_min < 1:
            for _ in range(300):
                if time.time() > deadline - 2:
                    break
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0:
                    break
                g = gaps[random.randint(0, len(gaps) - 1)]
                pairs = target_pairs[g]
                if not pairs:
                    continue
                a, b = pairs[random.randint(0, len(pairs) - 1)]
                added = False
                for elem in [a, b]:
                    if elem not in A_set:
                        freq[elem] = 1
                        new_counts = compute_counts(freq)
                        new_min = int(np.min(new_counts))
                        if new_min > cur_min:
                            A_set.add(elem)
                            A.append(elem)
                            A.sort()
                            counts = new_counts
                            cur_min = new_min
                            cur_peak = int(np.max(counts))
                            cur_obj = objective(counts)
                            added = True
                            break
                        else:
                            freq[elem] = 0
                if not added:
                    # Try adding both
                    if a not in A_set and b not in A_set:
                        freq[a] = 1
                        freq[b] = 1
                        new_counts = compute_counts(freq)
                        new_min = int(np.min(new_counts))
                        if new_min > cur_min:
                            A_set.add(a)
                            A_set.add(b)
                            A.extend([a, b])
                            A.sort()
                            counts = new_counts
                            cur_min = new_min
                            cur_peak = int(np.max(counts))
                            cur_obj = objective(counts)
                        else:
                            freq[a] = 0
                            freq[b] = 0
        
        if cur_min < 1:
            return A, float('inf')
        
        # Simulated annealing phase
        T_start = 10.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
        T = T_start
        
        best_A = list(A)
        best_peak = cur_peak
        best_obj = cur_obj
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            T *= cooling
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Swap: remove x from A, add y
                if len(A) < 2:
                    continue
                x = A[random.randint(0, len(A) - 1)]
                y = random.randint(0, n)
                if y in A_set or y == x:
                    continue
                
                freq[x] = 0
                freq[y] = 1
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                new_obj = objective(new_counts)
                
                delta = new_obj - cur_obj
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    A_set.discard(x)
                    A_set.add(y)
                    A[A.index(x)] = y
                    A.sort()
                    counts = new_counts
                    cur_min = new_min
                    cur_peak = new_peak
                    cur_obj = new_obj
                    
                    if new_peak < best_peak and new_min >= 1:
                        best_peak = new_peak
                        best_A = list(A)
                        best_obj = new_obj
                else:
                    freq[x] = 1
                    freq[y] = 0
            
            elif move_type < 0.7:
                # Remove an element
                if len(A) < 3:
                    continue
                x = A[random.randint(0, len(A) - 1)]
                
                freq[x] = 0
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                new_obj = objective(new_counts)
                
                delta = new_obj - cur_obj
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    A_set.discard(x)
                    A.remove(x)
                    counts = new_counts
                    cur_min = new_min
                    cur_peak = new_peak
                    cur_obj = new_obj
                    
                    if new_peak < best_peak and new_min >= 1:
                        best_peak = new_peak
                        best_A = list(A)
                        best_obj = new_obj
                else:
                    freq[x] = 1
            
            else:
                # Add an element
                y = random.randint(0, n)
                if y in A_set:
                    continue
                
                freq[y] = 1
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                new_obj = objective(new_counts)
                
                delta = new_obj - cur_obj
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    A_set.add(y)
                    A.append(y)
                    A.sort()
                    counts = new_counts
                    cur_min = new_min
                    cur_peak = new_peak
                    cur_obj = new_obj
                else:
                    freq[y] = 0
        
        # Final polish: greedy removal of elements that don't increase peak
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            improved = False
            for x in list(A):
                freq[x] = 0
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1 and new_peak <= best_peak:
                    A_set.discard(x)
                    A.remove(x)
                    counts = new_counts
                    best_peak = new_peak
                    best_A = list(A)
                    improved = True
                    break
                else:
                    freq[x] = 1
            if not improved:
                break
        
        return best_A, best_peak
    
    best_A = None
    best_peak = float('inf')
    
    random.seed(42)
    
    # Structured initializations
    init_sets = []
    
    # Even numbers
    init_sets.append(list(range(0, n + 1, 2)))
    # Odd numbers
    init_sets.append(list(range(1, n + 1, 2)))
    # Middle range
    init_sets.append(list(range(32, 97)))
    # End range
    init_sets.append(list(range(64, n + 1)))
    # Start range
    init_sets.append(list(range(0, 65)))
    # Quarter points
    init_sets.append(list(range(0, n + 1, 4)))
    # Random sets of various sizes
    for k in [24, 28, 32, 36, 40, 44, 48]:
        for _ in range(2):
            init_sets.append(sorted(random.sample(range(n + 1), k)))
    
    for A_init in init_sets:
        if time.time() > deadline - 5:
            break
        A_ls, peak_ls = simulated_annealing(A_init)
        if peak_ls < best_peak:
            # Verify
            freq = np.zeros(n + 1, dtype=np.int32)
            for x in A_ls:
                freq[x] = 1
            counts = compute_counts(freq)
            if np.min(counts) >= 1:
                best_A = A_ls
                best_peak = peak_ls
    
    # If we didn't find a good one, fallback to a dense set
    if best_A is None:
        best_A = list(range(64, 129))
    
    return list(best_A)
