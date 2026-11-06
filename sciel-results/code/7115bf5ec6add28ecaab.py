import time
import random
import numpy as np
from collections import Counter

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Use a highly optimized local search with vectorized count updates.
    2. Objective: Minimize Peak, then Sum of Squares.
    3. Use a "hard" constraint for validity (no zeros) and "soft" objective for peak/spread.
    4. Fast initialization using a structured set (e.g., arithmetic progression or dense block) 
       and then aggressive local search.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    target_min = 128
    target_max = 256
    N = 129  # Range [0, 128]
    R = target_max - target_min + 1  # 129
    
    # Pre-compute sum indices for fast lookup
    # sum_idx[s - target_min] = s
    # We need to update counts for sums in [128, 256]
    
    def compute_counts(mask):
        """
        Compute the representation counts for a given mask.
        Returns an array of length R (129) where counts[i] is the number of pairs (a,b) in A
        such that a+b = 128+i.
        """
        A = np.where(mask)[0]
        if len(A) == 0:
            return np.zeros(R, dtype=np.int32)
        
        # Efficiently compute pairwise sums
        # A is sorted. 
        # We can use convolution or just a loop if A is small, but A might be ~50-80 elements.
        # N=129, so A is at most 129.
        # O(|A|^2) is ~16000 ops, which is fast.
        
        counts = np.zeros(R, dtype=np.int32)
        # Use numpy for vectorized addition if possible, but loops in Python are slow for nested.
        # Let's use a trick: create a 2D array of sums? No, memory.
        # Just a double loop in C via numpy?
        # Actually, for |A| <= 129, a Python double loop is ~8000 iterations, which is < 1ms.
        # So we can just do it in Python for simplicity and correctness, or use a faster method.
        
        # Faster: Use the fact that we only care about sums in [128, 256].
        # For each a in A, the valid b's are those where 128-a <= b <= 256-a.
        # And b must be in A.
        
        # Let's stick to a simple but optimized Python loop for the initial computation,
        # but for local search, we will use incremental updates.
        
        # For now, let's just do the double loop. It's fast enough for initialization.
        for i in range(len(A)):
            a = A[i]
            for j in range(i, len(A)):
                b = A[j]
                s = a + b
                if target_min <= s <= target_max:
                    counts[s - target_min] += 1
        
        return counts

    def get_metrics(counts):
        peak = int(np.max(counts))
        sum_sq = float(np.sum(counts * counts))
        return peak, sum_sq

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def local_search(mask, counts, time_budget):
        """
        Perform local search to minimize peak and sum_sq.
        Uses incremental updates for counts.
        """
        # Work on copies
        mask = mask.copy()
        counts = counts.copy()
        
        best_mask = mask.copy()
        best_counts = counts.copy()
        cur_peak, cur_sum_sq = get_metrics(counts)
        best_peak = cur_peak
        best_sum_sq = cur_sum_sq
        
        T = 10.0  # Initial temperature
        cooling_rate = 0.99
        min_T = 0.1
        no_improve = 0
        max_no_improve = 500
        
        # Pre-compute list of active elements for fast access
        A_list = np.where(mask)[0].tolist()
        
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Choose a move:
            # 1. Remove an element
            # 2. Add an element
            # 3. Swap an element
            
            move_type = random.random()
            if move_type < 0.3:
                # Remove
                if len(A_list) <= 5:
                    continue
                x = random.choice(A_list)
                # Simulate removal
                new_counts = counts.copy()
                for y in A_list:
                    if y == x:
                        continue
                    s = x + y
                    if target_min <= s <= target_max:
                        new_counts[s - target_min] -= 1
                s = 2 * x
                if target_min <= s <= target_max:
                    new_counts[s - target_min] -= 1
                
                if not is_valid(new_counts):
                    no_improve += 1
                    continue
                
                new_peak, new_sum_sq = get_metrics(new_counts)
                # Always accept if valid and improves or equal? 
                # Removing an element reduces sum_sq, so it's usually good unless it breaks validity.
                # If valid, it must be an improvement in sum_sq (since counts decrease).
                # Does it improve peak? Maybe not, but sum_sq definitely decreases.
                # So accept if valid.
                
                # Apply removal
                mask[x] = False
                A_list.remove(x)
                counts = new_counts
                cur_peak, cur_sum_sq = new_peak, new_sum_sq
                
                if cur_peak < best_peak or (cur_peak == best_peak and cur_sum_sq < best_sum_sq):
                    best_peak = cur_peak
                    best_sum_sq = cur_sum_sq
                    best_mask = mask.copy()
                    best_counts = counts.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
            elif move_type < 0.6:
                # Add
                # Pick a random element not in A
                candidates = np.where(mask == False)[0]
                if len(candidates) == 0:
                    continue
                x = random.choice(candidates)
                
                # Simulate addition
                new_counts = counts.copy()
                for y in A_list:
                    s = x + y
                    if target_min <= s <= target_max:
                        new_counts[s - target_min] += 1
                s = 2 * x
                if target_min <= s <= target_max:
                    new_counts[s - target_min] += 1
                
                # Adding an element increases counts, so peak might increase.
                # We only accept if it helps reduce peak? No, adding usually increases peak.
                # But it might be necessary to cover gaps? No, we start valid.
                # So adding is generally bad for peak/sum_sq.
                # However, it might allow future removals that are better?
                # Let's only accept if it improves sum_sq? Unlikely.
                # Let's only accept if it reduces peak? Unlikely.
                # So we probably should not add elements unless we are stuck.
                # Let's skip adding for now, or only accept with very low probability if it doesn't worsen peak.
                
                new_peak, new_sum_sq = get_metrics(new_counts)
                if new_peak < cur_peak:
                    # Rare
                    mask[x] = True
                    A_list.append(x)
                    counts = new_counts
                    cur_peak, cur_sum_sq = new_peak, new_sum_sq
                    if cur_peak < best_peak or (cur_peak == best_peak and cur_sum_sq < best_sum_sq):
                        best_peak = cur_peak
                        best_sum_sq = cur_sum_sq
                        best_mask = mask.copy()
                        best_counts = counts.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            else:
                # Swap
                if len(A_list) == 0:
                    continue
                x = random.choice(A_list)
                candidates = np.where(mask == False)[0]
                if len(candidates) == 0:
                    continue
                y = random.choice(candidates)
                
                # Simulate swap: remove x, add y
                new_counts = counts.copy()
                
                # Remove x
                for a in A_list:
                    if a == x:
                        continue
                    s = x + a
                    if target_min <= s <= target_max:
                        new_counts[s - target_min] -= 1
                s = 2 * x
                if target_min <= s <= target_max:
                    new_counts[s - target_min] -= 1
                
                # Add y
                for a in A_list:
                    if a == x:
                        continue
                    s = y + a
                    if target_min <= s <= target_max:
                        new_counts[s - target_min] += 1
                s = 2 * y
                if target_min <= s <= target_max:
                    new_counts[s - target_min] += 1
                
                if not is_valid(new_counts):
                    no_improve += 1
                    continue
                
                new_peak, new_sum_sq = get_metrics(new_counts)
                
                delta_peak = new_peak - cur_peak
                delta_sum_sq = new_sum_sq - cur_sum_sq
                delta = delta_peak * 1000 + delta_sum_sq
                
                accepted = False
                if delta < 0:
                    accepted = True
                else:
                    # Simulated Annealing
                    if T > 0:
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            accepted = True
                
                if accepted:
                    # Apply swap
                    mask[x] = False
                    A_list.remove(x)
                    mask[y] = True
                    A_list.append(y)
                    counts = new_counts
                    cur_peak, cur_sum_sq = new_peak, new_sum_sq
                    
                    if cur_peak < best_peak or (cur_peak == best_peak and cur_sum_sq < best_sum_sq):
                        best_peak = cur_peak
                        best_sum_sq = cur_sum_sq
                        best_mask = mask.copy()
                        best_counts = counts.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling_rate
            
            if no_improve > max_no_improve:
                # Restart or break?
                # Let's just break for now and rely on multiple restarts.
                break
        
        return best_mask, best_counts

    # Main Loop
    best_global_peak = float('inf')
    best_global_sum_sq = float('inf')
    best_global_mask = None
    
    # Initializations
    # 1. Dense block: [0, 128] is too dense. 
    # Try a structured set like [0, 2, 4, ..., 128] (even numbers)
    # Sums of evens are evens. 128..256 evens are covered. Odds are not.
    # So we need both parities.
    
    # Let's try a random set of size ~40 and repair.
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random initialization
        size = random.randint(30, 50)
        indices = rng.choice(N, size=size, replace=False)
        mask = np.zeros(N, dtype=np.bool_)
        mask[indices] = True
        
        counts = compute_counts(mask)
        
        # Repair if invalid
        if not is_valid(counts):
            # Greedy repair
            for _ in range(100):
                if is_valid(counts):
                    break
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0:
                    break
                
                best_y = None
                best_gain = -1
                
                for y in range(N):
                    if mask[y]:
                        continue
                    gain = 0
                    for g in gaps:
                        s = target_min + g
                        a = s - y
                        if 0 <= a < N and (mask[a] or a == y):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                
                if best_y is not None and best_gain > 0:
                    mask[best_y] = True
                    counts = compute_counts(mask) # Recompute for simplicity in repair
                else:
                    break
            
            if not is_valid(counts):
                continue
        
        # Local Search
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        budget = min(8.0, time_left)
        m, c = local_search(mask, counts, time_budget=budget)
        p, ss = get_metrics(c)
        
        if p < best_global_peak or (p == best_global_peak and ss < best_global_sum_sq):
            best_global_peak = p
            best_global_sum_sq = ss
            best_global_mask = m
            
            # Print progress (optional)
            # print(f"Restart {restart}: Peak {p}, SumSq {ss:.1f}, Time {time.time()-start_time:.1f}")

    # Final Result
    if best_global_mask is None:
        # Fallback: Return a set that definitely works, even if peak is high.
        # [0, 128] works? 
        # Sums of 0..128 cover 0..256.
        # So [0, 128] is valid.
        return list(range(0, 129))
        
    result_list = np.where(best_global_mask)[0].tolist()
    
    if not result_list:
        return [128]
        
    return result_list
