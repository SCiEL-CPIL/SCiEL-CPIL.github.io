"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 85s deadline to be safe within 90s limit
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute indices for target sums
    # sum -> index in target_sums, or -1 if out of range
    sum_to_idx = np.full(2 * n + 1, -1, dtype=np.int32)
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def get_counts_from_set(A_set):
        """Compute representation counts for target sums given set A from scratch."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = list(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            # Pair (ai, ai)
            s = 2 * ai
            idx = sum_to_idx[s]
            if idx >= 0:
                counts[idx] += 1
            # Pairs (ai, aj) for j > i
            for j in range(i + 1, m):
                aj = A_list[j]
                s = ai + aj
                idx = sum_to_idx[s]
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval."""
        if np.any(counts == 0):
            # Invalid if any sum is not covered.
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    def initial_solution():
        """Create a random valid solution."""
        # Start with a medium size set
        size = random.randint(30, 50)
        A = set(random.sample(range(129), size))
        
        # Ensure validity by greedy fill if needed
        counts = get_counts_from_set(A)
        if np.any(counts == 0):
             uncovered_mask = (counts == 0)
             uncovered_count = np.sum(uncovered_mask)
             candidates = [x for x in range(129) if x not in A]
             random.shuffle(candidates)
             
             for x in candidates:
                 if uncovered_count == 0:
                     break
                 # Check if x helps using fast update
                 # Temporarily add x
                 A.add(x)
                 # Update counts
                 for y in list(A):
                     if y == x: continue
                     s = x + y
                     idx = sum_to_idx[s]
                     if idx >= 0:
                         counts[idx] += 1
                 # Also handle (x, x)
                 s = 2 * x
                 idx = sum_to_idx[s]
                 if idx >= 0:
                     counts[idx] += 1
                 
                 new_uncovered_count = np.sum(counts == 0)
                 if new_uncovered_count < uncovered_count:
                     uncovered_count = new_uncovered_count
                 else:
                     # Revert
                     for y in list(A):
                         if y == x: continue
                         s = x + y
                         idx = sum_to_idx[s]
                         if idx >= 0:
                             counts[idx] -= 1
                     s = 2 * x
                     idx = sum_to_idx[s]
                     if idx >= 0:
                         counts[idx] -= 1
                     A.remove(x)
        
        return A, calculate_peak(counts)

    # Simulated Annealing with Incremental Updates
    best_A, best_peak = initial_solution()
    current_A = set(best_A)
    current_counts = get_counts_from_set(current_A)
    current_peak = best_peak
    
    # Try a few random restarts to find a good starting point
    for _ in range(10):
        A_try, peak_try = initial_solution()
        if peak_try < current_peak:
            current_A = A_try
            current_counts = get_counts_from_set(current_A)
            current_peak = peak_try
            best_A = set(current_A)
            best_peak = current_peak
        if time.time() > deadline - 5.0:
            break

    # SA Parameters
    T_start = 5.0
    T_end = 0.01
    # Calculate alpha to cool down in a reasonable number of steps
    # We don't know steps in advance, so use a fixed alpha per step
    alpha = 0.9995 
    
    current_cost = current_peak
    
    T = T_start
    steps = 0
    
    # Helper to update counts when adding x to A
    def add_to_counts(A, x, counts):
        # Pairs (x, y) for y in A, y != x
        for y in A:
            if y == x:
                continue
            s = x + y
            idx = sum_to_idx[s]
            if idx >= 0:
                counts[idx] += 1
        # Pair (x, x)
        s = 2 * x
        idx = sum_to_idx[s]
        if idx >= 0:
            counts[idx] += 1

    # Helper to update counts when removing x from A
    def remove_from_counts(A, x, counts):
        # Pairs (x, y) for y in A, y != x
        for y in A:
            if y == x:
                continue
            s = x + y
            idx = sum_to_idx[s]
            if idx >= 0:
                counts[idx] -= 1
        # Pair (x, x)
        s = 2 * x
        idx = sum_to_idx[s]
        if idx >= 0:
            counts[idx] -= 1

    while time.time() < deadline - 2.0 and T > T_end:
        steps += 1
        
        # Generate a move
        move_type = random.random()
        
        # We need to be careful with set mutation during iteration or while checking
        # We will simulate the move, calculate cost, then commit or revert.
        
        if move_type < 0.4: # Add
            candidates = [x for x in range(129) if x not in current_A]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            # Apply move
            current_A.add(x)
            add_to_counts(current_A, x, current_counts)
            new_cost = calculate_peak(current_counts)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0:
                current_cost = new_cost
                if current_cost < best_peak:
                    best_peak = current_cost
                    best_A = set(current_A)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                else:
                    # Revert
                    remove_from_counts(current_A, x, current_counts)
                    current_A.remove(x)
                    
        elif move_type < 0.7: # Remove
            if len(current_A) <= 10:
                continue
            x = random.choice(list(current_A))
            
            # Apply move
            remove_from_counts(current_A, x, current_counts)
            current_A.remove(x)
            
            new_cost = calculate_peak(current_counts)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0:
                current_cost = new_cost
                if current_cost < best_peak:
                    best_peak = current_cost
                    best_A = set(current_A)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                else:
                    # Revert
                    current_A.add(x)
                    add_to_counts(current_A, x, current_counts)
        else: # Swap
            if len(current_A) < 2:
                continue
            x = random.choice(list(current_A))
            candidates = [y for y in range(129) if y not in current_A]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Apply move: Remove x, Add y
            remove_from_counts(current_A, x, current_counts)
            current_A.remove(x)
            current_A.add(y)
            add_to_counts(current_A, y, current_counts)
            
            new_cost = calculate_peak(current_counts)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0:
                current_cost = new_cost
                if current_cost < best_peak:
                    best_peak = current_cost
                    best_A = set(current_A)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                else:
                    # Revert: Remove y, Add x
                    remove_from_counts(current_A, y, current_counts)
                    current_A.remove(y)
                    current_A.add(x)
                    add_to_counts(current_A, x, current_counts)
        
        T *= alpha

    # Final check
    if current_cost < best_peak:
        best_peak = current_cost
        best_A = set(current_A)
        
    # Verify best_A
    final_counts = get_counts_from_set(best_A)
    if np.any(final_counts == 0):
        # Fallback to a known valid set if SA failed completely
        # A simple valid set: [64, ..., 128] covers [128, 256]
        # Max multiplicity for 192 is 32 (64+128, 65+127... 96+96). Wait.
        # Let's use a denser set for fallback.
        return list(range(64, 129))
        
    return list(best_A)
# EVOLVE-BLOCK-END
