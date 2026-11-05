"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute indices for target sums for faster lookup
    # sum -> index in target_sums, or -1 if out of range
    sum_to_idx = {}
    for i, s in enumerate(target_sums):
        sum_to_idx[s] = i

    def get_counts(A_set):
        """Compute representation counts for target sums given set A."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = list(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            # Pair (ai, ai)
            s = 2 * ai
            if s in sum_to_idx:
                counts[sum_to_idx[s]] += 1
            # Pairs (ai, aj) for j > i
            for j in range(i + 1, m):
                aj = A_list[j]
                s = ai + aj
                if s in sum_to_idx:
                    counts[sum_to_idx[s]] += 1
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval."""
        if np.any(counts == 0):
            # Invalid if any sum is not covered. 
            # We use a large penalty to guide search away from invalid states.
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    def initial_solution():
        """Create a random valid solution."""
        # Start with a dense set that is likely to cover everything
        # Then remove elements that are redundant to reduce peak.
        A = set(range(0, 129))
        # Randomly remove elements to make it smaller and potentially lower peak
        # We want to keep it valid.
        # Let's start with a medium size set, e.g., 40-60 elements.
        # Random subset of [0, 128]
        size = random.randint(35, 55)
        A = set(random.sample(range(129), size))
        
        # Ensure validity by adding elements if needed?
        # For SA, we can start invalid and let the penalty guide it, 
        # but starting valid is often faster.
        # Let's try to make it valid by a simple greedy fill if invalid.
        counts = get_counts(A)
        if np.any(counts == 0):
             # Greedy fill
             uncovered = [i for i in range(num_targets) if counts[i] == 0]
             candidates = [x for x in range(129) if x not in A]
             random.shuffle(candidates)
             for x in candidates:
                 if not uncovered:
                     break
                 # Check if x helps
                 # Calculate new counts efficiently? For initial setup, just recompute.
                 A.add(x)
                 counts = get_counts(A)
                 new_uncovered = [i for i in range(num_targets) if counts[i] == 0]
                 if len(new_uncovered) == len(uncovered):
                     # No improvement, remove x
                     A.remove(x)
                 else:
                     uncovered = new_uncovered
                     if not uncovered:
                         break
        return A, calculate_peak(get_counts(A))

    # Simulated Annealing
    best_A, best_peak = initial_solution()
    current_A = set(best_A)
    current_counts = get_counts(current_A)
    current_peak = best_peak
    
    # If initial solution is bad, try a few random restarts
    for _ in range(5):
        A_try, peak_try = initial_solution()
        if peak_try < current_peak:
            current_A = A_try
            current_counts = get_counts(current_A)
            current_peak = peak_try
            best_A = set(current_A)
            best_peak = current_peak
        if time.time() > deadline - 5.0:
            break

    # SA Parameters
    T_start = 5.0
    T_end = 0.01
    alpha = 0.999  # Cooling rate
    
    # We want to minimize peak.
    # Cost function: peak if valid, else 1000 + num_uncovered
    
    def cost(A_set, counts):
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return np.max(counts)
    
    current_cost = cost(current_A, current_counts)
    
    T = T_start
    steps = 0
    max_steps = 50000 # Rough estimate, time limited anyway
    
    while time.time() < deadline - 2.0 and T > T_end:
        steps += 1
        if steps % 1000 == 0:
            # Check time and update best
            if current_cost < best_peak:
                best_peak = current_cost
                best_A = set(current_A)
            T *= alpha
        
        # Generate a move
        # Types: Add, Remove, Swap
        move_type = random.random()
        
        new_A = set(current_A)
        
        if move_type < 0.3: # Add
            candidates = [x for x in range(129) if x not in new_A]
            if candidates:
                x = random.choice(candidates)
                new_A.add(x)
        elif move_type < 0.6: # Remove
            if len(new_A) > 10: # Don't get too small
                x = random.choice(list(new_A))
                new_A.remove(x)
        else: # Swap
            if len(new_A) > 1:
                x = random.choice(list(new_A))
                new_A.remove(x)
                candidates = [y for y in range(129) if y not in new_A]
                if candidates:
                    y = random.choice(candidates)
                    new_A.add(y)

        if new_A == current_A:
            continue
            
        new_counts = get_counts(new_A)
        new_cost = cost(new_A, new_counts)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Improvement
            current_A = new_A
            current_counts = new_counts
            current_cost = new_cost
            if current_cost < best_peak:
                best_peak = current_cost
                best_A = set(current_A)
        else:
            # Accept with probability
            # If cost is high (invalid), we might want to accept more often to escape local minima?
            # Standard SA: exp(-delta/T)
            # If delta is large (e.g., going from valid to invalid), T is small, so prob is low.
            # This is good.
            prob = np.exp(-delta_cost / T)
            if random.random() < prob:
                current_A = new_A
                current_counts = new_counts
                current_cost = new_cost
        
        T *= alpha

    # Final check
    if current_cost < best_peak:
        best_peak = current_cost
        best_A = set(current_A)
        
    # Verify best_A
    final_counts = get_counts(best_A)
    if np.any(final_counts == 0):
        # Fallback to a known valid set if SA failed completely (unlikely)
        return list(range(64, 129))
        
    return list(best_A)
# EVOLVE-BLOCK-END
