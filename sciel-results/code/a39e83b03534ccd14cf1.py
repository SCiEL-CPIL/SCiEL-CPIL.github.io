"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)
    
    # Precompute sum_to_idx for O(1) lookup
    sum_to_idx = {s: i for i, s in enumerate(target_sums)}

    def get_counts(A_set):
        """Compute representation counts for target sums given set A."""
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = list(A_set)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            s = 2 * ai
            if s in sum_to_idx:
                counts[sum_to_idx[s]] += 1
            for j in range(i + 1, m):
                aj = A_list[j]
                s = ai + aj
                if s in sum_to_idx:
                    counts[sum_to_idx[s]] += 1
        return counts

    def calculate_peak(counts):
        """Return the maximum multiplicity in the target interval.
        Returns a large penalty if coverage is incomplete."""
        if np.any(counts == 0):
            return 1000 + np.sum(counts == 0)
        return np.max(counts)

    def initial_greedy_solution():
        """
        Construct a valid solution using greedy set cover.
        This ensures we start with a valid set, avoiding the 'invalid penalty' region
        which often traps SA.
        """
        A = set()
        # Start with a reasonable random seed to break symmetry
        # We need to cover [128, 256].
        # A random set of ~30-40 elements usually covers most, but we fill greedily.
        
        # To make it faster and more robust, we start with a random sparse set
        # and add elements that cover the most currently uncovered sums.
        
        # Random initialization for diversity
        initial_size = random.randint(20, 40)
        A = set(random.sample(range(129), initial_size))
        
        counts = get_counts(A)
        uncovered = np.where(counts == 0)[0]
        
        # Greedy fill
        candidates = [x for x in range(129) if x not in A]
        random.shuffle(candidates)
        
        # We might need multiple passes if greedy choice is bad, 
        # but usually one pass of "add element that covers most uncovered" works.
        
        # To be safe and fast, we just iterate through candidates and add if it helps.
        # A more sophisticated greedy would pick the best candidate each time.
        # Given the small domain (129), we can afford a slightly smarter greedy.
        
        for _ in range(len(candidates)):
            if not np.any(counts == 0):
                break
            
            best_x = -1
            best_gain = 0
            best_new_counts = None
            
            # Sample a subset of candidates to keep it fast, or just iterate all if time permits
            # Since n=128 is small, iterating all 129 is cheap.
            for x in candidates:
                if x in A:
                    continue
                # Simulate adding x
                # Instead of full recompute, we can just check coverage improvement roughly?
                # No, let's just do full recompute for the few best candidates?
                # Actually, get_counts is O(k^2). k~50. 2500 ops. 129 candidates. 300k ops per pass.
                # 128 sums.
                
                # Let's just do a simple greedy: pick random candidate, if it reduces uncovered count, keep it.
                # This is slower than optimal greedy but simpler.
                # Let's try the "best improvement" greedy for the first few steps, then random.
                
                pass # Skip complex greedy for now, use random fill with check
                
            # Simple random fill with immediate check
            # Pick a random candidate
            x = random.choice(candidates)
            A.add(x)
            new_counts = get_counts(A)
            
            if np.sum(new_counts == 0) >= np.sum(counts == 0):
                # No improvement or worse, remove
                A.remove(x)
                # Try a few more random ones
                for _ in range(5):
                    if not np.any(counts == 0):
                        break
                    x2 = random.choice(candidates)
                    A.add(x2)
                    new_counts2 = get_counts(A)
                    if np.sum(new_counts2 == 0) < np.sum(counts == 0):
                        counts = new_counts2
                        candidates.remove(x2)
                        break
                    else:
                        A.remove(x2)
            else:
                counts = new_counts
                candidates.remove(x)
                
        return A, calculate_peak(counts)

    # 1. Get a valid starting point
    best_A, best_peak = initial_greedy_solution()
    current_A = set(best_A)
    current_counts = get_counts(current_A)
    current_peak = best_peak
    
    # If still invalid (unlikely), fallback to full set
    if current_peak > 100:
        current_A = set(range(129))
        current_counts = get_counts(current_A)
        current_peak = calculate_peak(current_counts)
        best_A = set(current_A)
        best_peak = current_peak

    # 2. Local Search: Min-Peak Removal and Swap
    # Strategy:
    # a) Try to remove an element if it doesn't break coverage (counts > 1 for all affected sums) 
    #    and reduces or maintains peak.
    # b) Try to swap an element (remove x, add y) if it reduces peak.
    
    T_start = 5.0
    T_end = 0.01
    alpha = 0.999
    T = T_start
    
    steps = 0
    max_steps = 100000
    
    # Precompute neighbors for faster swap evaluation? 
    # A swap (x, y) changes counts by:
    # -2*x, -x-z for z in A\{x}
    # +2*y, +y+z for z in A\{x}
    
    def try_improve():
        nonlocal current_A, current_counts, current_peak, best_A, best_peak
        
        # Attempt 1: Removal
        # If we remove x, sums involving x lose 1 count (or 2 for 2x).
        # We can only remove x if all sums involving x have count > 1 (for 2x) or > 0 (for x+y) 
        # AND the removal doesn't increase the peak.
        
        # To be efficient, we check candidates for removal.
        # Elements involved in high-peak sums are bad candidates for removal? 
        # Actually, elements that contribute to the *peak* sums are the ones we might want to remove 
        # IF the count drops below the peak.
        
        # Let's identify sums with max count.
        max_count = np.max(current_counts)
        peak_sums = np.where(current_counts == max_count)[0]
        
        # Find elements that contribute to these peak sums.
        # If we remove an element x, and x is part of a pair summing to a peak sum, 
        # that peak sum's count decreases.
        
        # Simple heuristic: Try removing random elements.
        # If valid and peak doesn't increase, keep it.
        
        A_list = list(current_A)
        random.shuffle(A_list)
        
        for x in A_list:
            if len(current_A) <= 10:
                break
            
            # Simulate removal
            new_A = set(current_A)
            new_A.remove(x)
            
            # Efficient count update for removal?
            # It's easier to just recompute if k is small.
            # k is ~30-50. Recompute is O(k^2) ~ 2500.
            # 129 elements to check. 129 * 2500 = 320k ops. Very fast.
            
            new_counts = get_counts(new_A)
            
            if np.any(new_counts == 0):
                continue # Invalid
            
            new_peak = np.max(new_counts)
            
            if new_peak < current_peak:
                # Strict improvement
                current_A = new_A
                current_counts = new_counts
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(current_A)
                return True # Found an improvement, restart search loop
            
            elif new_peak == current_peak:
                # Neutral move, might help escape local minima? 
                # Usually we prefer to keep set small. 
                # Let's accept with some probability or if it reduces variance?
                # For now, accept to potentially open up swap paths.
                if random.random() < 0.1:
                    current_A = new_A
                    current_counts = new_counts
                    # Peak same, so best_peak unchanged unless it was better before? 
                    # best_peak tracks min peak.
            
        # Attempt 2: Swap
        # Remove x, Add y.
        # We want to reduce peak.
        # Let's try swapping elements that contribute to peak sums with random others.
        
        # Identify elements in peak sums
        # This is complex to map directly. Let's just do random swaps.
        
        A_list = list(current_A)
        if len(A_list) == 0:
            return False
            
        x = random.choice(A_list)
        candidates = [y for y in range(129) if y not in current_A]
        if not candidates:
            return False
        y = random.choice(candidates)
        
        new_A = set(current_A)
        new_A.remove(x)
        new_A.add(y)
        
        new_counts = get_counts(new_A)
        
        if np.any(new_counts == 0):
            return False
        
        new_peak = np.max(new_counts)
        
        if new_peak < current_peak:
            current_A = new_A
            current_counts = new_counts
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
            return True
        
        return False

    # Main Loop
    no_improve_count = 0
    while time.time() < deadline - 2.0 and T > T_end:
        steps += 1
        
        if steps % 100 == 0:
            T *= alpha
            
        # Try a batch of moves
        improved = False
        for _ in range(10):
            if try_improve():
                improved = True
                break
                
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck for a while, try a more aggressive random restart or perturbation
        if no_improve_count > 100:
            # Perturb: Randomly swap 2-3 elements
            for _ in range(random.randint(2, 5)):
                A_list = list(current_A)
                if len(A_list) < 2:
                    break
                x1 = random.choice(A_list)
                x2 = random.choice([y for y in range(129) if y not in current_A])
                current_A.remove(x1)
                current_A.add(x2)
            
            current_counts = get_counts(current_A)
            
            if np.any(current_counts == 0):
                # Invalid, revert to best
                current_A = set(best_A)
                current_counts = get_counts(current_A)
                current_peak = best_peak
            else:
                current_peak = calculate_peak(current_counts)
            
            no_improve_count = 0

    # Final Verification
    final_counts = get_counts(best_A)
    if np.any(final_counts == 0):
        return list(range(64, 129))
        
    return list(best_A)
# EVOLVE-BLOCK-END
