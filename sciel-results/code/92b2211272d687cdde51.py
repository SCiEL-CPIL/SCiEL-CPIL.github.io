import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    
    def compute_peak(A_set):
        """
        Compute the peak representation multiplicity for all target sums.
        Returns (peak, uncovered_count)
        """
        A_list = sorted(A_set)
        m = len(A_list)
        if m == 0:
            return float('inf'), 129
        
        # Use numpy for fast counting
        # A_arr[i] + A_arr[j] for i <= j
        # We can use broadcasting for small m, but loop is fine for m~20
        # To be safe and fast, we'll use a loop with numpy array increment
        
        counts = np.zeros(257, dtype=np.int32)
        A_arr = np.array(A_list, dtype=np.int32)
        
        # Optimization: if m is small, nested loop is fast enough
        # m ~ 20 -> 210 pairs. Very fast.
        for i in range(m):
            ai = A_arr[i]
            # j from i to m-1
            # Sums: ai + A_arr[i:]
            sums = ai + A_arr[i:]
            # Filter sums in range [128, 256]
            mask = (sums >= 128) & (sums <= 256)
            if np.any(mask):
                valid_sums = sums[mask]
                # Count occurrences
                # np.bincount is efficient
                bin_counts = np.bincount(valid_sums, minlength=257)
                counts += bin_counts
        
        # Check coverage
        target_counts = counts[128:257]
        uncovered = np.sum(target_counts == 0)
        
        if uncovered > 0:
            return float('inf'), int(uncovered)
            
        peak = int(np.max(target_counts))
        return peak, 0

    def fitness(A_set):
        """
        Fitness function: lower is better.
        Prioritizes coverage, then minimizes peak.
        """
        peak, uncovered = compute_peak(A_set)
        if uncovered > 0:
            return 10000 + uncovered * 100 + peak
        return peak

    def generate_sidon_like(size, max_val=128):
        """
        Generate a random Sidon-like sequence.
        We want sums to be distinct.
        """
        A = set()
        # Start with 0 to help cover lower sums? Or random?
        # Let's try random start, then add elements that don't create duplicate sums
        # To ensure coverage of [128, 256], we need large elements.
        
        # Strategy: Pick a random base, then greedily add elements
        # ensuring the set is "Sidon" (all pair sums distinct)
        # This is hard to enforce perfectly while covering range.
        # Instead, just generate a random set of size 'size' and hope local search fixes it.
        
        # Better: Generate a random permutation of [0, 128] and take first 'size' elements?
        # No, that's just random.
        
        # Let's use a structured random generation:
        # Pick 'size' distinct elements from [0, 128]
        # Bias towards ends to cover extremes?
        # Actually, uniform random is fine for local search to start with.
        
        elems = random.sample(range(0, 129), size)
        return set(elems)

    def generate_structured_candidate():
        """
        Generate a candidate that is likely to have low peak.
        Use two clusters? Or spread out?
        Spread out is better for low peak.
        """
        # Random spread
        size = random.randint(16, 25)
        # Ensure min <= 64 and max >= 128 for coverage potential
        # Actually, local search will fix it.
        # But a good start helps.
        
        # Try a geometric progression or similar?
        # Let's just use random sampling but force 0 and 128 in?
        # If 0 in A, then A itself covers sums.
        # If 128 in A, then 128+a covers 128..256 if a in 0..128.
        # So if 0 and 128 are in A, we need A to cover [0, 128] via sums?
        # No, sums are a+b.
        # If 0 in A, sums are a. So A must cover [128, 256] \ A?
        # If 0 in A, sum 128 requires 128 in A. Sum 129 requires 129 in A (impossible) or 1+128.
        
        # Let's just generate random sets of varying sizes.
        return generate_sidon_like(random.randint(18, 28))

    best_A = None
    best_fit = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    # We'll run multiple restarts with different random seeds and initial structures
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Different initialization strategies
        if r % 3 == 0:
            # Random
            A = generate_sidon_like(random.randint(18, 28))
        elif r % 3 == 1:
            # Force 0 and 128
            A = {0, 128}
            rest = random.sample(range(1, 128), random.randint(16, 24))
            A.update(rest)
        else:
            # Two blocks, but sparse
            # Low block
            low = random.sample(range(0, 30), random.randint(8, 12))
            # High block
            high = random.sample(range(100, 129), random.randint(8, 12))
            A = set(low) | set(high)
            
        # Local Search on this candidate
        current_A = A.copy()
        current_fit = fitness(current_A)
        
        if current_fit < best_fit:
            best_fit = current_fit
            best_A = current_A.copy()
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.995
        min_T = 0.1
        
        while T > min_T and time.time() < deadline - 5:
            # Generate neighbor
            move_type = random.random()
            new_A = current_A.copy()
            
            if move_type < 0.4:
                # Swap
                if len(current_A) > 5:
                    remove_elem = random.choice(list(current_A))
                    add_elem = random.randint(0, 128)
                    if add_elem not in current_A:
                        new_A.discard(remove_elem)
                        new_A.add(add_elem)
                    else:
                        continue
                else:
                    continue
            elif move_type < 0.7:
                # Add
                add_elem = random.randint(0, 128)
                if add_elem not in current_A:
                    new_A.add(add_elem)
                else:
                    continue
            else:
                # Remove
                if len(current_A) > 5:
                    remove_elem = random.choice(list(current_A))
                    new_A.discard(remove_elem)
                else:
                    continue
            
            new_fit = fitness(new_A)
            
            # Acceptance criterion
            delta = new_fit - current_fit
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                current_fit = new_fit
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_A = current_A.copy()
            else:
                # Rejection, maybe try a different move?
                pass
                
            T *= cooling
            
            # If we achieved perfect coverage and low peak, we can tighten
            if current_fit < 5:
                T = max(min_T, T * 0.95)

    # Final cleanup: if best_A is None, return seed
    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))
    
    # One last local search pass on the best found to polish
    current_A = best_A.copy()
    current_fit = fitness(current_A)
    
    # Quick hill climb
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single swaps/adds/removes?
        # Just random ones for speed
        for _ in range(500):
            if time.time() > deadline:
                break
            move_type = random.random()
            new_A = current_A.copy()
            
            if move_type < 0.5:
                if len(current_A) > 5:
                    remove_elem = random.choice(list(current_A))
                    add_elem = random.randint(0, 128)
                    if add_elem not in current_A:
                        new_A.discard(remove_elem)
                        new_A.add(add_elem)
                    else:
                        continue
            elif move_type < 0.75:
                add_elem = random.randint(0, 128)
                if add_elem not in current_A:
                    new_A.add(add_elem)
                else:
                    continue
            else:
                if len(current_A) > 5:
                    remove_elem = random.choice(list(current_A))
                    new_A.discard(remove_elem)
                else:
                    continue
            
            new_fit = fitness(new_A)
            if new_fit < current_fit:
                current_A = new_A
                current_fit = new_fit
                improved = True
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_A = current_A.copy()
            
    return sorted(best_A)
