import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    
    # Precompute FFT size
    n_fft = 1
    while n_fft < 2 * (max_sum + 1):
        n_fft *= 2

    def get_counts_fast(A_set):
        """
        Fast count of unordered pair sums for a set A.
        Returns a numpy array of counts for sums 0..max_sum.
        """
        f = np.zeros(max_sum + 1, dtype=float)
        for x in A_set:
            if 0 <= x <= max_sum:
                f[x] = 1.0
        
        F = np.fft.rfft(f, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)
        
        counts = np.zeros(max_sum + 1, dtype=int)
        # Vectorized logic for unordered counts
        # ordered = conv[s]
        # diag = 1 if s even and s/2 in A else 0
        # unordered = (ordered + diag) // 2
        
        s_vals = np.arange(max_sum + 1)
        ordered = np.round(conv[:max_sum+1]).astype(int)
        
        # Determine which sums have a diagonal term (a=a => 2a=s)
        # s must be even, and s/2 must be in A
        even_mask = (s_vals % 2 == 0)
        half_vals = s_vals // 2
        
        # Check if half_vals are in A
        # Create a boolean array for membership in A
        in_A = np.zeros(max_sum + 1, dtype=bool)
        for x in A_set:
            if 0 <= x <= max_sum:
                in_A[x] = True
        
        diag = even_mask & in_A[half_vals]
        
        counts = (ordered + diag.astype(int)) // 2
        return counts

    def get_cost(A_set):
        """
        Calculate the cost: 
        1. If any target sum is uncovered, return 1000 + number of uncovered.
        2. Otherwise, return the max multiplicity in the target range.
        """
        counts = get_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum+1]
        
        if np.any(target_counts == 0):
            uncovered_count = int(np.sum(target_counts == 0))
            return 1000 + uncovered_count
        
        return int(np.max(target_counts))

    def local_search(A_init, time_limit):
        """
        Fast local search to polish the solution.
        Uses greedy construction for coverage, then swap-based optimization for peak.
        """
        A_set = set(A_init)
        current_cost = get_cost(A_set)
        best_cost = current_cost
        best_A = set(A_set)
        all_elements = set(range(N + 1))
        
        # Phase 1: Ensure coverage
        # If not covered, greedily add elements that cover the most uncovered sums
        while time.time() < time_limit:
            counts = get_counts_fast(A_set)
            target_counts = counts[min_sum:max_sum+1]
            uncovered = np.where(target_counts == 0)[0]
            
            if len(uncovered) == 0:
                break
            
            # Find best x to add
            best_x = None
            best_gain = 0
            uncovered_set = set(uncovered.tolist() + min_sum) # Sums in [128, 256]
            
            # We can optimize this by checking all candidates
            for x in all_elements - A_set:
                gain = 0
                # New sums formed by x are x + y for y in A_set
                # We only care if they are in uncovered_set
                # Since A_set is smallish, this is fast
                for y in A_set:
                    s = x + y
                    if s in uncovered_set:
                        gain += 1
                # Also x+x
                if x + x in uncovered_set:
                    gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is not None and best_gain > 0:
                A_set.add(best_x)
                current_cost = get_cost(A_set)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A_set)
            else:
                # No improvement possible with single addition?
                # Try random addition to break symmetry or stuck state
                candidates = list(all_elements - A_set)
                if candidates:
                    A_set.add(random.choice(candidates))
                    current_cost = get_cost(A_set)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A_set)
                else:
                    break

        # Phase 2: Minimize peak multiplicity
        # Now that we have coverage (hopefully), we swap elements to reduce the max count
        # We want to keep coverage but lower the peak.
        
        # If we don't have coverage, we can't really do Phase 2 properly without losing validity.
        # But get_cost penalizes coverage loss heavily, so swaps that break coverage will be rejected.
        
        while time.time() < time_limit:
            if best_cost <= 2:
                break
            
            improved = False
            A_list = list(A_set)
            random.shuffle(A_list)
            
            for x_in in A_list:
                if time.time() >= time_limit:
                    break
                
                candidates = list(all_elements - A_set)
                if not candidates:
                    continue
                
                # Limit candidates to speed up, but try a decent number
                random.shuffle(candidates)
                for x_out in candidates[:50]:
                    A_test = A_set.copy()
                    A_test.discard(x_in)
                    A_test.add(x_out)
                    
                    new_cost = get_cost(A_test)
                    
                    if new_cost < current_cost:
                        A_set = A_test
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A_set)
                        break
                    elif new_cost == current_cost and random.random() < 0.1:
                        # Accept equal cost to escape local minima
                        A_set = A_test
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A_set)
                        break

            if not improved:
                # Try a random move to escape
                if random.random() < 0.5 and A_set:
                    x_in = random.choice(list(A_set))
                    candidates = list(all_elements - A_set)
                    if candidates:
                        x_out = random.choice(candidates)
                        A_test = A_set.copy()
                        A_test.discard(x_in)
                        A_test.add(x_out)
                        new_cost = get_cost(A_test)
                        if new_cost <= current_cost:
                            A_set = A_test
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A_set)
                else:
                    # Small random perturbation: add a random element (increases size, might help coverage or balance)
                    candidates = list(all_elements - A_set)
                    if candidates:
                        x_out = random.choice(candidates)
                        A_test = A_set.copy()
                        A_test.add(x_out)
                        new_cost = get_cost(A_test)
                        if new_cost <= current_cost:
                            A_set = A_test
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A_set)

        return best_A, best_cost

    random.seed(42)
    np.random.seed(42)
    
    best_overall_A = None
    best_overall_cost = float('inf')
    
    # Strategy:
    # 1. Start with a few random initializations of different sizes.
    # 2. Use local search (greedy coverage + swap optimization) on each.
    # 3. Keep the best result.
    
    # Sizes to try: 
    # Minimum size for coverage: k^2/2 >= 129 => k >= 16.
    # Let's try 18, 20, 25, 30.
    
    sizes = [18, 20, 25, 30]
    
    # Time budget per restart
    num_restarts = 4
    time_per_restart = (deadline - start_time - 5) / num_restarts # Leave 5s for safety
    
    for i in range(num_restarts):
        if time.time() >= deadline - 5:
            break
        
        size = sizes[i % len(sizes)]
        # Ensure size is within [0, N+1]
        size = min(size, N + 1)
        
        A_init = set(random.sample(range(0, N + 1), size))
        
        # Run local search
        A_refined, cost = local_search(A_init, min(deadline - 5, time.time() + time_per_restart))
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_A = A_refined
            
        # Early exit if we found a perfect solution (peak 1 or 2 is very good, 1 is likely impossible for all sums)
        if best_overall_cost <= 2:
            break

    # Fallback if something went wrong
    if best_overall_A is None:
        # Construct a valid set, e.g., all even numbers or similar
        # A = {0, 2, ..., 128} covers even sums. 
        # To cover all, we need more.
        # Let's just return a random set that is likely valid if we are stuck, 
        # but the task requires validity. 
        # A simple valid set: A = [0, 1, ..., 128]
        best_overall_A = set(range(0, N + 1))
        best_overall_cost = get_cost(best_overall_A)

    return list(best_overall_A)

if __name__ == "__main__":
    # Quick test
    A = construct()
    print(f"Found set of size {len(A)}")
    # Verify
    counts = np.zeros(257, dtype=int)
    for i, a in enumerate(A):
        for b in A[i:]:
            counts[a+b] += 1
    target_counts = counts[128:257]
    if np.any(target_counts == 0):
        print("Invalid: Uncovered sums found")
        print(np.where(target_counts == 0)[0] + 128)
    else:
        print(f"Valid. Peak multiplicity: {np.max(target_counts)}")
