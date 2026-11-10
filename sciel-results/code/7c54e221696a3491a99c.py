import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 128
    TARGETS = list(range(128, 257))
    NUM_TARGETS = len(TARGETS)

    def compute_multiplicities(A_bool):
        """
        A_bool: numpy array of length 2*N+1 with 1s where A has elements.
        Returns numpy array of unordered-pair-with-replacement multiplicities for t=0..2*N.
        """
        conv = np.convolve(A_bool, A_bool)
        # conv[t] = number of ordered pairs (a,b) in A x A with a+b = t
        # Unordered with replacement: m(t) = (conv[t] + diag[t]) // 2
        # where diag[t] = 1 if t even and t//2 in A, else 0
        # Actually: conv[t] counts (a,b) and (b,a) separately.
        # For t even, if t//2 in A, then (t//2, t//2) is counted once in conv.
        # So unordered count = (conv[t] - diag_count) / 2 + diag_count = (conv[t] + diag_count) / 2
        # where diag_count = 1 if t even and t//2 in A, else 0.
        
        # Efficient computation:
        # diag[t] = 1 if t % 2 == 0 and A_bool[t//2] == 1
        # We can vectorize this.
        t_range = np.arange(2 * N + 1)
        even_mask = (t_range % 2 == 0)
        # For even t, check if t//2 is in A
        half_indices = t_range // 2
        diag = np.zeros(2 * N + 1, dtype=int)
        diag[even_mask] = A_bool[half_indices[even_mask]]
        
        mults = (conv + diag) // 2
        return mults

    def get_A_bool(A_set):
        A_bool = np.zeros(2 * N + 1, dtype=int)
        for a in A_set:
            A_bool[a] = 1
        return A_bool

    def is_complete(mults):
        return np.all(mults[128:257] > 0)

    def get_max_mult(mults):
        return int(np.max(mults[128:257]))

    # Strategy: Use a structured initial set based on arithmetic progressions,
    # then aggressively optimize with local search (remove/swap operations).
    # Key insight: small sets (k~16-25) with well-chosen elements can achieve
    # low peak multiplicity (3-4) while maintaining coverage.

    best_A = None
    best_mult = float('inf')

    # Precompute all possible candidate sets
    # We'll use a structured initialization: take a subset of [0, 128]
    # that is a union of a few arithmetic progressions.

    def make_structured_seed(seed_id):
        """Create a structured seed set with low initial multiplicity."""
        # Use a combination of approaches:
        # 1. Random subset of a sparse grid
        # 2. Two interleaved progressions
        # 3. Random with bias toward extremes
        
        rng = random.Random(seed_id)
        
        approach = rng.randint(0, 3)
        
        if approach == 0:
            # Sparse random: pick from a subset of [0,128]
            # Bias toward values that cover the target range well
            # For coverage of [128,256], we need elements that sum to those ranges
            # A element x covers targets [x+min(A), x+max(A)]
            # Best to have elements spread across [0, 128]
            k = rng.randint(16, 24)
            # Pick k elements from [0, 128] with some structure
            # Use a stride-based selection with random offsets
            stride = rng.choice([2, 3, 4, 5])
            base = rng.randint(0, stride)
            candidates = list(range(base, N + 1, stride))
            if len(candidates) < k:
                # Fill with random elements
                remaining = [x for x in range(N + 1) if x not in set(candidates)]
                rng.shuffle(remaining)
                candidates.extend(remaining)
            candidates = candidates[:k]
            # Add some random elements to ensure coverage
            if rng.random() < 0.5:
                extra = rng.randint(0, min(4, N + 1 - k))
                for _ in range(extra):
                    x = rng.randint(0, N)
                    if x not in candidates:
                        candidates.append(x)
            return set(candidates)
            
        elif approach == 1:
            # Two interleaved progressions
            k1 = rng.randint(8, 14)
            k2 = rng.randint(6, 12)
            stride1 = rng.choice([3, 4, 5])
            stride2 = rng.choice([3, 4, 5])
            base1 = rng.randint(0, stride1)
            base2 = rng.randint(0, stride2)
            set1 = set(range(base1, N + 1, stride1))
            set2 = set(range(base2, N + 1, stride2))
            # Take subsets
            elems1 = list(set1)
            elems2 = list(set2)
            rng.shuffle(elems1)
            rng.shuffle(elems2)
            result = set(elems1[:k1]) | set(elems2[:k2])
            return result
            
        elif approach == 2:
            # Concentrated in two regions: [0, 40] and [88, 128]
            # This helps cover both low and high ends of the target range
            k1 = rng.randint(8, 14)
            k2 = rng.randint(8, 14)
            region1 = rng.randint(0, 40)
            region2 = rng.randint(88, 128)
            set1 = set(rng.sample(range(max(0, region1 - 20), min(N + 1, region1 + 21)), 
                                  min(k1, 41)))
            set2 = set(rng.sample(range(max(0, region2 - 20), min(N + 1, region2 + 21)), 
                                  min(k2, 41)))
            return set1 | set2
            
        else:
            # Pure random with slight structure
            k = rng.randint(16, 24)
            # Use a mixture: 60% random, 40% from a progression
            base = rng.randint(0, N)
            stride = rng.choice([2, 3, 4])
            prog = list(range(base, N + 1, stride))
            rng.shuffle(prog)
            result = set(prog[:max(0, k // 2)])
            remaining = [x for x in range(N + 1) if x not in result]
            rng.shuffle(remaining)
            result.update(remaining[:k - len(result)])
            return result

    def greedy_complete(A_set):
        """Add elements to A_set to ensure full coverage of [128, 256]."""
        max_adds = 30
        for _ in range(max_adds):
            if time.time() > deadline - 5:
                break
            A_bool = get_A_bool(A_set)
            mults = compute_multiplicities(A_bool)
            if is_complete(mults):
                break
            
            uncovered = [t for t in TARGETS if mults[t] == 0]
            if not uncovered:
                break
            
            # Find best element to add
            best_gain = 0
            best_elem = None
            candidates = [x for x in range(N + 1) if x not in A_set]
            
            # Sample a subset of candidates for speed
            if len(candidates) > 30:
                sample_size = 30
                sampled = random.sample(candidates, sample_size)
            else:
                sampled = candidates
            
            for x in sampled:
                gain = 0
                for t in uncovered:
                    y = t - x
                    if 0 <= y <= N and y in A_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
                    
            if best_gain > 0 and best_elem is not None:
                A_set.add(best_elem)
            elif candidates:
                # Add random element
                A_set.add(random.choice(candidates))
            else:
                break
        
        # Final check: force completeness if needed
        A_bool = get_A_bool(A_set)
        mults = compute_multiplicities(A_bool)
        if not is_complete(mults):
            for t in TARGETS:
                if mults[t] == 0:
                    # Find any pair that sums to t
                    for a in range(t // 2 + 1):
                        b = t - a
                        if 0 <= a <= N and 0 <= b <= N:
                            A_set.add(a)
                            A_set.add(b)
                            break
                    mults = compute_multiplicities(get_A_bool(A_set))
        
        return A_set

    def local_search_remove(A_set, max_steps=200):
        """Try to remove elements to reduce peak multiplicity."""
        current_max = get_max_mult(compute_multiplicities(get_A_bool(A_set)))
        
        for _ in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            if current_max <= 2:
                break
            
            # Try removing each element
            best_new_max = current_max
            best_elem_to_remove = None
            
            A_list = list(A_set)
            random.shuffle(A_list)
            
            for elem in A_list:
                if time.time() > deadline - 3:
                    break
                new_A = set(A_set)
                new_A.remove(elem)
                if not new_A:
                    continue
                
                A_bool = get_A_bool(new_A)
                mults = compute_multiplicities(A_bool)
                
                if not is_complete(mults):
                    continue
                    
                new_max = get_max_mult(mults)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_elem_to_remove = elem
            
            if best_elem_to_remove is not None and best_new_max < current_max:
                A_set.remove(best_elem_to_remove)
                current_max = best_new_max
            else:
                # No improvement possible with removals, try a swap
                break
                
        return A_set, current_max

    def local_search_swap(A_set, max_steps=300):
        """Try swaps to reduce peak multiplicity."""
        current_max = get_max_mult(compute_multiplicities(get_A_bool(A_set)))
        
        # Temperature for simulated annealing
        temp = 5.0
        cooling = 0.97
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            if current_max <= 2:
                break
            
            # Pick a random element to swap out
            A_list = list(A_set)
            if not A_list:
                break
            elem_out = random.choice(A_list)
            
            # Pick a random element to swap in
            candidates = [x for x in range(N + 1) if x not in A_set]
            if not candidates:
                break
            elem_in = random.choice(candidates)
            
            new_A = set(A_set)
            new_A.remove(elem_out)
            new_A.add(elem_in)
            
            A_bool = get_A_bool(new_A)
            mults = compute_multiplicities(A_bool)
            
            if not is_complete(mults):
                temp *= cooling
                continue
                
            new_max = get_max_mult(mults)
            
            # Acceptance criterion
            if new_max < current_max:
                A_set = new_A
                current_max = new_max
            elif new_max == current_max:
                if random.random() < 0.3:
                    A_set = new_A
            else:
                delta = new_max - current_max
                prob = np.exp(-delta / temp)
                if random.random() < prob:
                    A_set = new_A
                    current_max = new_max
            
            temp *= cooling
            
        return A_set, current_max

    # Main loop: try multiple seeds and optimize each
    num_attempts = 30
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 5:
            break
        
        # Create structured seed
        seed_id = attempt * 12345 + 67890
        A_set = make_structured_seed(seed_id)
        
        # Ensure non-empty
        if not A_set:
            A_set = {0, N}
        
        # Greedy complete
        A_set = greedy_complete(A_set)
        
        if not A_set:
            continue
        
        # Check if complete
        A_bool = get_A_bool(A_set)
        mults = compute_multiplicities(A_bool)
        if not is_complete(mults):
            # Force complete
            for t in TARGETS:
                if mults[t] == 0:
                    a = t // 2
                    b = t - a
                    A_set.add(a)
                    A_set.add(b)
            mults = compute_multiplicities(get_A_bool(A_set))
        
        current_max = get_max_mult(mults)
        
        # Local search: first try removals, then swaps
        A_set, current_max = local_search_remove(A_set, max_steps=100)
        A_set, current_max = local_search_swap(A_set, max_steps=200)
        
        if current_max < best_mult:
            best_mult = current_max
            best_A = set(A_set)
            
            # If we found a very good solution, try to improve it further
            if best_mult <= 3 and time.time() < deadline - 10:
                # Additional optimization on the best solution
                A_set = set(best_A)
                A_set, improved_max = local_search_swap(A_set, max_steps=300)
                if improved_max < best_mult:
                    best_mult = improved_max
                    best_A = set(A_set)

    if best_A is None:
        # Fallback: use a known-good construction
        # A = [0, 2, 4, ..., 128] gives coverage but high multiplicity
        # Better: a sparse set
        best_A = set(range(0, N + 1, 4))  # 33 elements
        # Verify coverage
        A_bool = get_A_bool(best_A)
        mults = compute_multiplicities(A_bool)
        if not is_complete(mults):
            # Add elements to fix coverage
            best_A = greedy_complete(best_A)
    
    return list(best_A)
