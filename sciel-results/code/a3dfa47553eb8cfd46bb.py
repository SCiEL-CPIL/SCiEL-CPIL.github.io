import numpy as np
import time

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate divisors of 51975 (odd divisors > 1)
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    # Precompute masks for each (m, r) pair as Python integers (bitsets)
    # mask[m][r] is an integer where bit i is 1 if i % m == r
    # This allows fast bitwise operations:
    #   uncovered & ~mask[m][r] -> new uncovered set
    #   bin(uncovered & mask[m][r]).count('1') -> gain (but slow for large ints)
    # Better: use numpy for counting? Or just track uncovered count incrementally?
    # Actually, for greedy, we need to find the (m, r) that maximizes |uncovered & mask[m][r]|.
    # With bitsets, we can compute this intersection.
    
    # However, storing all masks as Python ints:
    # Number of (m, r) pairs = sum(m for m in divisors) ~ 120k.
    # Each int is ~52k bits = 6.5 KB.
    # Total memory ~ 120k * 6.5 KB ~ 780 MB. This might be too much or slow to initialize.
    
    # Alternative: Use numpy uint64 arrays for bitsets?
    # 51975 bits requires ceil(51975/64) = 813 uint64 words.
    # 120k masks * 813 words * 8 bytes ~ 780 MB. Same issue.
    
    # Let's stick to the current approach but optimize the inner loop.
    # The current program uses numpy boolean arrays.
    # The inspiration suggests using bitsets for faster updates.
    # But the bottleneck is finding the best (m, r).
    
    # Let's try a hybrid:
    # 1. Use numpy boolean array for `covered`.
    # 2. For the greedy step, instead of scanning all r for all m,
    #    we can use a faster method if we precompute something?
    
    # Actually, the current program's greedy step is:
    # for m in available:
    #   for r in range(m):
    #     count = np.count_nonzero(~covered[r::m])
    # This is O(sum(m) * N/m) = O(N * sum(1)) = O(N * num_moduli) per step?
    # No, for a fixed m, summing over r: sum_{r=0}^{m-1} (N/m) = N.
    # So for each m, it takes O(N) time to find the best r.
    # Total per step: O(N * num_available_moduli).
    # With 47 moduli and N=52k, that's ~2.4M ops per step.
    # 47 steps -> ~100M ops.
    # Numpy count_nonzero is fast, but the loop over r is in Python.
    
    # Optimization idea:
    # Instead of looping over r in Python, can we vectorize?
    # For a fixed m, we want to compute counts of uncovered in each residue class.
    # We can do:
    #   uncovered_flat = ~covered.ravel()
    #   # Reshape to (-1, m)? No, because the stride is m, not contiguous.
    #   # x % m == r means x is in {r, r+m, r+2m, ...}
    #   # This is not a simple reshape.
    
    # Let's try to use the bitset idea but only for the update, not the search.
    # Or, let's just make the current code as fast as possible.
    
    # One optimization: if the gain is 0, we can stop.
    # Also, we can sort moduli by size? Larger moduli cover fewer residues per class,
    # but we have to pick one class per modulus.
    # Actually, greedy usually picks the largest gain.
    
    # Let's implement the bitset approach for the *update* and see if it helps.
    # But the search is the bottleneck.
    
    # Alternative: Use the "inspiration" strategy of random restarts with coordinate descent.
    # The current program does greedy + local search (swaps).
    # The inspiration does greedy (fixed order) + coordinate descent (change r for each m).
    # Coordinate descent might be better for polishing.
    
    # Let's combine:
    # 1. Greedy initialization (fast, using numpy).
    # 2. Coordinate descent: for each m, try all r and pick the best.
    # 3. Swaps: try to swap a modulus with an unselected one.
    
    # To speed up coordinate descent, we can use the same `find_best_residue` function.
    
    covered = np.zeros(N, dtype=bool)
    classes = []
    available_moduli = divisors.copy()
    
    # 1. Greedy Initialization
    while available_moduli:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in available_moduli:
            max_count = 0
            best_r_m = 0
            for r in range(m):
                c = np.count_nonzero(~covered[r::m])
                if c > max_count:
                    max_count = c
                    best_r_m = r
            
            if max_count > best_gain:
                best_gain = max_count
                best_m = m
                best_r = best_r_m
        
        if best_m is None or best_gain == 0:
            break
            
        classes.append([best_m, best_r])
        covered[best_r::best_m] = True
        available_moduli.remove(best_m)
        
        if np.all(covered):
            break

    # 2. Local Search: Coordinate Descent + Swaps
    selected_moduli = {c[0] for c in classes}
    unselected_moduli = [m for m in divisors if m not in selected_moduli]
    
    def find_best_residue(m, covered_arr):
        max_count = 0
        best_r = 0
        for r in range(m):
            c = np.count_nonzero(~covered_arr[r::m])
            if c > max_count:
                max_count = c
                best_r = r
        return best_r, max_count

    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_uncovered = np.count_nonzero(~covered)
        
        # A. Coordinate Descent: For each selected m, try to find a better r
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m = classes[i][0]
            r_old = classes[i][1]
            
            # Temporarily remove
            covered[r_old::m] = False
            uncovered_after_removal = np.count_nonzero(~covered)
            
            # Find best new r
            r_new, gain_new = find_best_residue(m, covered)
            
            # Apply new r
            covered[r_new::m] = True
            uncovered_after_add = np.count_nonzero(~covered)
            
            # Check if improvement
            if uncovered_after_add < current_uncovered:
                classes[i][1] = r_new
                current_uncovered = uncovered_after_add
                improved = True
            else:
                # Revert
                covered[r_new::m] = False
                covered[r_old::m] = True
            
            if current_uncovered == 0:
                return classes

        # B. Swaps: Try to swap a selected modulus with an unselected one
        for i, m_out in enumerate(selected_moduli):
            if time.time() > deadline:
                break
            
            r_out = None
            idx_out = None
            for j, c in enumerate(classes):
                if c[0] == m_out:
                    r_out = c[1]
                    idx_out = j
                    break
            
            # Temporarily remove m_out
            covered[r_out::m_out] = False
            new_uncovered = np.count_nonzero(~covered)
            
            best_swap_gain = 0
            best_m_in = None
            best_r_in = None
            
            for m_in in unselected_moduli:
                r_in, gain_in = find_best_residue(m_in, covered)
                if gain_in > best_swap_gain:
                    best_swap_gain = gain_in
                    best_m_in = m_in
                    best_r_in = r_in
            
            # Check if swap is beneficial
            lost = new_uncovered - current_uncovered
            if best_m_in is not None and best_swap_gain > lost:
                # Perform swap
                classes[idx_out][0] = best_m_in
                classes[idx_out][1] = best_r_in
                covered[best_r_in::best_m_in] = True
                
                selected_moduli.remove(m_out)
                selected_moduli.add(best_m_in)
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(m_out)
                
                current_uncovered = np.count_nonzero(~covered)
                improved = True
                if current_uncovered == 0:
                    return classes
            else:
                # Revert
                covered[r_out::m_out] = True

        # C. Add unselected moduli if they help
        for m in unselected_moduli[:]:
            if time.time() > deadline:
                break
            r, gain = find_best_residue(m, covered)
            if gain > 0:
                classes.append([m, r])
                covered[r::m] = True
                unselected_moduli.remove(m)
                selected_moduli.add(m)
                current_uncovered = np.count_nonzero(~covered)
                improved = True
                if current_uncovered == 0:
                    return classes

    return classes
