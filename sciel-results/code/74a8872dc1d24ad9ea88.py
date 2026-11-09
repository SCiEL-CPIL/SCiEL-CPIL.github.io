"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # 15s safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def check_triple_sums_unique(arr):
        """Check if all unordered triples (with replacement) have unique sums using NumPy."""
        arr = np.array(arr)
        n = len(arr)
        if n <= 1:
            return True
        # Generate all unordered triples (i,j,k) with i<=j<=k
        # For efficiency, compute sums in batches
        sums = np.zeros((n, n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums[i, j, k] = arr[i] + arr[j] + arr[k]
        # Check for duplicates among the valid entries
        valid_sums = []
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    valid_sums.append(sums[i, j, k])
        valid_sums = np.array(valid_sums)
        return len(np.unique(valid_sums)) == len(valid_sums)

    def fast_check_add(arr, x, existing_sums=None):
        """Fast check if x can be added to arr without creating duplicate triple sums.
        If existing_sums is provided, use it instead of recomputing."""
        n = len(arr)
        if existing_sums is None:
            existing_sums = set()
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        existing_sums.add(arr[i] + arr[j] + arr[k])
        
        # Check new triples involving x
        # (x,x,x)
        if 3*x in existing_sums:
            return False
        # (x,x,y) for y in arr
        for y in arr:
            if 2*x + y in existing_sums:
                return False
        # (x,y,z) for y<=z in arr
        for i in range(n):
            for j in range(i, n):
                if x + arr[i] + arr[j] in existing_sums:
                    return False
        return True

    def build_greedy_ordering(order, max_size=MAX_SIZE):
        """Greedy: iterate through order, add element if triple sums remain unique."""
        selected = []
        triple_sums = set()
        
        for x in order:
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 10:
                break
            
            # Check if x can be added
            valid = True
            new_sums = set()
            
            # (x, x, x)
            s = 3 * x
            if s in triple_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                # (x, x, y) for each y in selected
                for y in selected:
                    s = 2 * x + y
                    if s in triple_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, z) for y <= z in selected
                n_sel = len(selected)
                for i in range(n_sel):
                    for j in range(i, n_sel):
                        s = x + selected[i] + selected[j]
                        if s in triple_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected

    def local_improvement(arr, max_size=MAX_SIZE):
        """Try to improve by swapping elements or adding new ones."""
        if not arr:
            return arr
        
        arr = list(arr)
        current_size = len(arr)
        arr_set = set(arr)
        
        # Precompute existing triple sums
        triple_sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    triple_sums.add(arr[i] + arr[j] + arr[k])
        
        # Try to add more elements
        for x in range(1, N + 1):
            if len(arr) >= max_size:
                break
            if time.time() > DEADLINE - 10:
                break
            if x in arr_set:
                continue
            
            if fast_check_add(arr, x, triple_sums):
                arr.append(x)
                arr_set.add(x)
                # Update triple sums
                n_new = len(arr)
                for i in range(n):
                    for j in range(i, n):
                        triple_sums.add(x + arr[i] + arr[j])
                for i in range(n):
                    triple_sums.add(2*x + arr[i])
                triple_sums.add(3*x)
                current_size += 1
        
        # Try swapping: replace one element with another to potentially allow more additions
        for _ in range(20):
            if time.time() > DEADLINE - 10:
                break
            improved = False
            for i in range(len(arr)):
                if time.time() > DEADLINE - 10:
                    break
                # Try swapping arr[i] with various candidates - finer granularity
                for x in range(1, N + 1, 10):
                    if x in arr_set or x == arr[i]:
                        continue
                    candidate = arr[:i] + [x] + arr[i+1:]
                    candidate.sort()
                    cand_set = set(candidate)
                    
                    # Quick check: verify uniqueness
                    cand_triple_sums = set()
                    nc = len(candidate)
                    valid = True
                    for a in range(nc):
                        for b in range(a, nc):
                            for c in range(b, nc):
                                s = candidate[a] + candidate[b] + candidate[c]
                                if s in cand_triple_sums:
                                    valid = False
                                    break
                                cand_triple_sums.add(s)
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Try to extend this candidate
                    for y in range(1, N + 1):
                        if len(candidate) >= max_size:
                            break
                        if time.time() > DEADLINE - 10:
                            break
                        if y in cand_set:
                            continue
                        if fast_check_add(candidate, y, cand_triple_sums):
                            candidate.append(y)
                            cand_set.add(y)
                            # Update triple sums
                            for a in range(len(candidate)-1):
                                for b in range(a, len(candidate)-1):
                                    cand_triple_sums.add(y + candidate[a] + candidate[b])
                            for a in range(len(candidate)-1):
                                cand_triple_sums.add(2*y + candidate[a])
                            cand_triple_sums.add(3*y)
                    
                    if len(candidate) > len(arr):
                        arr = candidate
                        arr_set = set(arr)
                        triple_sums = cand_triple_sums
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return arr

    # Strategy 1: Base-3 inspired construction
    # Elements with well-separated base-3 digits guarantee unique triple sums
    # a_i = c * 3^i gives unique sums, but we need to compress to [1, 4096]
    # Try: use elements where base-3 representation has 1s in well-separated positions
    
    def base3_sparsity_score(x):
        """Score based on how sparse the base-3 representation is."""
        digits = []
        while x > 0:
            digits.append(x % 3)
            x //= 3
        if not digits:
            return 0
        # Count zeros between non-zero digits
        non_zero_positions = [i for i, d in enumerate(digits) if d != 0]
        if len(non_zero_positions) <= 1:
            return 100 - len(digits)  # Prefer shorter representations
        # Max gap between consecutive non-zero digits
        max_gap = 0
        for i in range(1, len(non_zero_positions)):
            gap = non_zero_positions[i] - non_zero_positions[i-1] - 1
            max_gap = max(max_gap, gap)
        return max_gap * 10 - len(digits)
    
    # Strategy 1: Greedy with base-3 sparsity ordering
    order1 = sorted(range(1, N + 1), key=lambda x: -base3_sparsity_score(x))
    best = build_greedy_ordering(order1)
    best_size = len(best)
    
    # Strategy 2: Simple greedy from 1
    if time.time() < DEADLINE - 15:
        candidate = build_greedy_ordering(range(1, N + 1))
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Strategy 3: Greedy with exponential spacing preference
    if time.time() < DEADLINE - 15:
        # Prefer numbers that are far apart in log scale
        def log_spacing_score(x):
            # Score based on how well-spaced x is from powers of 3
            best_dist = float('inf')
            for i in range(1, 13):
                p = 3 ** i
                if p <= N:
                    best_dist = min(best_dist, abs(x - p))
            return -best_dist  # Prefer numbers close to powers of 3
        
        order3 = sorted(range(1, N + 1), key=lambda x: log_spacing_score(x))
        candidate = build_greedy_ordering(order3)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Strategy 4: Local improvement
    if time.time() < DEADLINE - 10:
        improved = local_improvement(best)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Strategy 5: Try multiple random restarts with different orderings
    if time.time() < DEADLINE - 10:
        import random
        random.seed(42)
        for trial in range(8):
            if time.time() > DEADLINE - 10:
                break
            # Create a randomized ordering with bias toward well-spread values
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Add some structure: interleave small and large values
            small_half = [x for x in order if x <= N // 2]
            large_half = [x for x in order if x > N // 2]
            interleaved = []
            i = j = 0
            while i < len(small_half) or j < len(large_half):
                if i < len(small_half):
                    interleaved.append(small_half[i])
                    i += 1
                if j < len(large_half):
                    interleaved.append(large_half[j])
                    j += 1
            order = interleaved
            candidate = build_greedy_ordering(order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(best)
    
    # Final local improvement if time permits
    if time.time() < DEADLINE - 5:
        improved = local_improvement(best)
        if len(improved) > best_size:
            best = improved
    
    # One more pass of local improvement with different random seed
    if time.time() < DEADLINE - 3:
        import random
        random.seed(123)
        # Try a few more random restarts focused on the current best
        for trial in range(3):
            if time.time() > DEADLINE - 3:
                break
            # Perturb the current best: remove a random element, try to rebuild
            if len(best) > 5:
                idx = random.randint(0, len(best) - 1)
                perturbed = best[:idx] + best[idx+1:]
                # Try to rebuild from this perturbed set
                perturbed_set = set(perturbed)
                # Add elements greedily
                for x in range(1, N + 1):
                    if len(perturbed) >= MAX_SIZE:
                        break
                    if time.time() > DEADLINE - 3:
                        break
                    if x in perturbed_set:
                        continue
                    # Quick conflict check
                    n_p = len(perturbed)
                    valid = True
                    for i in range(n_p):
                        for j in range(i, n_p):
                            if x + perturbed[i] + perturbed[j] in [perturbed[a] + perturbed[b] + perturbed[c] for a in range(n_p) for b in range(a, n_p) for c in range(b, n_p)]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        perturbed.append(x)
                        perturbed_set.add(x)
                if len(perturbed) > best_size:
                    best = perturbed
                    best_size = len(best)
    
    return best
# EVOLVE-BLOCK-END
